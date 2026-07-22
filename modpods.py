import re
import warnings
from collections import OrderedDict
from typing import Any

import control
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pysindy as ps
import scipy.signal as signal
import scipy.stats as stats
from pysindy.optimizers._constrained_sr3 import ConstrainedSR3 as _ConstrainedSR3
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Suppress the specific AxesWarning from pysindy after import
warnings.filterwarnings(
    "ignore", message=".*axes labeled for array with.*", module="pysindy"
)


# Bayesian optimization helper functions
def _expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    """Expected Improvement acquisition function for Bayesian optimization."""
    mu, sigma = gpr.predict(X, return_std=True)
    mu = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)

    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide="warn"):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def _propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=10):
    """Propose next sampling point by optimizing acquisition function."""
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr).flatten()

    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method="L-BFGS-B")
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x

    return min_x.reshape(-1, 1)


# =============================================================================
# Transform Cache - memoizes single-input gamma transforms to avoid recomputation
# =============================================================================


class TransformCache:
    """LRU cache for gamma-transformed time series.

    Caches results of convolving a forcing series with a gamma PDF kernel.
    Keys are quantized (input_name, shape, scale, loc) tuples so near-identical
    parameter sets reuse cached results.
    """

    def __init__(self, max_entries: int = 2000, quantization: float = 1e-6):
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self.max_entries = max_entries
        self.quantization = quantization
        self.hits = 0
        self.misses = 0

    def _quantize(self, value: float) -> float:
        """Quantize a float to reduce near-duplicate keys."""
        if self.quantization <= 0:
            return value
        return round(value / self.quantization) * self.quantization

    def _make_key(
        self, input_name: str, n: int, shape: float, scale: float, loc: float
    ) -> tuple:
        """Create a hashable cache key from input name and gamma params."""
        return (
            input_name,
            n,
            self._quantize(shape),
            self._quantize(scale),
            self._quantize(loc),
        )

    def get(
        self,
        input_name: str,
        forcing_values: np.ndarray,
        shape: float,
        scale: float,
        loc: float,
    ) -> np.ndarray:
        """Get cached transform or compute and cache it.

        Returns a COPY of the cached array to prevent mutation issues.
        """
        n = len(forcing_values)
        key = self._make_key(input_name, n, shape, scale, loc)

        if key in self._cache:
            self.hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key].copy()

        # Cache miss - compute the transform using FFT convolution
        self.misses += 1
        shape_time = np.arange(0, n, 1)
        gamma_kernel = stats.gamma.pdf(shape_time, shape, scale=scale, loc=loc)
        result = signal.fftconvolve(forcing_values, gamma_kernel, mode="full")[:n]

        # Store in cache
        self._cache[key] = result

        # Evict oldest if over capacity
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

        return result.copy()  # type: ignore[no-any-return]

    def clear(self):
        """Clear the cache and reset counters."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_entries": self.max_entries,
        }

    def __repr__(self):
        s = self.stats()
        return f"TransformCache(hits={s['hits']}, misses={s['misses']}, hit_rate={s['hit_rate']:.2%}, size={s['size']})"


# Global cache instance used throughout the module
_transform_cache = TransformCache(max_entries=2000, quantization=1e-6)


# delay model builds differential equations relating the dependent variables to transformations of all the variables
# if there are no independent variables, then dependent_columns should be a list of all the columns in the dataframe
# and independent_columns should be an empty list
# by default, only the independent variables are transformed, but if transform_dependent is set to True, then the dependent variables are also transformed
# REQUIRES:
# a pandas dataframe,
# the column names of the dependent and indepdent variables,
# the number of timesteps to "wind up" the latent states,
# the initial number of transformations to use in the optimization,
# the maximum number of transformations to use in the optimization,
# the maximum number of iterations to use in the optimization
# and the order of the polynomial to use in the optimization
# bibo_stable: if true, the highest order output autocorrelation term is constrained to be negative
# RETURNS:
# models for each number of transformations from min to max
# NOTE: this code works for MIMO models, however, if output variables are dependent on each other
# poor simulation fidelity is likely due to their errors contributing to each other
# if the learned dynamics are highly accurate such that errors do not grow too large in any dependent variable, a MIMO model should work fine
# if you anticipate significant errors in the simulation of any dependent variable, you should use multiple MISO models instead
# as the model predicts derivatives, system_data must represent a *causal* system
# that is, forcing and the response to that forcing cannot occur at the same timestep
# it may be necessary for the user to shift the forcing data back to make the system causal (especially for time aggregated data like daily rainfall-runoff)
# forcing_coef_constraints is a dictionary of column name and then a 1, 0, or -1 depending on whether the coefficients of that variable should be positive, unconstrained, or negative


def _run_scipy_optimizer(
    optimization_method: str,
    objective_function,
    bounds: np.ndarray,
    max_iter: int,
    verbose: bool,
    optimizer_kwargs: dict,
) -> np.ndarray:
    """
    Dispatch to scipy.optimize methods for global optimization.

    Supports any scipy.optimize method that accepts (objective, bounds, **kwargs).
    Common methods: 'differential_evolution', 'dual_annealing', 'simulated_annealing',
    'basinhopping', 'shgo', 'direct', 'brute'.

    Args:
        optimization_method: Name of scipy.optimize method to use
        objective_function: Callable that takes parameter vector and returns scalar to minimize
        bounds: Array of [min, max] bounds for each parameter
        max_iter: Maximum iterations (used as default for methods that support it)
        verbose: Whether to print progress
        optimizer_kwargs: Additional keyword arguments passed to the optimizer

    Returns:
        Best parameter vector found
    """
    import scipy.optimize as opt

    # Default parameters for each method
    method_defaults = {
        "differential_evolution": {
            "maxiter": max_iter,
            "popsize": 15,
            "mutation": (0.5, 1.5),
            "recombination": 0.7,
            "seed": 42,
            "updating": "deferred",
        },
        "dual_annealing": {
            "maxiter": max_iter * 4,  # DA needs more iterations for good exploration
            "seed": 42,
            "no_local_search": False,
        },
        "simulated_annealing": {
            "maxiter": max_iter * 4,
            "seed": 42,
        },
        "direct": {
            "maxiter": max_iter,
            "eps": 1e-4,
        },
        "brute": {
            "Ns": 20,
        },
    }

    # Get defaults for this method, or empty dict if unknown
    defaults = method_defaults.get(optimization_method, {})

    # Merge defaults with user-provided kwargs (user kwargs take precedence)
    params = {**defaults, **optimizer_kwargs}

    # Get the optimizer function
    optimizer = getattr(opt, optimization_method, None)
    if optimizer is None:
        raise ValueError(
            f"Unknown optimization_method: '{optimization_method}'. "
            f"Supported scipy.optimize methods: {list(method_defaults.keys())}, "
            f"or 'bayesian' for built-in Bayesian optimization."
        )

    if verbose:
        print(f"  Running scipy.optimize.{optimization_method} with params: {params}")

    # Run the optimizer
    result = optimizer(objective_function, bounds, **params)

    if verbose:
        print(
            f"  Optimization complete. Success: {result.success}, Message: {result.message}"
        )
        print(f"  Best value: {-result.fun:.6f} (R²)")

    return result.x


def delay_io_train(
    system_data,
    dependent_columns,
    independent_columns,
    windup_timesteps=0,
    init_transforms=1,
    max_transforms=4,
    max_iter=250,
    poly_order=3,
    transform_dependent=False,
    verbose=False,
    extra_verbose=False,
    include_bias=False,
    include_interaction=False,
    bibo_stable=False,
    transform_only=None,
    forcing_coef_constraints=None,
    early_stopping_threshold=0.005,
    optimization_method="bayesian",
    **optimizer_kwargs,
):
    forcing = system_data[independent_columns].copy(deep=True)

    orig_forcing_columns = forcing.columns
    response = system_data[dependent_columns].copy(deep=True)

    results = dict()  # to store the optimized models for each number of transformations
    prev_model = (
        None  # will hold the initial model for the current number of transforms
    )

    if transform_dependent:
        shape_factors = pd.DataFrame(
            columns=system_data.columns,
            index=range(init_transforms, max_transforms + 1),
        )
        shape_factors.iloc[0, :] = 1  # first transformation is [1,1,0] for each input
        scale_factors = pd.DataFrame(
            columns=system_data.columns,
            index=range(init_transforms, max_transforms + 1),
        )
        scale_factors.iloc[0, :] = 1  # first transformation is [1,1,0] for each input
        loc_factors = pd.DataFrame(
            columns=system_data.columns,
            index=range(init_transforms, max_transforms + 1),
        )
        loc_factors.iloc[0, :] = 0  # first transformation is [1,1,0] for each input
    elif transform_only is not None:  # the user provided a list of columns to transform
        shape_factors = pd.DataFrame(
            columns=transform_only, index=range(init_transforms, max_transforms + 1)
        )
        shape_factors.iloc[0, :] = 1  # first transformation is [1,1,0] for each input
        scale_factors = pd.DataFrame(
            columns=transform_only, index=range(init_transforms, max_transforms + 1)
        )
        scale_factors.iloc[0, :] = 1  # first transformation is [1,1,0] for each input
        loc_factors = pd.DataFrame(
            columns=transform_only, index=range(init_transforms, max_transforms + 1)
        )
        loc_factors.iloc[0, :] = 0  # first transformation is [1,1,0] for each input
    else:
        # the transformation factors should be pandas dataframes where the index is which transformation it is and the columns are the variables
        shape_factors = pd.DataFrame(
            columns=forcing.columns, index=range(init_transforms, max_transforms + 1)
        )
        shape_factors.iloc[0, :] = 1  # first transformation is [1,1,0] for each input
        scale_factors = pd.DataFrame(
            columns=forcing.columns, index=range(init_transforms, max_transforms + 1)
        )
        scale_factors.iloc[0, :] = 1  # first transformation is [1,1,0] for each input
        loc_factors = pd.DataFrame(
            columns=forcing.columns, index=range(init_transforms, max_transforms + 1)
        )
        loc_factors.iloc[0, :] = 0  # first transformation is [1,1,0] for each input
    # print(shape_factors)
    # print(scale_factors)
    # print(loc_factors)
    # first transformation is [1,1,0] for each input
    """
    shape_factors = np.ones(shape=(forcing.shape[1] , init_transforms)   )
    scale_factors = np.ones(shape=(forcing.shape[1] , init_transforms)   )
    loc_factors = np.zeros(shape=(forcing.shape[1] , init_transforms)   )
    """
    # speeds =  list([500,200,50,10, 5,2, 1.1, 1.05,1.01])
    speeds = list(
        [100, 50, 20, 10, 5, 2, 1.1, 1.05, 1.01]
    )  # I don't have a great idea of what good values for these are yet
    if transform_dependent:  # just trying something
        improvement_threshold = (
            1.001  # when improvements are tiny, tighten up the jumps
        )
    else:
        improvement_threshold = 1.0

    for num_transforms in range(init_transforms, max_transforms + 1):
        print("num_transforms")
        print(num_transforms)
        speed_idx = 0
        speed = speeds[speed_idx]

        if not num_transforms == init_transforms:  # if we're not starting right now
            # start dull
            shape_factors.iloc[num_transforms - 1, :] = 10 * (
                num_transforms - 1
            )  # start with a broad peak centered at ten timesteps
            scale_factors.iloc[num_transforms - 1, :] = 1
            loc_factors.iloc[num_transforms - 1, :] = 0
            if verbose:
                print(
                    "starting factors for additional transformation\nshape\nscale\nlocation"
                )
                print(shape_factors)
                print(scale_factors)
                print(loc_factors)

        # Choose optimization method
        if optimization_method == "bayesian":
            if verbose:
                print(f"Using Bayesian optimization for {num_transforms} transforms...")

            # Determine which columns to transform
            if transform_dependent:
                transform_columns = system_data.columns.tolist()
            elif transform_only is not None:
                transform_columns = transform_only
            else:
                transform_columns = independent_columns

            # Bayesian optimization for this number of transforms
            bounds_list: list[list[float]] = []
            for transform in range(1, num_transforms + 1):
                for col in transform_columns:
                    bounds_list.append([1.0, 50.0])  # shape_factors bounds
                    bounds_list.append([0.1, 5.0])  # scale_factors bounds
                    bounds_list.append([0.0, 20.0])  # loc_factors bounds
            bounds = np.array(bounds_list)

            def objective_function(params_vector):
                try:
                    # Convert vector to DataFrames
                    shape_factors_opt = pd.DataFrame(
                        columns=transform_columns, index=range(1, num_transforms + 1)
                    )
                    scale_factors_opt = pd.DataFrame(
                        columns=transform_columns, index=range(1, num_transforms + 1)
                    )
                    loc_factors_opt = pd.DataFrame(
                        columns=transform_columns, index=range(1, num_transforms + 1)
                    )

                    idx = 0
                    for transform in range(1, num_transforms + 1):
                        for col in transform_columns:
                            shape_factors_opt.loc[transform, col] = params_vector[idx]
                            scale_factors_opt.loc[transform, col] = params_vector[
                                idx + 1
                            ]
                            loc_factors_opt.loc[transform, col] = params_vector[idx + 2]
                            idx += 3

                    result = SINDY_delays_MI(
                        shape_factors_opt,
                        scale_factors_opt,
                        loc_factors_opt,
                        system_data.index,
                        forcing,
                        response,
                        False,
                        poly_order,
                        include_bias,
                        include_interaction,
                        windup_timesteps,
                        bibo_stable,
                        transform_dependent,
                        transform_only,
                        forcing_coef_constraints,
                        transform_cache=_transform_cache,
                    )

                    r2 = result["error_metrics"]["r2"]
                    if verbose:
                        print(f"  R² = {r2:.6f}")
                    return r2
                except Exception as e:
                    if verbose:
                        print(f"  Evaluation failed: {e}")
                    return -1.0

            # Bayesian optimization
            # Use more iterations for Bayesian optimization to build a good surrogate model
            # Cap at 200 to avoid memory issues with GP fitting
            bayesian_max_iter = min(max_iter * 4, 200)
            n_initial = min(20, max(10, bayesian_max_iter // 4))
            X_sample_list: list[Any] = []
            Y_sample_list: list[Any] = []

            # Generate initial random samples
            for i in range(n_initial):
                x = np.random.uniform(bounds[:, 0], bounds[:, 1])
                y = objective_function(x)
                X_sample_list.append(x)
                Y_sample_list.append(y)
                if verbose:
                    print(f"  Initial sample {i+1}/{n_initial}: R² = {y:.6f}")

            X_sample: np.ndarray = np.array(X_sample_list)
            Y_sample: np.ndarray = np.array(Y_sample_list).reshape(-1, 1)

            # Main Bayesian optimization loop
            best_r2 = np.max(Y_sample)
            best_params: np.ndarray = X_sample[np.argmax(Y_sample)]

            # Gaussian Process setup
            kernel = Matern(length_scale=1.0, nu=2.5)
            gpr = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42,
            )

            for iteration in range(bayesian_max_iter - n_initial):
                # Fit GP and find next point
                gpr.fit(X_sample, Y_sample.ravel())
                next_x = _propose_location(
                    _expected_improvement, X_sample, Y_sample, gpr, bounds
                )
                next_x = next_x.flatten()

                # Evaluate objective
                next_y = objective_function(next_x)

                if verbose:
                    print(
                        f"  BO iteration {iteration+1}/{bayesian_max_iter-n_initial}: R² = {next_y:.6f}"
                    )

                # Update samples
                X_sample = np.append(X_sample, [next_x], axis=0)
                Y_sample = np.append(Y_sample, next_y)

                # Update best
                if next_y > best_r2:
                    best_r2 = next_y
                    best_params = next_x
                    if verbose:
                        print(f"    New best R² = {best_r2:.6f}")

            # Convert best parameters back to DataFrames
            idx = 0
            for transform in range(1, num_transforms + 1):
                for col in transform_columns:
                    shape_factors.loc[transform, col] = best_params[idx]
                    scale_factors.loc[transform, col] = best_params[idx + 1]
                    loc_factors.loc[transform, col] = best_params[idx + 2]
                    idx += 3

            # Use the optimized parameters for final evaluation
            prev_model = SINDY_delays_MI(
                shape_factors,
                scale_factors,
                loc_factors,
                system_data.index,
                forcing,
                response,
                extra_verbose,
                poly_order,
                include_bias,
                include_interaction,
                windup_timesteps,
                bibo_stable,
                transform_dependent=transform_dependent,
                transform_only=transform_only,
                forcing_coef_constraints=forcing_coef_constraints,
                transform_cache=_transform_cache,
            )

        else:
            # Use scipy.optimize for all other methods (differential_evolution, dual_annealing,
            # basinhopping, shgo, direct, etc.)
            if verbose:
                print(
                    f"Using {optimization_method} optimization for {num_transforms} transforms..."
                )

            # Determine which columns to transform
            if transform_dependent:
                transform_columns = system_data.columns.tolist()
            elif transform_only is not None:
                transform_columns = transform_only
            else:
                transform_columns = independent_columns

            # Define parameter bounds for this number of transforms
            bounds_list: list[list[float]] = []
            for transform in range(1, num_transforms + 1):
                for col in transform_columns:
                    bounds_list.append([1.0, 50.0])  # shape_factors bounds
                    bounds_list.append([0.1, 5.0])  # scale_factors bounds
                    bounds_list.append([0.0, 20.0])  # loc_factors bounds
            bounds = np.array(bounds_list)

            def objective_function(params_vector):
                try:
                    # Convert vector to DataFrames
                    shape_factors_opt = pd.DataFrame(
                        columns=transform_columns, index=range(1, num_transforms + 1)
                    )
                    scale_factors_opt = pd.DataFrame(
                        columns=transform_columns, index=range(1, num_transforms + 1)
                    )
                    loc_factors_opt = pd.DataFrame(
                        columns=transform_columns, index=range(1, num_transforms + 1)
                    )

                    idx = 0
                    for transform in range(1, num_transforms + 1):
                        for col in transform_columns:
                            shape_factors_opt.loc[transform, col] = params_vector[idx]
                            scale_factors_opt.loc[transform, col] = params_vector[
                                idx + 1
                            ]
                            loc_factors_opt.loc[transform, col] = params_vector[idx + 2]
                            idx += 3

                    result = SINDY_delays_MI(
                        shape_factors_opt,
                        scale_factors_opt,
                        loc_factors_opt,
                        system_data.index,
                        forcing,
                        response,
                        False,
                        poly_order,
                        include_bias,
                        include_interaction,
                        windup_timesteps,
                        bibo_stable,
                        transform_dependent=transform_dependent,
                        transform_only=transform_only,
                        forcing_coef_constraints=forcing_coef_constraints,
                        transform_cache=_transform_cache,
                    )

                    r2 = result["error_metrics"]["r2"]
                    if verbose:
                        print(f"  R² = {r2:.6f}")
                    return -r2  # Minimize negative R² (maximize R²)
                except Exception as e:
                    if verbose:
                        print(f"  Evaluation failed: {e}")
                    return 1.0  # Poor score for failed evaluations

            # Dispatch to scipy.optimize method
            best_params = _run_scipy_optimizer(
                optimization_method=optimization_method,
                objective_function=objective_function,
                bounds=bounds,
                max_iter=max_iter,
                verbose=verbose,
                optimizer_kwargs=optimizer_kwargs,
            )

            # Convert best parameters back to DataFrames
            idx = 0
            for transform in range(1, num_transforms + 1):
                for col in transform_columns:
                    shape_factors.loc[transform, col] = best_params[idx]
                    scale_factors.loc[transform, col] = best_params[idx + 1]
                    loc_factors.loc[transform, col] = best_params[idx + 2]
                    idx += 3

            # Use the optimized parameters for final evaluation
            prev_model = SINDY_delays_MI(
                shape_factors,
                scale_factors,
                loc_factors,
                system_data.index,
                forcing,
                response,
                extra_verbose,
                poly_order,
                include_bias,
                include_interaction,
                windup_timesteps,
                bibo_stable,
                transform_dependent=transform_dependent,
                transform_only=transform_only,
                forcing_coef_constraints=forcing_coef_constraints,
                transform_cache=_transform_cache,
            )

        # For bayesian and scipy.optimize methods, we're done with optimization
        print("\nOptimization complete. Using optimized parameters for final model.")
        final_model = SINDY_delays_MI(
            shape_factors,
            scale_factors,
            loc_factors,
            system_data.index,
            forcing,
            response,
            True,
            poly_order,
            include_bias,
            include_interaction,
            windup_timesteps,
            bibo_stable,
            transform_dependent=transform_dependent,
            transform_only=transform_only,
            forcing_coef_constraints=forcing_coef_constraints,
            transform_cache=_transform_cache,
        )
        print("\nFinal model:\n")
        try:
            print(final_model["model"].print(precision=5))
        except Exception as e:
            print(e)
        print("R^2")
        print(prev_model["error_metrics"]["r2"])
        print("shape factors")
        print(shape_factors)
        print("scale factors")
        print(scale_factors)
        print("location factors")
        print(loc_factors)
        print("\n")
        results[num_transforms] = {
            "final_model": final_model.copy(),
            "shape_factors": shape_factors.copy(deep=True),
            "scale_factors": scale_factors.copy(deep=True),
            "loc_factors": loc_factors.copy(deep=True),
            "windup_timesteps": windup_timesteps,
            "dependent_columns": dependent_columns,
            "independent_columns": independent_columns,
            "transform_cache": _transform_cache,
        }

        # check if the benefit from adding the last transformation is less than the early stopping threshold
        if (
            num_transforms > init_transforms
            and results[num_transforms]["final_model"]["error_metrics"]["r2"]
            - results[num_transforms - 1]["final_model"]["error_metrics"]["r2"]
            < early_stopping_threshold
        ):
            print(
                "Last transformation added less than ",
                early_stopping_threshold * 100,
                " % to R2 score. Terminating early.",
            )
            break
        continue

        print("\nInitial model:\n")
        try:
            print(prev_model["model"].print(precision=5))
            print("R^2")
            print(prev_model["error_metrics"]["r2"])
        except Exception as e:  # and print the exception:
            print(e)
            pass
        print("shape factors")
        print(shape_factors)
        print("scale factors")
        print(scale_factors)
        print("location factors")
        print(loc_factors)
        print("\n")

        if not verbose:
            print("training ", end="")

        # no_improvement_last_time = False
        for iterations in range(0, max_iter):
            if not verbose and iterations % 5 == 0:
                print(str(iterations) + ".", end="")

            if transform_dependent:
                tuning_input = system_data.columns[
                    (iterations // num_transforms) % len(system_data.columns)
                ]  # row =  iter // width % height]
            elif transform_only is not None:
                tuning_input = transform_only[
                    (iterations // num_transforms) % len(transform_only)
                ]
            else:
                tuning_input = orig_forcing_columns[
                    (iterations // num_transforms) % len(orig_forcing_columns)
                ]  # row =  iter // width % height
            tuning_line = (
                iterations % num_transforms + 1
            )  # col =  % width (plus one because there's no zeroth transformation)
            if verbose:
                print(
                    str(
                        "tuning input: {i} | tuning transformation: {l:g}".format(
                            i=tuning_input, l=tuning_line
                        )
                    )
                )

            sooner_locs = loc_factors.copy(deep=True)
            # sooner_locs[tuning_input][tuning_line] = float(loc_factors[tuning_input][tuning_line] - speed/10  )
            sooner_locs.loc[tuning_line, tuning_input] = float(
                loc_factors.loc[tuning_line, tuning_input] - speed / 10
            )
            if sooner_locs[tuning_input][tuning_line] < 0:
                sooner = {"error_metrics": {"r2": -1}}
            else:
                sooner = SINDY_delays_MI(
                    shape_factors,
                    scale_factors,
                    sooner_locs,
                    system_data.index,
                    forcing,
                    response,
                    extra_verbose,
                    poly_order,
                    include_bias,
                    include_interaction,
                    windup_timesteps,
                    bibo_stable,
                    transform_dependent=transform_dependent,
                    transform_only=transform_only,
                    forcing_coef_constraints=forcing_coef_constraints,
                    transform_cache=_transform_cache,
                )

            later_locs = loc_factors.copy(deep=True)
            # later_locs[tuning_input][tuning_line] = float ( loc_factors[tuning_input][tuning_line]  +   1.01*speed/10 )
            later_locs.loc[tuning_line, tuning_input] = float(
                loc_factors.loc[tuning_line, tuning_input] + 1.01 * speed / 10
            )
            later = SINDY_delays_MI(
                shape_factors,
                scale_factors,
                later_locs,
                system_data.index,
                forcing,
                response,
                extra_verbose,
                poly_order,
                include_bias,
                include_interaction,
                windup_timesteps,
                bibo_stable,
                transform_dependent=transform_dependent,
                transform_only=transform_only,
                forcing_coef_constraints=forcing_coef_constraints,
                transform_cache=_transform_cache,
            )

            shape_up = shape_factors.copy(deep=True)
            # shape_up[tuning_input][tuning_line] = float ( shape_factors[tuning_input][tuning_line]*speed*1.01 )
            shape_up.loc[tuning_line, tuning_input] = float(
                shape_factors.loc[tuning_line, tuning_input] * speed * 1.01
            )
            shape_upped = SINDY_delays_MI(
                shape_up,
                scale_factors,
                loc_factors,
                system_data.index,
                forcing,
                response,
                extra_verbose,
                poly_order,
                include_bias,
                include_interaction,
                windup_timesteps,
                bibo_stable,
                transform_dependent=transform_dependent,
                transform_only=transform_only,
                forcing_coef_constraints=forcing_coef_constraints,
                transform_cache=_transform_cache,
            )

            shape_down = shape_factors.copy(deep=True)
            # shape_down[tuning_input][tuning_line] = float ( shape_factors[tuning_input][tuning_line]/speed )
            shape_down.loc[tuning_line, tuning_input] = float(
                shape_factors.loc[tuning_line, tuning_input] / speed
            )
            if shape_down[tuning_input][tuning_line] < 1:
                shape_downed = {
                    "error_metrics": {"r2": -1}
                }  # return a score of negative one as this is illegal
            else:
                shape_downed = SINDY_delays_MI(
                    shape_down,
                    scale_factors,
                    loc_factors,
                    system_data.index,
                    forcing,
                    response,
                    extra_verbose,
                    poly_order,
                    include_bias,
                    include_interaction,
                    windup_timesteps,
                    bibo_stable,
                    transform_dependent=transform_dependent,
                    transform_only=transform_only,
                    forcing_coef_constraints=forcing_coef_constraints,
                    transform_cache=_transform_cache,
                )

            scale_up = scale_factors.copy(deep=True)
            # scale_up[tuning_input][tuning_line] = float(  scale_factors[tuning_input][tuning_line]*speed*1.01 )
            scale_up.loc[tuning_line, tuning_input] = float(
                scale_factors.loc[tuning_line, tuning_input] * speed * 1.01
            )
            scaled_up = SINDY_delays_MI(
                shape_factors,
                scale_up,
                loc_factors,
                system_data.index,
                forcing,
                response,
                extra_verbose,
                poly_order,
                include_bias,
                include_interaction,
                windup_timesteps,
                bibo_stable,
                transform_dependent=transform_dependent,
                transform_only=transform_only,
                forcing_coef_constraints=forcing_coef_constraints,
                transform_cache=_transform_cache,
            )

            scale_down = scale_factors.copy(deep=True)
            # scale_down[tuning_input][tuning_line] = float ( scale_factors[tuning_input][tuning_line]/speed )
            scale_down.loc[tuning_line, tuning_input] = float(
                scale_factors.loc[tuning_line, tuning_input] / speed
            )
            scaled_down = SINDY_delays_MI(
                shape_factors,
                scale_down,
                loc_factors,
                system_data.index,
                forcing,
                response,
                extra_verbose,
                poly_order,
                include_bias,
                include_interaction,
                windup_timesteps,
                bibo_stable,
                transform_dependent=transform_dependent,
                transform_only=transform_only,
                forcing_coef_constraints=forcing_coef_constraints,
                transform_cache=_transform_cache,
            )

            # rounder
            rounder_shape = shape_factors.copy(deep=True)
            # rounder_shape[tuning_input][tuning_line] = shape_factors[tuning_input][tuning_line]*(speed*1.01)
            rounder_shape.loc[tuning_line, tuning_input] = shape_factors.loc[
                tuning_line, tuning_input
            ] * (speed * 1.01)
            rounder_scale = scale_factors.copy(deep=True)
            # rounder_scale[tuning_input][tuning_line] = scale_factors[tuning_input][tuning_line]/(speed*1.01)
            rounder_scale.loc[tuning_line, tuning_input] = scale_factors.loc[
                tuning_line, tuning_input
            ] / (speed * 1.01)
            rounder = SINDY_delays_MI(
                rounder_shape,
                rounder_scale,
                loc_factors,
                system_data.index,
                forcing,
                response,
                extra_verbose,
                poly_order,
                include_bias,
                include_interaction,
                windup_timesteps,
                bibo_stable,
                transform_dependent=transform_dependent,
                transform_only=transform_only,
                forcing_coef_constraints=forcing_coef_constraints,
                transform_cache=_transform_cache,
            )

            # sharper
            sharper_shape = shape_factors.copy(deep=True)
            # sharper_shape[tuning_input][tuning_line] = shape_factors[tuning_input][tuning_line]/speed
            sharper_shape.loc[tuning_line, tuning_input] = (
                shape_factors.loc[tuning_line, tuning_input] / speed
            )
            if sharper_shape[tuning_input][tuning_line] < 1:
                sharper = {
                    "error_metrics": {"r2": -1}
                }  # lower bound on shape to avoid inf
            else:
                sharper_scale = scale_factors.copy(deep=True)
                # sharper_scale[tuning_input][tuning_line] = scale_factors[tuning_input][tuning_line]*speed
                sharper_scale.loc[tuning_line, tuning_input] = (
                    scale_factors.loc[tuning_line, tuning_input] * speed
                )
                sharper = SINDY_delays_MI(
                    sharper_shape,
                    sharper_scale,
                    loc_factors,
                    system_data.index,
                    forcing,
                    response,
                    extra_verbose,
                    poly_order,
                    include_bias,
                    include_interaction,
                    windup_timesteps,
                    bibo_stable,
                    transform_dependent=transform_dependent,
                    transform_only=transform_only,
                    forcing_coef_constraints=forcing_coef_constraints,
                    transform_cache=_transform_cache,
                )

            scores = [
                prev_model["error_metrics"]["r2"],
                shape_upped["error_metrics"]["r2"],
                shape_downed["error_metrics"]["r2"],
                scaled_up["error_metrics"]["r2"],
                scaled_down["error_metrics"]["r2"],
                sooner["error_metrics"]["r2"],
                later["error_metrics"]["r2"],
                rounder["error_metrics"]["r2"],
                sharper["error_metrics"]["r2"],
            ]
            # print(scores)

            if (
                sooner["error_metrics"]["r2"] >= max(scores)
                and sooner["error_metrics"]["r2"]
                > improvement_threshold * prev_model["error_metrics"]["r2"]
            ):
                prev_model = sooner.copy()
                loc_factors = sooner_locs.copy(deep=True)
            elif (
                later["error_metrics"]["r2"] >= max(scores)
                and later["error_metrics"]["r2"]
                > improvement_threshold * prev_model["error_metrics"]["r2"]
            ):
                prev_model = later.copy()
                loc_factors = later_locs.copy(deep=True)
            elif (
                shape_upped["error_metrics"]["r2"] >= max(scores)
                and shape_upped["error_metrics"]["r2"]
                > improvement_threshold * prev_model["error_metrics"]["r2"]
            ):
                prev_model = shape_upped.copy()
                shape_factors = shape_up.copy(deep=True)
            elif (
                shape_downed["error_metrics"]["r2"] >= max(scores)
                and shape_downed["error_metrics"]["r2"]
                > improvement_threshold * prev_model["error_metrics"]["r2"]
            ):
                prev_model = shape_downed.copy()
                shape_factors = shape_down.copy(deep=True)
            elif (
                scaled_up["error_metrics"]["r2"] >= max(scores)
                and scaled_up["error_metrics"]["r2"]
                > improvement_threshold * prev_model["error_metrics"]["r2"]
            ):
                prev_model = scaled_up.copy()
                scale_factors = scale_up.copy(deep=True)
            elif (
                scaled_down["error_metrics"]["r2"] >= max(scores)
                and scaled_down["error_metrics"]["r2"]
                > improvement_threshold * prev_model["error_metrics"]["r2"]
            ):
                prev_model = scaled_down.copy()
                scale_factors = scale_down.copy(deep=True)
            elif (
                rounder["error_metrics"]["r2"] >= max(scores)
                and rounder["error_metrics"]["r2"]
                > improvement_threshold * prev_model["error_metrics"]["r2"]
            ):
                prev_model = rounder.copy()
                shape_factors = rounder_shape.copy(deep=True)
                scale_factors = rounder_scale.copy(deep=True)
            elif (
                sharper["error_metrics"]["r2"] >= max(scores)
                and sharper["error_metrics"]["r2"]
                > improvement_threshold * prev_model["error_metrics"]["r2"]
            ):
                prev_model = sharper.copy()
                shape_factors = sharper_shape.copy(deep=True)
                scale_factors = sharper_scale.copy(deep=True)
            # the middle was best, but it's bad, tighten up the bounds (if we're at the last tuning line of the last input)

            elif (
                num_transforms == tuning_line
                and tuning_input == shape_factors.columns[-1]
            ):  # no improvement transforming last column
                # no_improvement_last_time=True
                speed_idx = speed_idx + 1
                if verbose:
                    print("\n\ntightening bounds\n\n")
                """
            elif (num_transforms == tuning_line and tuning_input == orig_forcing_columns[0] and no_improvement_last_time): # no improvement next iteration (first column)
                speed_idx = speed_idx + 1
                no_improvement_last_time=False
                if verbose:
                    print("\n\ntightening bounds\n\n")
                    """

            if speed_idx >= len(speeds):
                print("\n\noptimization complete\n\n")
                break
            speed = speeds[speed_idx]
            if verbose:
                print(
                    "\nprevious, shape up, shape down, scale up, scale down, sooner, later, rounder, sharper"
                )
                print(scores)
                print("speed")
                print(speed)
                print("shape factors")
                print(shape_factors)
                print("scale factors")
                print(scale_factors)
                print("location factors")
                print(loc_factors)
                print("iteration no:")
                print(iterations)
                print("model")
                try:
                    prev_model["model"].print(precision=5)
                except Exception as e:
                    print(e)
                print("\n")

        final_model = SINDY_delays_MI(
            shape_factors,
            scale_factors,
            loc_factors,
            system_data.index,
            forcing,
            response,
            True,
            poly_order,
            include_bias,
            include_interaction,
            windup_timesteps,
            bibo_stable,
            transform_dependent=transform_dependent,
            transform_only=transform_only,
            forcing_coef_constraints=forcing_coef_constraints,
            transform_cache=_transform_cache,
        )
        print("\nFinal model:\n")
        try:
            print(final_model["model"].print(precision=5))
        except Exception as e:
            print(e)
        print("R^2")
        print(prev_model["error_metrics"]["r2"])
        print("shape factors")
        print(shape_factors)
        print("scale factors")
        print(scale_factors)
        print("location factors")
        print(loc_factors)
        print("\n")
        results[num_transforms] = {
            "final_model": final_model.copy(),
            "shape_factors": shape_factors.copy(deep=True),
            "scale_factors": scale_factors.copy(deep=True),
            "loc_factors": loc_factors.copy(deep=True),
            "windup_timesteps": windup_timesteps,
            "dependent_columns": dependent_columns,
            "independent_columns": independent_columns,
            "transform_cache": _transform_cache,
        }

        # check if the benefit from adding the last transformation is less than the early stopping threshold
        if (
            num_transforms > init_transforms
            and results[num_transforms]["final_model"]["error_metrics"]["r2"]
            - results[num_transforms - 1]["final_model"]["error_metrics"]["r2"]
            < early_stopping_threshold
        ):
            print(
                "Last transformation added less than ",
                early_stopping_threshold * 100,
                " % to R2 score. Terminating early.",
            )
            break

    return results


def SINDY_delays_MI(
    shape_factors,
    scale_factors,
    loc_factors,
    index,
    forcing,
    response,
    final_run,
    poly_degree,
    include_bias,
    include_interaction,
    windup_timesteps,
    bibo_stable=False,
    transform_dependent=False,
    transform_only=None,
    forcing_coef_constraints=None,
    transform_cache=None,
):
    if transform_only is not None:
        transformed_forcing = transform_inputs(
            shape_factors,
            scale_factors,
            loc_factors,
            index,
            forcing.loc[:, transform_only],
            cache=transform_cache,
        )
        untransformed_forcing = forcing.drop(columns=transform_only)
        # combine forcing and transformed forcing column-wise
        forcing = pd.concat(
            (untransformed_forcing, transformed_forcing), axis="columns"
        )
    else:
        forcing = transform_inputs(
            shape_factors,
            scale_factors,
            loc_factors,
            index,
            forcing,
            cache=transform_cache,
        )

    feature_names = response.columns.tolist() + forcing.columns.tolist()

    # SINDy
    if (
        not bibo_stable and forcing_coef_constraints is None
    ):  # no constraints, normal mode
        model = ps.SINDy(
            differentiation_method=ps.FiniteDifference(),
            feature_library=ps.PolynomialLibrary(
                degree=poly_degree,
                include_bias=include_bias,
                include_interaction=include_interaction,
            ),
            optimizer=ps.STLSQ(threshold=0),
        )
    elif forcing_coef_constraints is not None and not bibo_stable:
        library = ps.PolynomialLibrary(
            degree=poly_degree,
            include_bias=include_bias,
            include_interaction=include_interaction,
        )
        total_train = pd.concat((response, forcing), axis="columns")
        library.fit([ps.AxesArray(total_train, {"ax_sample": 0, "ax_coord": 1})])
        n_features = library.n_output_features_
        n_targets = len(response.columns)
        constraint_rhs = np.zeros((n_features,))  # every feature is constrained
        # one row per constraint, one column per coefficient
        constraint_lhs = np.zeros((n_features, n_targets * n_features))

        # now implement the forcing coefficient constraints
        for i, col in enumerate(feature_names):
            for key in forcing_coef_constraints.keys():
                if key in col:
                    constraint_lhs[i, i] = -forcing_coef_constraints[key]
                    # invert the sign because the eqn is written as "leq 0"

        model = ps.SINDy(
            differentiation_method=ps.FiniteDifference(),
            feature_library=ps.PolynomialLibrary(
                degree=poly_degree,
                include_bias=include_bias,
                include_interaction=include_interaction,
            ),
            optimizer=_ConstrainedSR3(
                reg_weight_lam=0,
                regularizer="l2",
                constraint_lhs=constraint_lhs,
                constraint_rhs=constraint_rhs,
                inequality_constraints=True,
            ),
        )
    elif (
        bibo_stable
    ):  # highest order output autocorrelation is constrained to be negative
        # import cvxpy
        # run_cvxpy= True
        # Figure out how many library features there will be
        library = ps.PolynomialLibrary(
            degree=poly_degree,
            include_bias=include_bias,
            include_interaction=include_interaction,
        )
        total_train = pd.concat((response, forcing), axis="columns")
        library.fit([ps.AxesArray(total_train, {"ax_sample": 0, "ax_coord": 1})])
        n_features = library.n_output_features_
        # print(f"Features ({n_features}):", library.get_feature_names(input_features=total_train.columns))
        feature_names = library.get_feature_names(input_features=total_train.columns)
        # Set constraints
        n_targets = total_train.shape[
            1
        ]  # not sure what targets means after reading through the pysindy docs
        # print("n_targets")
        # print(n_targets)
        constraint_rhs = np.zeros((len(response.columns), 1))
        # one row per constraint, one column per coefficient
        constraint_lhs = np.zeros((len(response.columns), n_features))

        # print(constraint_rhs)
        # print(constraint_lhs)
        # constrain the highest order output autocorrelation to be negative
        # this indexing is only right for include_interaction=False, include_bias=False, and pure polynomial library
        # for more complex libraries, some conditional logic will be needed to grab the right column
        constraint_lhs[
            :, -len(forcing.columns) - len(response.columns) : -len(forcing.columns)
        ] = 1
        # leq 0
        # print("constraint lhs")
        # print(constraint_lhs)

        # forcing_coef_constraints only implemented for bibo stable MISO models right now
        if forcing_coef_constraints is not None:
            n_targets = len(response.columns)
            constraint_rhs = np.zeros((n_features,))  # every feature is constrained
            # one row per constraint, one column per coefficient
            constraint_lhs = np.zeros((n_features, n_targets * n_features))
            # bibo stability, set the highest order output autocorrelation to be negative for each response variable
            # the index corresponds to the last entry in "feature_names" which includes the name of the response column
            highest_power_col_idx = 0
            for i, col in enumerate(feature_names):
                if response.columns[0] in col:
                    highest_power_col_idx = i
            constraint_lhs[0, highest_power_col_idx] = (
                1  # first row, highest power of the response variable
            )

            # now implement the forcing coefficient constraints
            for i, col in enumerate(feature_names):
                for key in forcing_coef_constraints.keys():
                    if key in col:
                        constraint_lhs[i, i] = -forcing_coef_constraints[key]
                        # invert the sign because the eqn is written as "leq 0"
            """'
            print(forcing.columns)
            forcing_constraints_array = np.ndarray(shape=(1,len(forcing.columns)))
            for i, col in enumerate(forcing.columns):
                if col in forcing_coef_constraints.keys(): # invert the sign because the eqn is written as "leq 0"
                    forcing_constraints_array[0,i] = -forcing_coef_constraints[col]
                elif str(col).replace('_tr_1','') in forcing_coef_constraints.keys():
                    forcing_constraints_array[0,i] = -forcing_coef_constraints[str(col).replace('_tr_1','')]
                elif str(col).replace('_tr_2','') in forcing_coef_constraints.keys():
                    forcing_constraints_array[0,i] = -forcing_coef_constraints[str(col).replace('_tr_2','')]
                elif str(col).replace('_tr_3','') in forcing_coef_constraints.keys():
                    forcing_constraints_array[0,i] = -forcing_coef_constraints[str(col).replace('_tr_3','')]
                else:
                    forcing_constraints_array[0,i] = 0

            for row in range(n_targets, n_features):
                constraint_lhs[row, row] = forcing_constraints_array[0,row - n_targets]
            """

            # constrain the highest order output autocorrelation to be negative
            # this indexing is only right for include_interaction=False, include_bias=False, and pure polynomial library
            # for more complex libraries, some conditional logic will be needed to grab the right column
            # constraint_lhs[:n_targets,-len(forcing.columns)-len(response.columns):-len(forcing.columns)] = 1

            # print(forcing_constraints_array)

            # print('constraint lhs')
            # print(constraint_lhs)
            # print('constraint rhs')
            # print(constraint_rhs)

        model = ps.SINDy(
            differentiation_method=ps.FiniteDifference(),
            feature_library=ps.PolynomialLibrary(
                degree=poly_degree,
                include_bias=include_bias,
                include_interaction=include_interaction,
            ),
            optimizer=_ConstrainedSR3(
                reg_weight_lam=0,
                regularizer="l2",
                constraint_lhs=constraint_lhs,
                constraint_rhs=constraint_rhs,
                inequality_constraints=True,
            ),
        )
    if transform_dependent:
        # combine response and forcing into one dataframe
        total_train = pd.concat((response, forcing), axis="columns")
        total_train = transform_inputs(
            shape_factors, scale_factors, loc_factors, index, total_train
        )
        # remove the columns in total_train that are already in response (just want to keep the transformed forcing)
        total_train = total_train.drop(columns=response.columns)
        feature_names = response.columns.tolist() + total_train.columns.tolist()

        # need to add constraints such that variables don't depend on their own past values (but they can have autocorrelations)

        library = ps.PolynomialLibrary(
            degree=poly_degree,
            include_bias=include_bias,
            include_interaction=include_interaction,
        )
        library_terms = pd.concat((total_train, response), axis="columns")
        library.fit([ps.AxesArray(library_terms, {"ax_sample": 0, "ax_coord": 1})])
        n_features = library.n_output_features_
        # print(f"Features ({n_features}):", library.get_feature_names())
        # Set constraints
        n_targets = response.shape[
            1
        ]  # not sure what targets means after reading through the pysindy docs

        constraint_rhs = np.zeros((n_targets,))
        # one row per constraint, one column per coefficient
        constraint_lhs = np.zeros((n_targets, n_features * n_targets))
        # for bibo stability, starting guess is that each dependent variable is negatively autocorrelated and depends on no other variable
        if bibo_stable:
            initial_guess = np.zeros((n_targets, n_features))
            for idx in range(0, n_targets):
                initial_guess[idx, idx] = -1
        else:
            initial_guess = None
        # print(constraint_rhs)
        # print(constraint_lhs)
        # set the coefficient on a variable's own transformed value to 0
        for idx in range(0, n_targets):
            constraint_lhs[idx, (idx + 1) * n_features - n_targets + idx] = 1

        # print("constraint lhs")
        # print(constraint_lhs)

        model = ps.SINDy(
            differentiation_method=ps.FiniteDifference(),
            feature_library=library,
            optimizer=_ConstrainedSR3(
                reg_weight_lam=0,
                regularizer="l0",
                relax_coeff_nu=10e9,
                initial_guess=initial_guess,
                constraint_lhs=constraint_lhs,
                constraint_rhs=constraint_rhs,
                inequality_constraints=False,
                max_iter=10000,
            ),
        )

        try:
            # windup latent states (if your windup is too long, this will error)
            model.fit(
                response.values[windup_timesteps:, :],
                t=np.arange(0, len(index), 1)[windup_timesteps:],
                u=total_train.values[windup_timesteps:, :],
            )
            r2 = model.score(
                response.values[windup_timesteps:, :],
                t=np.arange(0, len(index), 1)[windup_timesteps:],
                u=total_train.values[windup_timesteps:, :],
            )  # training data score
        except Exception as e:  # and print the exception
            print("Exception in model fitting, returning r2=-1\n")
            print(e)
            error_metrics = {
                "MAE": [False],
                "RMSE": [False],
                "NSE": [False],
                "alpha": [False],
                "beta": [False],
                "HFV": [False],
                "HFV10": [False],
                "LFV": [False],
                "FDC": [False],
                "r2": -1,
            }
            return {
                "error_metrics": error_metrics,
                "model": None,
                "simulated": False,
                "response": response,
                "forcing": forcing,
                "index": index,
                "diverged": False,
            }

    else:
        try:
            # windup latent states (if your windup is too long, this will error)
            model.fit(
                response.values[windup_timesteps:, :],
                t=np.arange(0, len(index), 1)[windup_timesteps:],
                u=forcing.values[windup_timesteps:, :],
            )
            r2 = model.score(
                response.values[windup_timesteps:, :],
                t=np.arange(0, len(index), 1)[windup_timesteps:],
                u=forcing.values[windup_timesteps:, :],
            )  # training data score
        except Exception as e:  # and print the exception
            print("Exception in model fitting, returning r2=-1\n")
            print(e)
            error_metrics = {
                "MAE": [False],
                "RMSE": [False],
                "NSE": [False],
                "alpha": [False],
                "beta": [False],
                "HFV": [False],
                "HFV10": [False],
                "LFV": [False],
                "FDC": [False],
                "r2": -1,
            }
            return {
                "error_metrics": error_metrics,
                "model": None,
                "simulated": False,
                "response": response,
                "forcing": forcing,
                "index": index,
                "diverged": False,
            }
        # r2 is how well we're doing across all the outputs. that's actually good to keep model accuracy lumped because that's what makes most sense to drive the optimization
    # even though the metrics we'll want to evaluate models on are individual output accuracy
    # print("training R^2", r2)
    # model.print(precision=5)

    # return false for things not evaluated / don't exist
    error_metrics = {
        "MAE": [False],
        "RMSE": [False],
        "NSE": [False],
        "alpha": [False],
        "beta": [False],
        "HFV": [False],
        "HFV10": [False],
        "LFV": [False],
        "FDC": [False],
        "r2": r2,
    }
    simulated = False
    if final_run:  # only simulate final runs because it's slow
        try:  # once in high volume training put this back in, but want to see the errors during development
            if transform_dependent:
                simulated = model.simulate(
                    response.values[windup_timesteps, :],
                    t=np.arange(0, len(index), 1)[windup_timesteps:],
                    u=total_train.values[windup_timesteps:, :],
                )
            else:
                simulated = model.simulate(
                    response.values[windup_timesteps, :],
                    t=np.arange(0, len(index), 1)[windup_timesteps:],
                    u=forcing.values[windup_timesteps:, :],
                )
            mae = list()
            rmse = list()
            nse = list()
            alpha = list()
            beta = list()
            hfv = list()
            hfv10 = list()
            lfv = list()
            fdc = list()
            for col_idx in range(
                0, len(response.columns)
            ):  # univariate performance metrics
                error = (
                    response.values[windup_timesteps + 1 :, col_idx]
                    - simulated[:, col_idx]
                )

                # print("error")
                # print(error)
                # nash sutcliffe efficiency between response and simulated
                mae.append(np.mean(np.abs(error)))
                rmse.append(np.sqrt(np.mean(error**2)))
                # print("mean measured = ", np.mean(response.values[windup_timesteps+1:,col_idx]  ))
                # print("sum of squared error between measured and model = ", np.sum((error)**2 ))
                # print("sum of squared error between measured and mean of measured = ", np.sum((response.values[windup_timesteps+1:,col_idx]-np.mean(response.values[windup_timesteps+1:,col_idx]  ) )**2 ))
                nse.append(
                    1
                    - np.sum((error) ** 2)
                    / np.sum(
                        (
                            response.values[windup_timesteps + 1 :, col_idx]
                            - np.mean(response.values[windup_timesteps + 1 :, col_idx])
                        )
                        ** 2
                    )
                )
                alpha.append(
                    np.std(simulated[:, col_idx])
                    / np.std(response.values[windup_timesteps + 1 :, col_idx])
                )
                beta.append(
                    np.mean(simulated[:, col_idx])
                    / np.mean(response.values[windup_timesteps + 1 :, col_idx])
                )
                hfv.append(
                    100
                    * np.sum(
                        np.sort(simulated[:, col_idx])[-int(0.02 * len(index)) :]
                        - np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                            -int(0.02 * len(index)) :
                        ]
                    )
                    / np.sum(
                        np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                            -int(0.02 * len(index)) :
                        ]
                    )
                )
                hfv10.append(
                    100
                    * np.sum(
                        np.sort(simulated[:, col_idx])[-int(0.1 * len(index)) :]
                        - np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                            -int(0.1 * len(index)) :
                        ]
                    )
                    / np.sum(
                        np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                            -int(0.1 * len(index)) :
                        ]
                    )
                )
                lfv.append(
                    100
                    * np.sum(
                        np.sort(simulated[:, col_idx])[-int(0.3 * len(index)) :]
                        - np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                            -int(0.3 * len(index)) :
                        ]
                    )
                    / np.sum(
                        np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                            -int(0.3 * len(index)) :
                        ]
                    )
                )
                fdc.append(
                    100
                    * (
                        np.log10(
                            np.sort(simulated[:, col_idx])[int(0.2 * len(simulated))]
                        )
                        - np.log10(
                            np.sort(simulated[:, col_idx])[int(0.7 * len(simulated))]
                        )
                        - np.log10(
                            np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                                int(0.2 * len(simulated))
                            ]
                        )
                        + np.log10(
                            np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                                int(0.7 * len(simulated))
                            ]
                        )
                    )
                    / np.log10(
                        np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                            int(0.2 * len(simulated))
                        ]
                    )
                    - np.log10(
                        np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                            int(0.7 * len(simulated))
                        ]
                    )
                )

            print("MAE = ", mae)
            print("RMSE = ", rmse)
            print("NSE = ", nse)
            # alpha nse decomposition due to gupta et al 2009
            print("alpha = ", alpha)
            print("beta = ", beta)
            # top 2% peak flow bias (HFV) due to yilmaz et al 2008
            print("HFV = ", hfv)
            # top 10% peak flow bias (HFV) due to yilmaz et al 2008
            print("HFV10 = ", hfv10)
            # 30% low flow bias (LFV) due to yilmaz et al 2008
            print("LFV = ", lfv)
            # bias of FDC midsegment slope due to yilmaz et al 2008
            print("FDC = ", fdc)
            # compile all the error metrics into a dictionary
            error_metrics = {
                "MAE": mae,
                "RMSE": rmse,
                "NSE": nse,
                "alpha": alpha,
                "beta": beta,
                "HFV": hfv,
                "HFV10": hfv10,
                "LFV": lfv,
                "FDC": fdc,
                "r2": r2,
            }

        except Exception as e:  # and print the exception:
            print("Exception in simulation\n")
            print(e)
            error_metrics = {
                "MAE": [np.nan],
                "RMSE": [np.nan],
                "NSE": [np.nan],
                "alpha": [np.nan],
                "beta": [np.nan],
                "HFV": [np.nan],
                "HFV10": [np.nan],
                "LFV": [np.nan],
                "FDC": [np.nan],
                "r2": r2,
            }

            return {
                "error_metrics": error_metrics,
                "model": model,
                "simulated": response[1:],
                "response": response,
                "forcing": forcing,
                "index": index,
                "diverged": True,
            }

    return {
        "error_metrics": error_metrics,
        "model": model,
        "simulated": simulated,
        "response": response,
        "forcing": forcing,
        "index": index,
        "diverged": False,
    }
    # return [r2, model, mae, rmse, index, simulated , response , forcing]


def transform_inputs(
    shape_factors, scale_factors, loc_factors, index, forcing, *, cache=None
):
    """Vectorized implementation of transform_inputs for greater speed.

    Applies gamma PDF transformations to forcing inputs using FFT-based convolution
    instead of element-wise iteration. Optional LRU cache avoids recomputation for
    near-identical parameters during optimization.

    Args:
        shape_factors: DataFrame of gamma shape parameters
        scale_factors: DataFrame of gamma scale parameters
        loc_factors: DataFrame of gamma location parameters
        index: Time index
        forcing: DataFrame of forcing inputs
        cache: Optional TransformCache instance for memoization (default None)
    """
    # original forcing columns -> columns of forcing that don't have _tr_ in their name
    orig_forcing_columns = [col for col in forcing.columns if "_tr_" not in col]

    # how many rows of shape_factors do not contain NaNs?
    num_transforms = int(shape_factors.count().iloc[0])

    n = len(index)

    for input_col in orig_forcing_columns:
        forcing_values = forcing[input_col].to_numpy(dtype=float)

        for transform_idx in range(1, num_transforms + 1):
            col_name = f"{input_col}_tr_{transform_idx}"

            # Get gamma parameters
            shape = float(shape_factors[input_col][transform_idx])
            scale = float(scale_factors[input_col][transform_idx])
            loc = float(loc_factors[input_col][transform_idx])

            if cache is not None:
                # Use cached transform
                result = cache.get(input_col, forcing_values, shape, scale, loc)
            else:
                # Compute directly using FFT convolution (faster than np.convolve for large arrays)
                shape_time = np.arange(0, n, 1)
                gamma_kernel = stats.gamma.pdf(shape_time, shape, scale=scale, loc=loc)
                result = signal.fftconvolve(forcing_values, gamma_kernel, mode="full")[
                    :n
                ]

            forcing.loc[:, col_name] = result

    # assert there are no NaNs in the forcing
    if forcing.isnull().values.any():
        raise ValueError("Transform inputs produced NaN values")
    return forcing


# REQUIRES: the output of delay_io_train, starting value of otuput, forcing timeseries
# EFFECTS: returns a simulated response given forcing and a model
# REQUIRED EDITS: not written to accomodate transform_dependent yet
def delay_io_predict(
    delay_io_model,
    system_data,
    num_transforms=1,
    evaluation=False,
    windup_timesteps=None,
):
    if (
        windup_timesteps is None
    ):  # user didn't specify windup timesteps, use what the model trained with.
        windup_timesteps = delay_io_model[num_transforms]["windup_timesteps"]
    forcing = system_data[delay_io_model[num_transforms]["independent_columns"]].copy(
        deep=True
    )
    response = system_data[delay_io_model[num_transforms]["dependent_columns"]].copy(
        deep=True
    )

    # Use cache from model if available, otherwise create a new one
    transform_cache = delay_io_model[num_transforms].get("transform_cache", None)
    transformed_forcing = transform_inputs(
        shape_factors=delay_io_model[num_transforms]["shape_factors"],
        scale_factors=delay_io_model[num_transforms]["scale_factors"],
        loc_factors=delay_io_model[num_transforms]["loc_factors"],
        index=system_data.index,
        forcing=forcing,
        cache=transform_cache,
    )
    try:
        prediction = delay_io_model[num_transforms]["final_model"]["model"].simulate(
            system_data[delay_io_model[num_transforms]["dependent_columns"]].iloc[
                windup_timesteps, :
            ],
            t=np.arange(0, len(system_data.index), 1)[windup_timesteps:],
            u=transformed_forcing[windup_timesteps:],
        )
    except Exception as e:  # and print the exception:
        print("Exception in simulation\n")
        print(e)
        print("diverged.")
        error_metrics = {
            "MAE": [np.nan],
            "RMSE": [np.nan],
            "NSE": [np.nan],
            "alpha": [np.nan],
            "beta": [np.nan],
            "HFV": [np.nan],
            "HFV10": [np.nan],
            "LFV": [np.nan],
            "FDC": [np.nan],
        }
        return {
            "prediction": np.nan
            * np.ones(shape=response[windup_timesteps + 1 :].shape),
            "error_metrics": error_metrics,
            "diverged": True,
        }

    # return all the error metrics if the prediction is being evaluated against known measurements
    if evaluation:
        try:
            mae = list()
            rmse = list()
            nse = list()
            alpha = list()
            beta = list()
            hfv = list()
            hfv10 = list()
            lfv = list()
            fdc = list()
            for col_idx in range(
                0, len(response.columns)
            ):  # univariate performance metrics
                error = (
                    response.values[windup_timesteps + 1 :, col_idx]
                    - prediction[:, col_idx]
                )

                initial_error_length = len(error)
                error = error[~np.isnan(error)]
                if len(error) < 0.75 * initial_error_length:
                    print("WARNING: More than 25% of the entries in error were NaN")

                # print("error")
                # print(error)
                # nash sutcliffe efficiency between response and prediction
                mae.append(np.mean(np.abs(error)))
                rmse.append(np.sqrt(np.mean(error**2)))
                # print("mean measured = ", np.mean(response.values[windup_timesteps+1:,col_idx]  ))
                # print("sum of squared error between measured and model = ", np.sum((error)**2 ))
                # print("sum of squared error between measured and mean of measured = ", np.sum((response.values[windup_timesteps+1:,col_idx]-np.mean(response.values[windup_timesteps+1:,col_idx]  ) )**2 ))
                nse.append(
                    1
                    - np.sum((error) ** 2)
                    / np.sum(
                        (
                            response.values[windup_timesteps + 1 :, col_idx]
                            - np.mean(response.values[windup_timesteps + 1 :, col_idx])
                        )
                        ** 2
                    )
                )
                alpha.append(
                    np.std(prediction[:, col_idx])
                    / np.std(response.values[windup_timesteps + 1 :, col_idx])
                )
                beta.append(
                    np.mean(prediction[:, col_idx])
                    / np.mean(response.values[windup_timesteps + 1 :, col_idx])
                )
                hfv.append(
                    np.sum(
                        np.sort(prediction[:, col_idx])[
                            -int(0.02 * len(system_data.index)) :
                        ]
                    )
                    / np.sum(
                        np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                            -int(0.02 * len(system_data.index)) :
                        ]
                    )
                )
                hfv10.append(
                    np.sum(
                        np.sort(prediction[:, col_idx])[
                            -int(0.1 * len(system_data.index)) :
                        ]
                    )
                    / np.sum(
                        np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                            -int(0.1 * len(system_data.index)) :
                        ]
                    )
                )
                lfv.append(
                    np.sum(
                        np.sort(prediction[:, col_idx])[
                            : int(0.3 * len(system_data.index))
                        ]
                    )
                    / np.sum(
                        np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                            : int(0.3 * len(system_data.index))
                        ]
                    )
                )
                fdc.append(
                    np.mean(
                        np.sort(prediction[:, col_idx])[
                            -int(0.6 * len(system_data.index)) : -int(
                                0.4 * len(system_data.index)
                            )
                        ]
                    )
                    / np.mean(
                        np.sort(response.values[windup_timesteps + 1 :, col_idx])[
                            -int(0.6 * len(system_data.index)) : -int(
                                0.4 * len(system_data.index)
                            )
                        ]
                    )
                )

            print("MAE = ", mae)
            print("RMSE = ", rmse)

            print("NSE = ", nse)
            # alpha nse decomposition due to gupta et al 2009
            print("alpha = ", alpha)
            print("beta = ", beta)
            # top 2% peak flow bias (HFV) due to yilmaz et al 2008
            print("HFV = ", hfv)
            # top 10% peak flow bias (HFV) due to yilmaz et al 2008
            print("HFV10 = ", hfv10)
            # 30% low flow bias (LFV) due to yilmaz et al 2008
            print("LFV = ", lfv)
            # bias of FDC midsegment slope due to yilmaz et al 2008
            print("FDC = ", fdc)
            # compile all the error metrics into a dictionary
            error_metrics = {
                "MAE": mae,
                "RMSE": rmse,
                "NSE": nse,
                "alpha": alpha,
                "beta": beta,
                "HFV": hfv,
                "HFV10": hfv10,
                "LFV": lfv,
                "FDC": fdc,
            }
            # omit r2 here because it doesn't mean the same thing as it does for training, would be misleading.
            # r2 in training expresses how much of the derivative is predicted by the model, whereas in evaluation it expresses how much of the response is predicted by the model

            return {
                "prediction": prediction,
                "error_metrics": error_metrics,
                "diverged": False,
            }
        except Exception as e:  # and print the exception:
            print(e)
            print("Simulation diverged.")
            error_metrics = {
                "MAE": [np.nan],
                "RMSE": [np.nan],
                "NSE": [np.nan],
                "alpha": [np.nan],
                "beta": [np.nan],
                "HFV": [np.nan],
                "HFV10": [np.nan],
                "LFV": [np.nan],
                "FDC": [np.nan],
                "diverged": [True],
            }

            return {"prediction": prediction, "error_metrics": error_metrics}
    else:
        error_metrics = {
            "MAE": [np.nan],
            "RMSE": [np.nan],
            "NSE": [np.nan],
            "alpha": [np.nan],
            "beta": [np.nan],
            "HFV": [np.nan],
            "HFV10": [np.nan],
            "LFV": [np.nan],
            "FDC": [np.nan],
        }
        return {
            "prediction": prediction,
            "error_metrics": error_metrics,
            "diverged": False,
        }


### the functions below are for generating LTI systems directly from data (aka system identification)


# the function below returns an LTI system (in the matrices A, B, and C) that mimic the shape of a given gamma distribution
# scaling should be correct, but need to verify that
# max state dim, resolution, and max iterations could be icnrased to improve accuracy
def lti_from_gamma(
    shape,
    scale,
    location,
    dt=0,
    desired_NSE=0.999,
    verbose=False,
    max_state_dim=50,
    max_iterations=200,
    max_pole_speed=5,
    min_pole_speed=0.01,
):
    # a pole of speed -5 decays to less than 1% of it's value after one timestep
    # a pole of speed -0.01 decays to more than 99% of it's value after one timestep

    # i've assumed here that gamma pdf is defined the same as in matlab
    # if that's not true testing will show it soon enough
    t50 = shape * scale + location  # center of mass
    skewness = 2 / np.sqrt(shape)
    total_time_base = (
        2 * t50
    )  # not that this contains the full shape, but if we fit this much of the curve perfectly we'll be close enough
    # resolution = (t50)/((skewness + location)) # make this coarser for faster debugging
    resolution = (t50) / (10 * (skewness + location))  # production version

    # resolution = 1/ skewness
    decay_rate = 1 / resolution
    decay_rate = np.clip(decay_rate, min_pole_speed, max_pole_speed)
    state_dim = int(
        np.floor(total_time_base * decay_rate)
    )  # this keeps the time base fixed for a given decay rate
    if state_dim > max_state_dim:
        state_dim = max_state_dim
        decay_rate = state_dim / total_time_base
        resolution = 1 / decay_rate
    if state_dim < 1:
        state_dim = 1
        decay_rate = state_dim / total_time_base
        resolution = 1 / decay_rate

    decay_rate = np.clip(decay_rate, min_pole_speed, max_pole_speed)

    if verbose:
        print("state dimension is ", state_dim)
        print("decay rate is ", decay_rate)
        print("total time base is ", total_time_base)
        print("resolution is", resolution)

    # make the timestep one so that the relative error is correct (dt too small makes error bigger than written)
    # t = np.linspace(0,3*total_time_base,1000)
    # desired_error = desired_error / dt
    """
    if dt > 0: # true if numeric
        t = np.arange(0,2*total_time_base,dt)
    else:
        t= np.linspace(0,2*total_time_base,num=200)
    """
    t = np.linspace(0, 2 * total_time_base, num=200)

    # if verbose:
    #    print("dt is ",dt)
    #    print("scaled desired error is ",desired_error)

    gam = stats.gamma.pdf(t, shape, location, scale)

    # A is a cascade with the appropriate decay rate
    A = decay_rate * np.diag(np.ones((state_dim - 1)), -1) - decay_rate * np.diag(
        np.ones((state_dim)), 0
    )
    # influence enters at the top state only
    B = np.concatenate((np.ones((1, 1)), np.zeros((state_dim - 1, 1))))
    # contributions of states to the output will be scaled to match the gamma distribution
    C = np.ones((1, state_dim)) * max(gam)
    lti_sys = control.ss(A, B, C, 0)

    lti_approx = control.impulse_response(lti_sys, t)
    """
    error = np.sum(np.abs(gam - lti_approx.y))
    if(verbose):
        print("initial error")
        print(error)
        #print("desired error")
        #print(max(gam))
        #print(desired_error)
        """
    NSE = 1 - (
        np.sum(np.square(gam - lti_approx.y)) / np.sum(np.square(gam - np.mean(gam)))
    )
    # if NSE is nan, set to -10e6
    if np.isnan(NSE):
        NSE = -10e6

    if verbose:
        print("initial NSE")
        print(NSE)
        print("desired NSE")
        print(desired_NSE)

    iterations = 0

    speeds = [10, 5, 2, 1.1, 1.05, 1.01, 1.001]
    speed_idx = 0
    leap = speeds[speed_idx]
    # the area under the curve is normalized to be one. so rather than basing our desired error off the
    # max of the distribution, it might be better to make it a percentage error, one percent or five percent
    while NSE < desired_NSE and iterations < max_iterations:

        og_was_best = (
            True  # start each iteration assuming that the original is the best
        )
        # search across the C vector
        for i in range(
            C.shape[1] - 1, int(-1), int(-1)
        ):  # accross the columns # start at the end and come back
            # for i in range(int(0),C.shape[1],int(1)): # accross the columns, start at the beginning and go forward

            og_approx = control.ss(A, B, C, 0)
            og_y = np.ndarray.flatten(control.impulse_response(og_approx, t).y)
            og_error = np.sum(np.abs(gam - og_y))
            og_NSE = 1 - (np.sum((gam - og_y) ** 2) / np.sum((gam - np.mean(gam)) ** 2))

            Ctwice = np.array(C, copy=True)
            Ctwice[0, i] = leap * C[0, i]
            twice_approx = control.ss(A, B, Ctwice, 0)
            twice_y = np.ndarray.flatten(control.impulse_response(twice_approx, t).y)
            np.sum(np.abs(gam - twice_y))
            twice_NSE = 1 - (
                np.sum((gam - twice_y) ** 2) / np.sum((gam - np.mean(gam)) ** 2)
            )

            Chalf = np.array(C, copy=True)
            Chalf[0, i] = (1 / leap) * C[0, i]
            half_approx = control.ss(A, B, Chalf, 0)
            half_y = np.ndarray.flatten(control.impulse_response(half_approx, t).y)
            np.sum(np.abs(gam - half_y))
            half_NSE = 1 - (
                np.sum((gam - half_y) ** 2) / np.sum((gam - np.mean(gam)) ** 2)
            )
            """
            Cneg = np.array(C,copy=True)
            Cneg[0,i] = -C[0,i]
            neg_approx = control.ss(A,B,Cneg,0)
            neg_y = np.ndarray.flatten(control.impulse_response(neg_approx,t).y)
            neg_error = np.sum(np.abs(gam - neg_y))
            neg_NSE = 1 - (np.sum((gam - neg_y)**2) / np.sum((gam - np.mean(gam))**2))
            """
            faster = np.array(A, copy=True)
            faster[i, i] = A[i, i] * leap  # faster decay
            if abs(faster[i, i]) < abs(max_pole_speed):
                if (
                    i > 0
                ):  # first reservoir doesn't receive contribution from another reservoir. want to keep B at 1 for scaling
                    faster[i, i - 1] = A[i, i - 1] * leap  # faster rise
                faster_approx = control.ss(faster, B, C, 0)
                faster_y = np.ndarray.flatten(
                    control.impulse_response(faster_approx, t).y
                )
                np.sum(np.abs(gam - faster_y))
                faster_NSE = 1 - (
                    np.sum((gam - faster_y) ** 2) / np.sum((gam - np.mean(gam)) ** 2)
                )
            else:
                faster_NSE = -10e6  # disallowed because the pole is too fast

            slower = np.array(A, copy=True)
            slower[i, i] = A[i, i] / leap  # slower decay
            if abs(slower[i, i]) > abs(min_pole_speed):
                if i > 0:
                    slower[i, i - 1] = A[i, i - 1] / leap  # slower rise
                slower_approx = control.ss(slower, B, C, 0)
                slower_y = np.ndarray.flatten(
                    control.impulse_response(slower_approx, t).y
                )
                np.sum(np.abs(gam - slower_y))
                slower_NSE = 1 - (
                    np.sum((gam - slower_y) ** 2) / np.sum((gam - np.mean(gam)) ** 2)
                )
            else:
                slower_NSE = -10e6  # disallowed because the pole is too slow

            # all_errors = [og_error, twice_error, half_error, faster_error, slower_error]
            all_NSE = [
                og_NSE,
                twice_NSE,
                half_NSE,
                faster_NSE,
                slower_NSE,
            ]  # , neg_NSE]

            if twice_NSE >= max(all_NSE) and twice_NSE > og_NSE:
                C = Ctwice
                if twice_NSE > 1.001 * og_NSE:  # an appreciable difference
                    og_was_best = False  # did we change something this iteration?
            elif half_NSE >= max(all_NSE) and half_NSE > og_NSE:
                C = Chalf
                if half_NSE > 1.001 * og_NSE:  # an appreciable difference
                    og_was_best = False  # did we change something this iteration?

            elif slower_NSE >= max(all_NSE) and slower_NSE > og_NSE:
                A = slower
                if slower_NSE > 1.001 * og_NSE:  # an appreciable difference
                    og_was_best = False  # did we change something this iteration?
            elif faster_NSE >= max(all_NSE) and faster_NSE > og_NSE:
                A = faster
                if faster_NSE > 1.001 * og_NSE:  # an appreciable difference
                    og_was_best = False  # did we change something this iteration?
                    """
            elif (neg_NSE >= max(all_NSE) and neg_NSE > og_NSE):
                C = Cneg
                if neg_NSE > 1.001*og_NSE:
                    og_was_best = False
                    """

        NSE = og_NSE
        error = og_error
        iterations += 1  # this shouldn't be the termination condition unless the resolution is too coarse
        # normally the optimization should exit because the leap has become too small
        if (
            og_was_best
        ):  # the original was the best, so we're going to tighten up the optimization
            speed_idx += 1
            if speed_idx > len(speeds) - 1:
                break  # we're done
            leap = speeds[speed_idx]
        # print the iteration count every ten
        # comment out for production
        if iterations % 2 == 0 and verbose:
            print("iterations = ", iterations)
            print("error = ", error)
            print("NSE = ", NSE)
            print("leap = ", leap)

    lti_approx = control.ss(A, B, C, 0)
    y = np.ndarray.flatten(control.impulse_response(og_approx, t).y)
    error = np.sum(np.abs(gam - og_y))
    print("LTI_from_gamma final NSE")
    print(NSE)
    if verbose:
        print("final system\n")
        print("A")
        print(A)
        print("B")
        print(B)
        print("C")
        print(C)

        print("\nfinal error")
        print(error)

    # are any of the final eigenvalues outside the bounds specified?
    E = np.linalg.eigvals(A)
    if np.any(np.abs(E) > max_pole_speed) or np.any(np.abs(E) < min_pole_speed):
        print("WARNING: final eigenvalues are outside the bounds specified")

    return {
        "lti_approx": lti_approx,
        "lti_approx_output": y,
        "error": error,
        "t": t,
        "gamma_pdf": gam,
    }


# this function takes the system data and the causative topology and returns an LTI system
# if the causative topology isn't already defined, it needs to be created using infer_causative_topology
def lti_system_gen(
    causative_topology,
    system_data,
    independent_columns,
    dependent_columns,
    max_iter=250,
    swmm=False,
    bibo_stable=False,
    max_transition_state_dim=50,
    max_transforms=1,
    early_stopping_threshold=0.005,
):

    # cast the columns and indices of causative_topology to strings so sindy can run properly
    # We need the tuples to link the columns in system_data to the object names in the swmm model
    # so we'll cast these back to tuples once we're done
    if swmm:
        causative_topology.columns = causative_topology.columns.astype(str)
        causative_topology.index = causative_topology.index.astype(str)

        print("causative topology \n")
        print(causative_topology.index)
        print(causative_topology.columns)

        # do the same for dependent_columns and independent_columns
        dependent_columns = [str(col) for col in dependent_columns]
        independent_columns = [str(col) for col in independent_columns]
        print(dependent_columns)
        print(independent_columns)

        # do the same for the columns of system_data
        system_data.columns = system_data.columns.astype(str)
        print(system_data.columns)

    A = pd.DataFrame(index=dependent_columns, columns=dependent_columns)
    B = pd.DataFrame(index=dependent_columns, columns=independent_columns)
    C = pd.DataFrame(index=dependent_columns, columns=dependent_columns)
    C.loc[:, :] = np.diag(
        np.ones(len(dependent_columns))
    )  # these are the states which are observable

    # copy the corresponding entries from the causative topology into B
    for row in B.index:
        for col in B.columns:
            B.loc[row, col] = causative_topology.loc[row, col]
    # and into A
    for row in A.index:
        for col in A.columns:
            A.loc[row, col] = causative_topology.loc[row, col]

    print("A")
    print(A)
    print("B")
    print(B)
    print("C")
    print(C)
    # use transform_only when calling delay_io_train to only train transfomrations for connections marked "d"
    # train a MISO model for each output
    delay_models: dict[Any, Any] = {key: None for key in dependent_columns}

    for row in A.index:
        immediate_forcing = []
        delayed_forcing = []
        for col in A.columns:
            if col == row:
                continue  # don't need to include the output state as a forcing variable. it's already included by default
            if A[col][row] == "d":
                delayed_forcing.append(col)
            elif A[col][row] == "i":
                immediate_forcing.append(col)
        for col in B.columns:
            if B[col][row] == "d":
                delayed_forcing.append(col)
            elif B[col][row] == "i":
                immediate_forcing.append(col)
        # make total_forcing the union of immediate and delayed forcing
        total_forcing = immediate_forcing + delayed_forcing
        feature_names = [row] + total_forcing
        if delayed_forcing:
            print(
                "training delayed model for ",
                row,
                " with forcing ",
                total_forcing,
                "\n",
            )
            delay_models[row] = delay_io_train(
                system_data,
                [row],
                total_forcing,
                transform_only=delayed_forcing,
                max_transforms=max_transforms,
                poly_order=1,
                max_iter=max_iter,
                verbose=False,
                bibo_stable=bibo_stable,
            )
            # we'll parse this delayed causation into the matrices A, B, and C later
        else:
            ####### TODO: incorporate bibo stability constraint into instantaneous fits ########
            print(
                "training immediate model for ",
                row,
                " with forcing ",
                total_forcing,
                "\n",
            )
            delay_models[row] = None
            # we can put immediate causation into the matrices A, B, and C now

            if bibo_stable:  # negative autocorrelatoin
                # Figure out how many library features there will be
                library = ps.PolynomialLibrary(
                    degree=1, include_bias=False, include_interaction=False
                )
                # total_train = pd.concat((response,forcing), axis='columns')
                # fit on a dummy (2, n_features) array; 2 rows is the minimum pysindy requires
                library.fit(np.zeros((2, len(feature_names))))
                n_features = library.n_output_features_
                # print(f"Features ({n_features}):", library.get_feature_names())
                # Set constraints
                # n_targets = total_train.shape[1] # not sure what targets means after reading through the pysindy docs
                # print("n_targets")
                # print(n_targets)
                constraint_rhs = np.zeros(1)
                # one row per constraint, one column per coefficient
                constraint_lhs = np.zeros((1, n_features))

                # print(constraint_rhs)
                # print(constraint_lhs)
                # constrain the highest order output autocorrelation to be negative
                # this indexing is only right for include_interaction=False, include_bias=False, and pure polynomial library
                # for more complex libraries, some conditional logic will be needed to grab the right column
                constraint_lhs[:, 0] = 1

                model = ps.SINDy(
                    differentiation_method=ps.FiniteDifference(),
                    feature_library=ps.PolynomialLibrary(
                        degree=1, include_bias=False, include_interaction=False
                    ),
                    optimizer=_ConstrainedSR3(
                        reg_weight_lam=0,
                        regularizer="l2",
                        constraint_lhs=constraint_lhs,
                        constraint_rhs=constraint_rhs,
                        inequality_constraints=True,
                    ),
                )

            else:  # unoconstrained
                model = ps.SINDy(
                    differentiation_method=ps.FiniteDifference(
                        order=10, drop_endpoints=True
                    ),
                    feature_library=ps.PolynomialLibrary(
                        degree=1, include_bias=False, include_interaction=False
                    ),
                    optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
                )
            if system_data.loc[
                :, immediate_forcing
            ].empty:  # the subsystem is autonomous
                instant_fit = model.fit(
                    x=system_data.loc[:, row], t=np.arange(0, len(system_data.index), 1)
                )
                instant_fit.print(precision=3)
                print(
                    "Training r2 = ",
                    instant_fit.score(
                        x=system_data.loc[:, row],
                        t=np.arange(0, len(system_data.index), 1),
                    ),
                )
                print(instant_fit.coefficients())
            else:  # there is some forcing
                # instant_fit = model.fit(x = system_data.loc[:,row] ,t = system_data.index.values, u = system_data.loc[:,immediate_forcing]) # sindy can't take datetime indices
                instant_fit = model.fit(
                    x=system_data.loc[:, row],
                    t=np.arange(0, len(system_data.index), 1),
                    u=system_data.loc[:, immediate_forcing],
                )
                instant_fit.print(precision=3)
                print(
                    "Training r2 = ",
                    instant_fit.score(
                        x=system_data.loc[:, row],
                        t=np.arange(0, len(system_data.index), 1),
                        u=system_data.loc[:, immediate_forcing],
                    ),
                )
                print(instant_fit.coefficients())
            for idx in range(len(feature_names)):
                if feature_names[idx] in A.columns:
                    A.loc[row, feature_names[idx]] = instant_fit.coefficients()[0][idx]
                elif feature_names[idx] in B.columns:
                    B.loc[row, feature_names[idx]] = instant_fit.coefficients()[0][idx]
                else:
                    print("couldn't find a column for ", feature_names[idx])
            # print("updated A")
            # print(A)
            # print("updated B")
            # print(B)

    original_A = A.copy(deep=True)
    # now, parse the delay models into the A, B, and C matrices
    # the changes will be as follows:
    # the A matrix will have matrices of the form [B_gam, A_gam; 0 , C_gam] inserted into it
    # where X_gam are the matrices generated by the lti_from_gamma function to represent the delayed causation shape
    # the B and C matrices will just have zeros inserted into them to maintain compatible dimensions
    # none of these cascades are observable or directly receive input.
    for row in original_A.index:
        if delay_models[row] is None:
            pass
        else:  # we want the model with the most transformations where the last trnasformation added at least 0.5% to the R2 score
            for num_transforms in range(1, max_transforms + 1):
                if num_transforms == 1:
                    optimal_number_transforms = num_transforms
                elif (
                    delay_models[row][num_transforms]["final_model"]["error_metrics"][
                        "r2"
                    ]
                    - delay_models[row][num_transforms - 1]["final_model"][
                        "error_metrics"
                    ]["r2"]
                    < early_stopping_threshold
                ):
                    optimal_number_transforms = num_transforms - 1
                    break  # improvement is too small to justify additional complexity
                else:
                    optimal_number_transforms = (
                        num_transforms  # the most recent one was worth it
                    )

            transformation_approximations: dict[Any, Any] = {
                transform_key: None
                for transform_key in delay_models[row][optimal_number_transforms][
                    "shape_factors"
                ].columns
            }
            for transform_key in transformation_approximations.keys():  # which input
                for idx in range(
                    1, optimal_number_transforms + 1
                ):  # which transformation
                    print("variable = ", transform_key, ", transformation = ", idx)
                    delay_models[row][optimal_number_transforms]["final_model"][
                        "model"
                    ].print(precision=5)
                    shape = delay_models[row][optimal_number_transforms][
                        "shape_factors"
                    ].loc[idx, transform_key]
                    scale = delay_models[row][optimal_number_transforms][
                        "scale_factors"
                    ].loc[idx, transform_key]
                    loc = delay_models[row][optimal_number_transforms][
                        "loc_factors"
                    ].loc[idx, transform_key]
                    """
                    # infer the timestep of system_data from the index
                    timestep = system_data.index[1] - system_data.index[0]
                    try: # if the timestep is numeric
                        pd.to_numeric(timestep)
                        transformation_approximations[transform_key] = lti_from_gamma(shape,scale,loc,dt=timestep)

                        Agam = transformation_approximations[transform_key]['lti_approx'].A / timestep
                        Bgam = transformation_approximations[transform_key]['lti_approx'].B / timestep
                        Cgam = transformation_approximations[transform_key]['lti_approx'].C / timestep
                    except Exception as e: # if the timestep is something like a datetime
                        print(e)"""
                    # this will get overwritten if we use more than one transformation per input. i think that's okay.
                    transformation_approximations[transform_key] = lti_from_gamma(
                        shape, scale, loc, max_state_dim=max_transition_state_dim
                    )

                    Agam = transformation_approximations[transform_key]["lti_approx"].A
                    Bgam = transformation_approximations[transform_key][
                        "lti_approx"
                    ].B  # only entry is unit impulse at top state
                    Cgam = transformation_approximations[transform_key]["lti_approx"].C

                    tr_string = str("_tr_" + str(idx))

                    # Cgam needs to be scaled by the coefficient the forcing term had in the delay model
                    # coefficients = {coef_key: None for coef_key in delay_models[row][1]['final_model']['model'].feature_names}
                    coefficients = {
                        coef_key: None
                        for coef_key in delay_models[row][optimal_number_transforms][
                            "final_model"
                        ]["model"].feature_names
                    }
                    for coef_key in coefficients.keys():
                        coef_index = delay_models[row][optimal_number_transforms][
                            "final_model"
                        ]["model"].feature_names.index(coef_key)
                        coefficients[coef_key] = delay_models[row][
                            optimal_number_transforms
                        ]["final_model"]["model"].coefficients()[0][coef_index]
                        # if "_tr_1" in coef_key and coef_key.replace("_tr_1","") == transform_key.replace("_tr_1",""):
                        if tr_string in coef_key and coef_key.replace(
                            tr_string, ""
                        ) == transform_key.replace(tr_string, ""):
                            """
                            try:
                                pd.to_numeric(timestep,errors='raise')
                                Cgam = Cgam * coefficients[coef_key] / timestep
                            except Exception as e:
                                print(e)
                                Cgam = Cgam * coefficients[coef_key]
                            """

                            Cgam = Cgam * coefficients[coef_key]  # scaling
                        else:  # these are the immediate effects, insert them now
                            if coef_key in A.columns:
                                A.loc[row, coef_key] = coefficients[coef_key]
                            elif coef_key in B.columns:
                                B.loc[row, coef_key] = coefficients[coef_key]

                    Agam_index = []
                    for agam_idx in range(Agam.shape[0]):
                        # Agam_index.append(transform_key.replace("_tr_1","") + "->" + row + "_" + str(idx))
                        Agam_index.append(
                            transform_key.replace(tr_string, "")
                            + "->"
                            + row
                            + tr_string
                            + "_"
                            + str(agam_idx)
                        )
                    Agam = pd.DataFrame(Agam, index=Agam_index, columns=Agam_index)
                    Bgam = pd.DataFrame(
                        Bgam,
                        index=Agam_index,
                        columns=[transform_key.replace(tr_string, "")],
                    )
                    Cgam = pd.DataFrame(Cgam, index=[row], columns=Agam_index)
                    # print("Agam")
                    # print(Agam)
                    # print("Bgam")
                    # print(Bgam)
                    # print("Cgam")
                    # print(Cgam)
                    # insert these into the A, B, and C matrices
                    # for Agam, the insertion row is immediately after the source (key)
                    # the insertion column is also immediately after the source (key)

                    ### everything below this point is garbage. not performing at all as desired at the moment

                    # first need to create space for the new rows and columns
                    # create before_index and after_index variables, which record the parts of the index of A that occur before and after row
                    before_index = []
                    # after_index = []
                    # if transform_key.replace("_tr_1","") not in A.index: # it's one of the forcing terms. put it in at the beginning
                    if (
                        transform_key.replace(tr_string, "") not in A.index
                    ):  # it's one of the forcing terms. put it in at the beginning
                        after_index = list(
                            A.index
                        )  # it's a forcing variable, so we don't want it in the newA index
                    else:  # it is a state variable
                        # before_index = list(A.index[:A.index.get_loc(transform_key.replace("_tr_1",""))])
                        before_index = list(
                            A.index[
                                : A.index.get_loc(transform_key.replace(tr_string, ""))
                            ]
                        )

                        # after_index = list(A.index[A.index.get_loc(transform_key.replace("_tr_1",""))+1:])
                        after_index = list(
                            A.index[
                                A.index.get_loc(transform_key.replace(tr_string, ""))
                                + 1 :
                            ]
                        )

                        """
                        for idx in A.index:
                            if idx == key.replace("_tr_1",""):
                                before_index.append(idx) # if it's a state variable, we want it in the newA index
                                break
                            else:
                                before_index.append(idx)
                        for idx in range(A.index.get_loc(key.replace("_tr_1",""))+1,len(A.index)):
                            after_index.append(A.index[idx])
                            """
                    # if transform_key.replace("_tr_1","") in A.index: # the transform key refers to a state (x)
                    if transform_key.replace(tr_string, "") in A.index:
                        # states = before_index + [transform_key.replace("_tr_1","")] + Agam_index + after_index # state dim expands by the number of rows in Agam
                        states = (
                            before_index
                            + [transform_key.replace(tr_string, "")]
                            + Agam_index
                            + after_index
                        )  # state dim expands by the number of rows in Agam
                        # include the current transform key in A because it's a state variable
                    # elif transform_key.replace("_tr_1","") in B.columns: # the transform key refers to a control input (u)
                    elif (
                        transform_key.replace(tr_string, "") in B.columns
                    ):  # the transform key refers to a control input (u)
                        states = (
                            before_index + Agam_index + after_index
                        )  # state dim expands by the number of rows in Agam
                        # don't include the current transform key in A because it's a control input, not a state variable

                    newA = pd.DataFrame(index=states, columns=states)
                    newB = pd.DataFrame(
                        index=states, columns=B.columns
                    )  # input dim remains consistent (columns of B)
                    newC = pd.DataFrame(
                        index=C.index, columns=states
                    )  # output dim remains consistent (rows of C)

                    # fill in newA with the corresponding entries from A
                    for idx in newA.index:
                        for col in newA.columns:
                            if (
                                idx in A.index and col in A.columns
                            ):  # if it's in the original A matrix, copy it over
                                newA.loc[idx, col] = A.loc[idx, col]
                            if (
                                idx in Agam.index and col in Agam.columns
                            ):  # if it's in Agam, copy it over
                                newA.loc[idx, col] = Agam.loc[idx, col]
                            if (
                                idx in Bgam.index and col in Bgam.columns
                            ):  # the input to the cascade is a state
                                newA.loc[idx, col] = Bgam.loc[idx, col]

                    for idx in newB.index:
                        for col in newB.columns:
                            if (
                                idx in B.index and col in B.columns
                            ):  # if it's in the original B matrix, copy it over
                                newB.loc[idx, col] = B.loc[idx, col]
                            if (
                                idx in Bgam.index and col in Bgam.columns
                            ):  # the input to the cascade is a forcing term
                                newB.loc[idx, col] = Bgam.loc[idx, col]

                    for idx in newC.index:
                        for col in newC.columns:
                            if (
                                idx in C.index and col in C.columns
                            ):  # if it's in the original C matrix, copy it over
                                newC.loc[idx, col] = C.loc[idx, col]
                            if (
                                idx in Cgam.index and col in Cgam.columns
                            ):  # outputs from the cascades
                                newA.loc[idx, col] = Cgam.loc[idx, col]

                    # print("newA")
                    # print(newA.to_string())
                    # print("newB")
                    # print(newB.to_string())
                    # print("newC")
                    # print(newC.to_string())

                    # copy over
                    A = newA.copy(deep=True)
                    B = newB.copy(deep=True)
                    C = newC.copy(deep=True)

    A.replace("n", 0.0, inplace=True)
    B.replace("n", 0.0, inplace=True)
    C.replace("n", 0.0, inplace=True)

    if swmm:
        pass
        #############
        # TODO: cast strings back to tuples in the indices and columns
        #############
        # cast the index and columns of causative_topology to tuples. they'll be of the form "(X,Y)"

        # do the same for dependent_columns and independent_columns

        # do the same for the columns of system_data

    A = A.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    B = B.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    C = C.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # if bibo_stable is specified and A not hurwitz, make A hurwitz by defining A' = A - I*max(real(eig(A)))
    # this will gaurantee stability (max eigenvalue will have real part < 0)
    if bibo_stable:
        orig_eigs, _ = np.linalg.eig(A)
        if any(np.real(orig_eigs) > 0):
            print("stabilizing unstable plant by subtracting I*max(real(eig)) from A")
            epsilon = 10e-4
            A_stab = A - np.eye(len(A)) * (1 + epsilon) * max(
                np.real(orig_eigs)
            )  # add factor of (1+epsilon) for stability, not marginal stabilty
            stab_eigs, _ = np.linalg.eig(A_stab)
            A = A_stab.copy(deep=True)

    # sindy will scale the coefficients according to the timestep if the index is numeric
    # so the whole system needs to be scaled by the timestep if its numeric
    try:
        pd.to_numeric(
            system_data.index, errors="raise"
        )  # can the index be converted to a numeric type?
        dt = system_data.index.values[1] - system_data.index.values[0]
        A = A / dt
        B = B / dt
        C = C  # what we observe doesn't need to be adjusted, just the dynamics
        print("system response data index converted to numeric type. dt = ")
        print(dt)
    except Exception as e:
        print(e)
        dt = None

    # cast all of A, B, and C to type float (integers cause issues with LQR / LQE calculations)
    A = A.astype(float)
    B = B.astype(float)
    C = C.astype(float)

    lti_sys = control.ss(
        A, B, C, 0, inputs=B.columns, outputs=C.index, states=A.columns
    )

    # returning the matrices too because control.ss strips the labels from the pandas dataframes and stores them as numpy matrices
    return {"system": lti_sys, "A": A, "B": B, "C": C}


# Legacy function kept for backward compatibility - DEPRECATED
def find_topology(
    system_data,
    dependent_columns,
    independent_columns,
    method="ccm",
    graph_type="Weak-Conn",
    verbose=False,
):
    """
    DEPRECATED: This function has been replaced by the improved SINDy-based
    topology inference via the local SINDy-based implementation.

    Please use infer_causative_topology(method='sindy') or directly import
    use infer_causative_topology().
    """
    import warnings

    warnings.warn(
        "find_topology() is deprecated. Use infer_causative_topology(method='sindy') "
        "or use infer_causative_topology() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Delegate to the new implementation
    return infer_causative_topology(
        system_data=system_data,
        dependent_columns=dependent_columns,
        independent_columns=independent_columns,
        graph_type=graph_type,
        verbose=verbose,
        method="sindy",
    )


def infer_causative_topology(
    system_data,
    dependent_columns,
    independent_columns,
    graph_type="Weak-Conn",
    verbose=False,
    method="sindy",
):
    from scipy.optimize import minimize

    # drop columns from system_data which aren't in dependent_columns or independent_columns
    # this ensures we only analyze the variables of interest
    system_data = pd.concat(
        (system_data[independent_columns], system_data[dependent_columns]),
        axis="columns",
    )

    # Create a shared cache for all transform_inputs calls in this function
    _topology_cache = TransformCache()

    # Store results for each column pair
    best_params = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=object
    )
    pd.DataFrame(index=dependent_columns, columns=system_data.columns, dtype=float)
    pd.DataFrame(index=dependent_columns, columns=system_data.columns, dtype=float)
    pd.DataFrame(index=dependent_columns, columns=system_data.columns, dtype=float)
    r2_values = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=float
    )
    edges = pd.DataFrame(
        index=system_data.columns, columns=system_data.columns, dtype=int, data=0
    )  # from column, to row. causation, not flow.

    for dep_col in dependent_columns:
        np.array(system_data[dep_col].values)

        for forcing_col in system_data.columns:
            if forcing_col == dep_col:
                continue  # autocorrelation is always included in the sindy fit

            print(f"\nOptimizing transformation for {forcing_col} -> {dep_col}")
            forcing_orig = system_data[[forcing_col]].copy(deep=True)

            # Objective function to minimize (negative because we want to maximize correlation - p_value)
            def objective(params):
                shape, scale, loc = params

                # Create transformation parameter DataFrames
                shape_factors = pd.DataFrame(columns=[forcing_col], index=[1])
                shape_factors.loc[1, forcing_col] = shape
                scale_factors = pd.DataFrame(columns=[forcing_col], index=[1])
                scale_factors.loc[1, forcing_col] = scale
                loc_factors = pd.DataFrame(columns=[forcing_col], index=[1])
                loc_factors.loc[1, forcing_col] = loc

                try:
                    # build the candidate input set
                    # selected_inputs = list(edges.loc[output_variable, edges.loc[output_variable,:] == 1].index)
                    # candidate_inputs = selected_inputs + [forcing_variable]
                    # build the transformed timeseries for these candidate inputs using the best transformation parameters found earlier
                    transformed_inputs = pd.DataFrame(index=system_data.index)
                    """
                    for input_var in candidate_inputs:
                        shape, scale, loc = best_params.loc[output_variable, input_var]
                        shape_factors = pd.DataFrame(columns=[input_var], index=[1])
                        shape_factors.loc[1, input_var] = shape
                        scale_factors = pd.DataFrame(columns=[input_var], index=[1])
                        scale_factors.loc[1, input_var] = scale
                        loc_factors = pd.DataFrame(columns=[input_var], index=[1])
                        loc_factors.loc[1, input_var] = loc
                        forcing_orig = system_data[[input_var]].copy()
                        transformed = transform_inputs(shape_factors, scale_factors, loc_factors,
                                                        system_data.index, forcing_orig)
                        transformed_inputs = pd.concat((transformed_inputs, transformed[[input_var + "_tr_1"]]), axis='columns')
                    """
                    # SINDY way
                    transformed = transform_inputs(
                        shape_factors,
                        scale_factors,
                        loc_factors,
                        system_data.index,
                        forcing_orig,
                        cache=_topology_cache,
                    )
                    transformed_inputs = pd.concat(
                        (transformed_inputs, transformed[[forcing_col + "_tr_1"]]),
                        axis="columns",
                    )
                    # build a sindy model with these inputs
                    # feature_names = [output_variable] + candidate_inputs
                    feature_names = [dep_col, str(forcing_col + "_tr_1")]
                    model = ps.SINDy(
                        differentiation_method=ps.FiniteDifference(
                            order=10, drop_endpoints=True
                        ),
                        feature_library=ps.PolynomialLibrary(
                            degree=1, include_bias=False, include_interaction=False
                        ),
                        optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
                    )
                    # fit = model.fit(x = system_data.loc[:,output_variable] ,t = np.arange(0,len(system_data.index),1) , u = transformed_inputs,
                    #        feature_names = feature_names)
                    fit = model.fit(
                        x=system_data.loc[:, dep_col],
                        u=transformed_inputs,
                        t=np.arange(0, len(system_data.index), 1),
                        feature_names=feature_names,
                    )
                    r2 = fit.score(
                        x=system_data.loc[:, dep_col],
                        u=transformed_inputs,
                        t=np.arange(0, len(system_data.index), 1),
                    )
                    # model.print(precision=5)

                    """
                    # polynomial regression way (might be faster than sindy, doesn't consider autocorrelation)
                    forcing = np.array(transformed[forcing_col + "_tr_1"].values)


                    # fourth order polynomial regression between transformed forcing and derivative of response
                    coeffs = np.polyfit(forcing, np.gradient(response), 4)
                    r_value_poly = np.corrcoef(np.polyval(coeffs, forcing), np.gradient(response))[0, 1]

                    # r^2 likely makes more sense as our criterion.
                    r2 = sklearn.metrics.r2_score(np.gradient(response), np.polyval(coeffs, forcing))
                    """

                    return -r2  # Negative because minimize
                except Exception as e:
                    # if e contains any letters or numbers, print it for debugging
                    if any(c.isalnum() for c in str(e)):
                        if verbose:
                            print(f"Exception in objective function: {e}")

                    return 1e10  # Large penalty for invalid parameters

            # Initial guess and bounds
            x0 = [2.0, 2.0, 0.0]
            bounds = [(1.0, 100.0), (0.1, 100.0), (0.0, 20.0)]  # shape, scale, loc

            # Optimize
            # result = minimize(objective, x0, method='Nelder-Mead',
            #                    options={'maxiter': 5, 'disp': verbose})
            # optimize using a method that supports bounds
            result = minimize(
                objective,
                x0,
                method="Nelder-Mead",
                bounds=bounds,
                options={"maxiter": 50, "disp": verbose},
            )

            # Store best results
            best_shape, best_scale, best_loc = result.x

            # Compute final correlation and p_value with best parameters
            shape_factors = pd.DataFrame(columns=[forcing_col], index=[1])
            shape_factors.loc[1, forcing_col] = best_shape
            scale_factors = pd.DataFrame(columns=[forcing_col], index=[1])
            scale_factors.loc[1, forcing_col] = best_scale
            loc_factors = pd.DataFrame(columns=[forcing_col], index=[1])
            loc_factors.loc[1, forcing_col] = best_loc

            transformed = transform_inputs(
                shape_factors,
                scale_factors,
                loc_factors,
                system_data.index,
                forcing_orig,
                cache=_topology_cache,
            )
            np.array(transformed[forcing_col + "_tr_1"].values)
            feature_names = [dep_col, forcing_col]
            model = ps.SINDy(
                differentiation_method=ps.FiniteDifference(
                    order=10, drop_endpoints=True
                ),
                feature_library=ps.PolynomialLibrary(
                    degree=2, include_bias=False, include_interaction=False
                ),
                optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
            )
            fit = model.fit(
                x=system_data.loc[:, dep_col],
                t=np.arange(0, len(system_data.index), 1),
                u=transformed,
                feature_names=feature_names,
            )
            # evaluate the r2 score
            r2 = fit.score(
                x=system_data.loc[:, dep_col],
                t=np.arange(0, len(system_data.index), 1),
                u=transformed,
            )
            try:
                model.print()
            except Exception as e:
                print(e)

            r2_values.loc[dep_col, forcing_col] = r2
            print(f"\nOptimizing transformation for {forcing_col} -> {dep_col}")
            print(
                f"  BEST: shape={best_shape:.2f}, scale={best_scale:.2f}, loc={best_loc:.2f}"
            )
            # save the best parameters
            best_params.loc[dep_col, forcing_col] = (best_shape, best_scale, best_loc)

            print("R2 Values:")
            print(r2_values)
            print("\n")

    print("Final SISO R2 Values:")
    print(r2_values)
    current_best_r2 = pd.Series(index=dependent_columns, dtype=float, data=0.0)

    # first identify the maximum r^2 value in each row. we know these will be included in the final topology
    for dep_col in dependent_columns:
        forcing_col = r2_values.loc[dep_col, :].idxmax()
        edges.loc[dep_col, forcing_col] = 1
        current_best_r2[dep_col] = r2_values.loc[dep_col, forcing_col]

    forcing_corr_w_existing = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=float
    )
    corr_wted_r2 = r2_values.copy(deep=True)
    # for each entry, weight it by (1 - its correlation with other inputs already selected for that output)
    for dep_col in dependent_columns:
        selected_inputs = list(edges.loc[dep_col, edges.loc[dep_col, :] == 1].index)
        for forcing_col in system_data.columns:
            if forcing_col in selected_inputs or forcing_col == dep_col:
                continue  # skip already selected inputs / autocorrelation
            # compute the average correlation of forcing_col with selected_inputs
            if len(selected_inputs) > 0:
                correlations = []
                for sel_input in selected_inputs:
                    # corr = np.corrcoef(system_data[forcing_col], system_data[sel_input])[0,1]
                    # compute the correlation between the transformed versions of forcing_col and sel_input
                    shape_factors_1 = pd.DataFrame(columns=[forcing_col], index=[1])
                    shape_factors_1.loc[1, forcing_col] = best_params.loc[
                        dep_col, forcing_col
                    ][0]
                    scale_factors_1 = pd.DataFrame(columns=[forcing_col], index=[1])
                    scale_factors_1.loc[1, forcing_col] = best_params.loc[
                        dep_col, forcing_col
                    ][1]
                    loc_factors_1 = pd.DataFrame(columns=[forcing_col], index=[1])
                    loc_factors_1.loc[1, forcing_col] = best_params.loc[
                        dep_col, forcing_col
                    ][2]
                    transformed_1 = transform_inputs(
                        shape_factors_1,
                        scale_factors_1,
                        loc_factors_1,
                        system_data.index,
                        system_data[[forcing_col]],
                        cache=_topology_cache,
                    )
                    shape_factors_2 = pd.DataFrame(columns=[sel_input], index=[1])
                    shape_factors_2.loc[1, sel_input] = best_params.loc[
                        dep_col, sel_input
                    ][0]
                    scale_factors_2 = pd.DataFrame(columns=[sel_input], index=[1])
                    scale_factors_2.loc[1, sel_input] = best_params.loc[
                        dep_col, sel_input
                    ][1]
                    loc_factors_2 = pd.DataFrame(columns=[sel_input], index=[1])
                    loc_factors_2.loc[1, sel_input] = best_params.loc[
                        dep_col, sel_input
                    ][2]
                    transformed_2 = transform_inputs(
                        shape_factors_2,
                        scale_factors_2,
                        loc_factors_2,
                        system_data.index,
                        system_data[[sel_input]],
                        cache=_topology_cache,
                    )
                    together = pd.DataFrame(index=system_data.index)
                    together[forcing_col] = transformed_1[str(forcing_col + "_tr_1")]
                    together[sel_input] = transformed_2[str(sel_input + "_tr_1")]
                    # Check for zero variance before computing correlation
                    if (
                        together[forcing_col].std() == 0
                        or together[sel_input].std() == 0
                    ):
                        corr = 2.0  # if this variable is constant, it's not contributing any information, so set to 2.0 so it gets excluded
                    else:
                        corr = np.corrcoef(together[forcing_col], together[sel_input])[
                            0, 1
                        ]
                        # Handle NaN from corrcoef (shouldn't happen after std check, but be safe)
                        forcing_corr_w_existing.loc[dep_col, forcing_col] = corr
                        if np.isnan(corr):
                            corr = 0.0
                    correlations.append(abs(corr))
                max_corr = np.max(correlations)
                np.sum(correlations)
            else:
                max_corr = 0.0
            corr_wted_r2.loc[dep_col, forcing_col] = r2_values.loc[
                dep_col, forcing_col
            ] * ((1 - max_corr) ** 10)
            # might want sum of correlation rather than max if multiple rounds are ever used.

    r2_values.stack().sort_values(ascending=False)
    sorted_corr_wted_r2 = corr_wted_r2.stack().sort_values(ascending=False)
    # iterate descending over sorted corr_wted_r2 values, adding edges if they improve the model r^2 significantly
    for idx in sorted_corr_wted_r2.index:
        # do we already have this edge?
        if edges.loc[idx[0], idx[1]] == 1:
            continue  # already have this edge

        output_variable = idx[0]
        forcing_variable = idx[1]
        r2 = r2_values.loc[output_variable, forcing_variable]

        non_rain_edges = edges.loc[
            ~edges.index.str.contains("rain"), ~edges.columns.str.contains("rain")
        ]

        # would adding this edge reduce the number of components in the graph? (not considering rain)
        non_rain_edges_if_added = non_rain_edges.copy(deep=True)
        non_rain_edges_if_added.loc[output_variable, forcing_variable] = 1

        n_components_now = nx.number_weakly_connected_components(
            nx.from_pandas_adjacency(non_rain_edges, create_using=nx.DiGraph)
        )
        if n_components_now == 1:
            print("graph is weakly connected.")
            # done
            break

        n_components = nx.number_weakly_connected_components(
            nx.from_pandas_adjacency(non_rain_edges_if_added, create_using=nx.DiGraph)
        )
        if "rain" not in forcing_variable.lower():  # always allow rain edges
            if n_components >= n_components_now:
                print(
                    f"Skipping addition of {forcing_variable} -> {output_variable} as it does not improve connectivity"
                )
                continue  # skip this addition as it doesn't improve connectivity

        print(
            f"Evaluating edge {forcing_variable} -> {output_variable} with r2 = {r2:.4f}"
        )
        print("current best r2 values:")
        print(current_best_r2)
        # build the candidate input set
        selected_inputs = list(
            edges.loc[output_variable, edges.loc[output_variable, :] == 1].index
        )
        candidate_inputs = selected_inputs + [forcing_variable]

        # optimize the transformations for all candidate inputs together, using siso best params as initial guesses
        def joint_objective(params):
            # params is a flat list of shape, scale, loc for each candidate input
            transformed_inputs = pd.DataFrame(index=system_data.index)
            for i, input_var in enumerate(candidate_inputs):
                shape = params[i * 3]
                scale = params[i * 3 + 1]
                loc = params[i * 3 + 2]
                shape_factors = pd.DataFrame(columns=[input_var], index=[1])
                shape_factors.loc[1, input_var] = shape
                scale_factors = pd.DataFrame(columns=[input_var], index=[1])
                scale_factors.loc[1, input_var] = scale
                loc_factors = pd.DataFrame(columns=[input_var], index=[1])
                loc_factors.loc[1, input_var] = loc
                forcing_orig = system_data[[input_var]].copy()
                transformed = transform_inputs(
                    shape_factors,
                    scale_factors,
                    loc_factors,
                    system_data.index,
                    forcing_orig,
                    cache=_topology_cache,
                )
                transformed_inputs = pd.concat(
                    (transformed_inputs, transformed[[input_var + "_tr_1"]]),
                    axis="columns",
                )
            # build and fit the sindy model
            feature_names = [output_variable] + candidate_inputs
            model = ps.SINDy(
                differentiation_method=ps.FiniteDifference(
                    order=10, drop_endpoints=True
                ),
                feature_library=ps.PolynomialLibrary(
                    degree=2, include_bias=False, include_interaction=False
                ),
                optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
            )
            fit = model.fit(
                x=system_data.loc[:, output_variable],
                t=np.arange(0, len(system_data.index), 1),
                u=transformed_inputs,
                feature_names=feature_names,
            )
            r2 = fit.score(
                x=system_data.loc[:, output_variable],
                t=np.arange(0, len(system_data.index), 1),
                u=transformed_inputs,
            )
            return -r2  # Negative because minimize

        # initial guesses
        x0 = []
        for input_var in candidate_inputs:
            shape, scale, loc = best_params.loc[output_variable, input_var]
            x0.extend([shape, scale, loc])
        bounds = []
        for input_var in candidate_inputs:
            bounds.extend(
                [(1.0, 100.0), (0.1, 100.0), (0.0, 20.0)]
            )  # shape, scale, loc

        # optimize
        result = minimize(
            joint_objective,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 50, "disp": verbose},
        )
        # extract best params
        optimized_params = result.x
        for i, input_var in enumerate(candidate_inputs):
            shape = optimized_params[i * 3]
            scale = optimized_params[i * 3 + 1]
            loc = optimized_params[i * 3 + 2]
            best_params.loc[output_variable, input_var] = (shape, scale, loc)
        # compute final r2 with optimized params
        transformed_inputs = pd.DataFrame(index=system_data.index)
        for i, input_var in enumerate(candidate_inputs):
            shape = optimized_params[i * 3]
            scale = optimized_params[i * 3 + 1]
            loc = optimized_params[i * 3 + 2]
            shape_factors = pd.DataFrame(columns=[input_var], index=[1])
            shape_factors.loc[1, input_var] = shape
            scale_factors = pd.DataFrame(columns=[input_var], index=[1])
            scale_factors.loc[1, input_var] = scale
            loc_factors = pd.DataFrame(columns=[input_var], index=[1])
            loc_factors.loc[1, input_var] = loc
            forcing_orig = system_data[[input_var]].copy()
            transformed = transform_inputs(
                shape_factors,
                scale_factors,
                loc_factors,
                system_data.index,
                forcing_orig,
                cache=_topology_cache,
            )
            transformed_inputs = pd.concat(
                (transformed_inputs, transformed[[input_var + "_tr_1"]]), axis="columns"
            )
        feature_names = [output_variable] + candidate_inputs
        model = ps.SINDy(
            differentiation_method=ps.FiniteDifference(order=10, drop_endpoints=True),
            feature_library=ps.PolynomialLibrary(
                degree=2, include_bias=False, include_interaction=False
            ),
            optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
        )
        fit = model.fit(
            x=system_data.loc[:, output_variable],
            t=np.arange(0, len(system_data.index), 1),
            u=transformed_inputs,
            feature_names=feature_names,
        )
        r2 = fit.score(
            x=system_data.loc[:, output_variable],
            t=np.arange(0, len(system_data.index), 1),
            u=transformed_inputs,
        )

        print(
            f"Testing inputs {candidate_inputs} for output {output_variable} -> r2 = {r2:.4f}"
        )
        if (
            r2 > current_best_r2[output_variable] + 0.01
        ):  # only keep it if it improves the r2 by at least 1%
            # add a conditional here for reducing the number of components in the graph. if it doesn't connect things that were previously unconnected, we don't want it.
            selected_inputs = candidate_inputs
            current_best_r2[output_variable] = r2
            print(
                f"  Accepted new input {forcing_variable}, updated r2 = {current_best_r2[output_variable]:.4f}"
            )
            edges.loc[output_variable, forcing_variable] = 1
        else:
            print(f"  Rejected new input {forcing_variable}, r2 would be {r2:.4f}")

    return {"edges": edges, "best_params": best_params}

    """
    # build the transformed timeseries for these candidate inputs using the best transformation parameters found earlier
    transformed_inputs = pd.DataFrame(index=system_data.index)
    for input_var in candidate_inputs:
        shape, scale, loc = best_params.loc[output_variable, input_var]
        shape_factors = pd.DataFrame(columns=[input_var], index=[1])
        shape_factors.loc[1, input_var] = shape
        scale_factors = pd.DataFrame(columns=[input_var], index=[1])
        scale_factors.loc[1, input_var] = scale
        loc_factors = pd.DataFrame(columns=[input_var], index=[1])
        loc_factors.loc[1, input_var] = loc
        forcing_orig = system_data[[input_var]].copy()
        transformed = transform_inputs(shape_factors, scale_factors, loc_factors,
                                        system_data.index, forcing_orig)
        transformed_inputs = pd.concat((transformed_inputs, transformed[[input_var + "_tr_1"]]), axis='columns')

    # build a sindy model with these inputs
    feature_names = [output_variable] + candidate_inputs
    model = ps.SINDy(
            differentiation_method= ps.FiniteDifference(order=10,drop_endpoints=True),
            feature_library=ps.PolynomialLibrary(degree=2,include_bias = False, include_interaction=False),
            optimizer=ps.optimizers.STLSQ(threshold=0,alpha=0)
            )
    if system_data.loc[:,candidate_inputs].empty: # the subsystem is autonomous
        fit = model.fit(x = system_data.loc[:,output_variable] ,t = np.arange(0,len(system_data.index),1),
            feature_names = feature_names)
    else: # there is some forcing
        #fit = model.fit(x = system_data.loc[:,output_variable] ,t = np.arange(0,len(system_data.index),1) , u = system_data.loc[:,candidate_inputs])
        fit = model.fit(x = system_data.loc[:,output_variable] ,t = np.arange(0,len(system_data.index),1) , u = transformed_inputs,
            feature_names = feature_names)
    # evaluate the r2 score
    r2 = fit.score(x = system_data.loc[:,output_variable] ,t = np.arange(0,len(system_data.index),1), u = transformed_inputs)
    model.print(precision=5)
    """

    """
    for dep_col in dependent_columns:
        response = np.array(system_data[dep_col].values)

        for forcing_col in system_data.columns:
            #if forcing_col == dep_col:
            #    continue  # autocorrelation is always included.
            # we already know autocorrelation will be included in the model, but we still want to quantify the r^2 value of that autocorrelation.
            print(f"\nOptimizing transformation for {forcing_col} -> {dep_col}")
            forcing_orig = system_data[[forcing_col]].copy()

            # Objective function to minimize (negative because we want to maximize correlation - p_value)
            def objective(params):
                shape, scale, loc = params

                # Create transformation parameter DataFrames
                shape_factors = pd.DataFrame(columns=[forcing_col], index=[1])
                shape_factors.loc[1, forcing_col] = shape
                scale_factors = pd.DataFrame(columns=[forcing_col], index=[1])
                scale_factors.loc[1, forcing_col] = scale
                loc_factors = pd.DataFrame(columns=[forcing_col], index=[1])
                loc_factors.loc[1, forcing_col] = loc

                # Transform the forcing
                try:
                    transformed = transform_inputs(shape_factors, scale_factors, loc_factors,
                                                    system_data.index, forcing_orig)
                    forcing = np.array(transformed[forcing_col + "_tr_1"].values)

                    # Compute CCM
                    cross_map = ccm(forcing, response)
                    correlation, p_value = cross_map.causality()
                    # linear regression between transformed forcing and derivative of response
                    #slope, intercept, r_value, p_value_lr, std_err = stats.linregress(forcing, np.gradient(response))

                    # fourth order polynomial regression between transformed forcing and derivative of response
                    coeffs = np.polyfit(forcing, np.gradient(response), 4)
                    r_value_poly = np.corrcoef(np.polyval(coeffs, forcing), np.gradient(response))[0, 1]
                    result = r_value_poly
                    # not actually using the CCM library right now
                    # r^2 likely makes more sense as our criterion.
                    r2_wrong_way = r_value_poly**2
                    r2 = sklearn.metrics.r2_score(np.gradient(response), np.polyval(coeffs, forcing))
                    r2_diff = r2 - r2_wrong_way
                    #result = correlation - p_value + np.abs(r_value) - np.abs(intercept) - p_value_lr
                    #print(f"  shape={shape:.2f}, scale={scale:.2f}, loc={loc:.2f} -> corr={correlation:.4f}, p={p_value:.4f}, slope = {slope:.4f}, r_value_lr = {r_value:.4f}, obj={result:.4f}")
                    print(f"  shape={shape:.2f}, scale={scale:.2f}, loc={loc:.2f} -> corr={correlation:.4f}, p={p_value:.4f}, r_value_poly={result:.4f}")
                    return -result  # Negative because minimize
                except Exception as e:
                    print(f"  Error with params: {e}")
                    return 1e10  # Large penalty for invalid parameters

            # Initial guess and bounds
            x0 = [2.0, 2.0, 0.0]
            bounds = [(1.0, 100.0), (0.1, 100.0), (0.0, 20.0)]  # shape, scale, loc

            # Optimize
            result = minimize(objective, x0, method='Nelder-Mead',
                                options={'maxiter': 25, 'disp': verbose})

            # Store best results
            best_shape, best_scale, best_loc = result.x

            # Compute final correlation and p_value with best parameters
            shape_factors = pd.DataFrame(columns=[forcing_col], index=[1])
            shape_factors.loc[1, forcing_col] = best_shape
            scale_factors = pd.DataFrame(columns=[forcing_col], index=[1])
            scale_factors.loc[1, forcing_col] = best_scale
            loc_factors = pd.DataFrame(columns=[forcing_col], index=[1])
            loc_factors.loc[1, forcing_col] = best_loc

            transformed = transform_inputs(shape_factors, scale_factors, loc_factors,
                                            system_data.index, forcing_orig)
            forcing = np.array(transformed[forcing_col + "_tr_1"].values)
            cross_map = ccm(forcing, response)
            correlation, p_value = cross_map.causality()
            slope, intercept, r_value, p_value_lr, std_err = stats.linregress(forcing, np.gradient(response))

            coeffs = np.polyfit(forcing, np.gradient(response), 4)
            r_value_poly = np.corrcoef(np.polyval(coeffs, forcing), np.gradient(response))[0, 1]
            result = r_value_poly
            r2 = sklearn.metrics.r2_score(np.gradient(response), np.polyval(coeffs, forcing))
            r2_wrong_way = r_value_poly**2
            r2_diff = r2 - r2_wrong_way

            best_correlations.loc[dep_col, forcing_col] = correlation
            best_p_values.loc[dep_col, forcing_col] = p_value
            #best_scores.loc[dep_col, forcing_col] = correlation - p_value + np.abs(r_value) - np.abs(intercept) - p_value_lr
            best_scores.loc[dep_col, forcing_col] = result
            best_params.loc[dep_col, forcing_col] = (best_shape, best_scale, best_loc)
            r2_values.loc[dep_col, forcing_col] = r2
            print(f"\nOptimizing transformation for {forcing_col} -> {dep_col}")
            print(f"  BEST: shape={best_shape:.2f}, scale={best_scale:.2f}, loc={best_loc:.2f}")
            print(f"  BEST: corr={correlation:.4f}, p={p_value:.4f}, r_value_poly={result:.4f}")

            #print("Best Scores (r_value of polynomial):")
            #print(best_scores)
            print("R2 Values:")
            print(r2_values)

            #print(f"  Final: corr={correlation:.4f}, p={p_value:.4f}")
            #print(f"  Final: corr={correlation:.4f}, p={p_value:.4f}, slope = {slope:.4f}, r_value_lr = {r_value:.4f}")
            #print(f"  Score: {correlation - p_value + np.abs(r_value) - np.abs(intercept) - p_value_lr:.4f}")

            # display time series of response and transformed forcing
            plt.figure()
            plt.plot(system_data.index, response, label='Response')
            plt.plot(system_data.index, forcing, label='Transformed Forcing')
            plt.xlabel('Time')
            plt.legend()

            # display phase portrait of response derivative vs transformed forcing
            plt.figure()
            plt.scatter(forcing,np.gradient(response),alpha=0.3)
            # plot the linear regression line
            #plt.plot(forcing, intercept + slope*forcing, color='red', label='Linear Fit')
            # plot the polynomial regression line
            x_vals = np.linspace(min(forcing), max(forcing), 100)
            plt.plot(x_vals, np.polyval(coeffs, x_vals), color='red', label='Polynomial Fit')
            plt.ylabel('Response Derivative')
            plt.xlabel('Transformed Forcing')
            plt.title(f'Phase Portrait for {dep_col} vs {forcing_col}')




    #print("Best Scores:")
    #print(best_scores)
    #plt.show()
    print("r2 Values:")
    print(r2_values)
    """
    # once we start actually inferring the topology using these scores I'm thinking the inclusion decision should consider:
    # 1 - what is the row-wise (output variable) sum of r^2 for the links added so far? Once we reach E(r^2) = 85%, it's probably not necessary to add more edges toward this variable
    # the r^2 threhsold is noise dependent, so it would be good to scale it automatically based on the data.
    # -> this idea is not applicable when there's a high degree of correlation between different transformed input variables (eg basin depths and rainfall)
    # 2 - what is our current connectivity? It makes sense for our application to identify the minimum number of edges for a dendritic network
    # our output is then "identified main flow paths" rather than "every feasible connection" -> currently implemented.
    # also output those "likely, but not included" connections for review. you'd plot those as dotted lines rather than solid. -> not yet implemented nov 10.
    # 3 - are these variables already indirectly causally connected? ie, is this a skip connection for a path already there?
    # for example, if 1 -> 4 looks strong, but they're already linked through O4, that's a lower priority.
    # a caveat here would be raingage data, but that will be clearly labeled as separate from the flow / depth data
    # also, perhaps raindata isn't even an issue. it might make sense to not directly consider rainfall forcing for an interceptor
    # and headwaters will obviously have to consider rainfall forcing directly as they won't achieve r^2 thresholds without it.
    # solution -> so long as we exclude rainfall, this consideration is handled by only accepting edges that reduce the number of components in the graph.
    # 4 - distinguish correlation from causation. Is a high degree of the variation in both of these variables explained by a third variable?
    # if, C dominates the dynamics of A and B, A and B are strongly correlated.
    # in the icud example, rainfall explains a lot of variation in all 4 locations, so they're all strongly correlated.
    # edge selection logic will consider this. If edges are weighted by their r^2 score, there's likely a graph theoretic algorithm with applicability
    # perhaps minimzing the sum of r^2 weights while building a spanning tree would give you the desired behavior in regards to this. "don't overexplain"
    # this might complicate the early stopping. there could be a parameter like "initial_pair_eval" where we only look at the n (5?) closest for our first pass
    # and then we only examine parts of the network that aren't yet connected afterwards.
    # 5 - max junction degree. in a dendritic network, junctions don't have more than 3 inflows usually.
    # so we can cap the number of incoming edges to each output variable. or at least favor a more parsimonious topology.
    # solution -> this is basically handled by only accepting edges that reduce the number of components in the graph.
    # 6 - consider implications on graph structure when choosing links. which combination of links produces the fewest components (most connected) graph?
    # which combination minimizes skip connections? does one combination imply a loop?
    # 7 - the expensive part of modpods is optimizing the input transformations.
    # so we could incorporate the (hybrid)-sindy r^2 achieved by different combinations of transformed input variables into the decision without a ton of additional comp expense
    # done that way we're learning the model at the same time we're inferring the topology. which does make a lot of sense.
    # nov10 - the more I've been working on this I'm thinking that you need a mechanistic model to infer the causation.
    # that is, it seems like you can't say where the causation is if you're not also identifying how it's happening.

    # early stopping / comp expense idea:
    # begin with the pairs with the least geographic distance between them
    # once any output variables r^2 sum exceeds a noise-based threshold, we can skip any remaining forcing variables
    # we could end up evaluating a very small fraction of the possible connections this way

    # stray thought: for a pump station your forcing would either be the change in the wet well level (hybrid system)
    # or the pump run-times (still continuous, just step inputs)

    # return {'best_params': best_params, 'correlations': best_correlations, 'p_values': best_p_values, 'scores': best_scores}


# === Topology inference helper functions ===
# These implement the SINDy-based topology inference


def _compute_distances(sensor_locations, columns):
    """Compute Euclidean distances between all pairs of sensors.

    Args:
        sensor_locations: dict mapping column names to {"lat": float, "lon": float}
        columns: list of column names to compute distances for

    Returns:
        pd.DataFrame with distances, index and columns are sensor names
    """
    distances = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for c1 in columns:
        for c2 in columns:
            if c1 == c2:
                distances.loc[c1, c2] = 0.0
            elif c1 in sensor_locations and c2 in sensor_locations:
                lat1, lon1 = sensor_locations[c1]["lat"], sensor_locations[c1]["lon"]
                lat2, lon2 = sensor_locations[c2]["lat"], sensor_locations[c2]["lon"]
                # Euclidean distance
                distances.loc[c1, c2] = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
            else:
                # If location not available, set to infinity (will be evaluated last)
                distances.loc[c1, c2] = np.inf
    return distances


def _get_n_nearest_neighbors(dep_col, distances, n, exclude_cols=None):
    """Get the n nearest neighbors for a dependent column.

    Args:
        dep_col: the dependent column to find neighbors for
        distances: DataFrame of distances between sensors
        n: number of neighbors to return
        exclude_cols: columns to exclude from consideration (e.g., independent columns)

    Returns:
        list of column names of the n nearest neighbors
    """
    if exclude_cols is None:
        exclude_cols = []

    # Get distances from dep_col to all other columns
    dists = distances.loc[dep_col].copy()

    # Exclude self and any specified columns
    dists = dists.drop(
        labels=[dep_col] + [c for c in exclude_cols if c in dists.index],
        errors="ignore",
    )

    # Sort by distance and return the n nearest
    sorted_dists = dists.sort_values()
    return list(sorted_dists.head(n).index)


def find_topology_no_geo(
    system_data,
    dependent_columns,
    independent_columns,
    max_iterations=250,
    graph_type="Weak-Conn",
    verbose=False,
    sensor_locations=None,
    init_neighbors=3,
):
    """
    Infer network topology from time series data using SINDy-based optimization.

    Args:
        system_data: pd.DataFrame with time series data, columns are variables
        dependent_columns: list of column names that are dependent variables
        independent_columns: list of column names that are independent/forcing variables
        max_iterations: maximum iterations for optimization
        graph_type: type of graph connectivity requirement ('Weak-Conn')
        verbose: whether to print detailed output
        sensor_locations: optional dict mapping column names to {"lat": float, "lon": float}.
            If provided, uses geographic filtering to reduce computation by only evaluating
            nearby sensors as potential forcings. Format: {"station_A": {"lat": 41.5, "lon": -74.5}, ...}
        init_neighbors: initial number of nearest neighbors to evaluate when sensor_locations
            is provided (default: 3). Ignored if sensor_locations is None.

    Returns:
        dict with keys: "edges", "best_params", "r2_values", "lead_lag"
    """

    # If sensor_locations provided, use the geo-filtering implementation
    if sensor_locations is not None:
        return _find_topology_with_geo(
            system_data=system_data,
            dependent_columns=dependent_columns,
            independent_columns=independent_columns,
            sensor_locations=sensor_locations,
            max_iterations=max_iterations,
            graph_type=graph_type,
            verbose=verbose,
            init_neighbors=init_neighbors,
        )

    # only print 3 places past the decimal for floats. don't use scientific notation. if less than 0.001, print as <0.001
    pd.options.display.float_format = "{:.3f}".format

    # Helper function to find the lag with strongest cross-correlation
    def cross_correlation_lag(x, y, max_lag):
        """Find the lag with strongest cross-correlation between x and y.

        Returns:
            best_lag: Positive lag means x leads y (x happens before y)
                      Negative lag means y leads x (y happens before x)
            best_corr: The correlation coefficient at best_lag
        """
        best_lag, best_corr = 0, -2
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                xs = x.iloc[-lag:]
                ys = y.iloc[: len(xs)]
            elif lag > 0:
                ys = y.iloc[lag:]
                xs = x.iloc[: len(ys)]
            else:
                xs, ys = x, y
            if len(xs) < 5 or xs.std() == 0 or ys.std() == 0:
                continue
            c = np.corrcoef(xs, ys)[0, 1]
            if np.isnan(c):
                continue
            if c > best_corr:
                best_corr, best_lag = c, lag
        return best_lag, best_corr

    # drop columns from system_data which aren't in dependent_columns or independent_columns
    # this ensures we only analyze the variables of interest
    system_data = pd.concat(
        (system_data[independent_columns], system_data[dependent_columns]),
        axis="columns",
    )

    # Store results for each column pair
    best_params = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=object
    )
    best_correlations = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=float
    )
    best_p_values = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=float
    )
    best_scores = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=float
    )
    r2_values = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=float
    )
    lead_lag = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=float
    )
    edges = pd.DataFrame(
        index=system_data.columns, columns=system_data.columns, dtype=int, data=0
    )  # from column, to row. causation, not flow.

    for dep_col in dependent_columns:
        response = np.array(system_data[dep_col].values)

        # First, compute autocorrelation-only R² (no external forcing)
        # This tells us how much of the dynamics can be explained by the state alone
        # print(f"\nComputing autocorrelation R² for {dep_col}")
        model = ps.SINDy(
            differentiation_method=ps.FiniteDifference(order=10, drop_endpoints=True),
            feature_library=ps.PolynomialLibrary(
                degree=2, include_bias=False, include_interaction=False
            ),
            optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
        )
        # Fit with no control input (u=None), just the state
        fit = model.fit(
            x=system_data.loc[:, dep_col],
            t=np.arange(0, len(system_data.index), 1),
            feature_names=[dep_col],
        )
        auto_r2 = fit.score(
            x=system_data.loc[:, dep_col], t=np.arange(0, len(system_data.index), 1)
        )
        r2_values.loc[dep_col, dep_col] = auto_r2
        # print(f"  Autocorrelation R² for {dep_col}: {auto_r2:.4f}")
        # try:
        #    model.print()
        # except Exception as e:
        #    print(e)

        for forcing_col in system_data.columns:
            if forcing_col == dep_col:
                continue  # already computed autocorrelation above

            # EXPERIMENTAL: Check lead/lag before expensive SISO optimization
            # Skip if forcing doesn't lead response (comment out to disable this check)
            max_lag_check = min(len(system_data) // 4, 100)
            early_lag, early_xcorr = cross_correlation_lag(
                system_data[forcing_col], system_data[dep_col], max_lag_check
            )
            if early_lag < -5:
                print(
                    f"\nSkipping {forcing_col} -> {dep_col}: forcing lags response (lag={early_lag})"
                )
                lead_lag.loc[dep_col, forcing_col] = early_lag
                r2_values.loc[dep_col, forcing_col] = 0.0
                best_params.loc[dep_col, forcing_col] = (
                    2.0,
                    2.0,
                    0.0,
                )  # default params
                continue
            # END EXPERIMENTAL

            print(f"\nOptimizing transformation for {forcing_col} -> {dep_col}")
            forcing_orig = system_data[[forcing_col]].copy(deep=True)

            # Objective function to minimize (negative because we want to maximize correlation - p_value)
            def objective(params):
                shape, scale, loc = params

                # Create transformation parameter DataFrames
                shape_factors = pd.DataFrame(columns=[forcing_col], index=[1])
                shape_factors.loc[1, forcing_col] = shape
                scale_factors = pd.DataFrame(columns=[forcing_col], index=[1])
                scale_factors.loc[1, forcing_col] = scale
                loc_factors = pd.DataFrame(columns=[forcing_col], index=[1])
                loc_factors.loc[1, forcing_col] = loc

                try:
                    # build the candidate input set
                    # selected_inputs = list(edges.loc[output_variable, edges.loc[output_variable,:] == 1].index)
                    # candidate_inputs = selected_inputs + [forcing_variable]
                    # build the transformed timeseries for these candidate inputs using the best transformation parameters found earlier
                    transformed_inputs = pd.DataFrame(index=system_data.index)
                    """
                    for input_var in candidate_inputs:
                        shape, scale, loc = best_params.loc[output_variable, input_var]
                        shape_factors = pd.DataFrame(columns=[input_var], index=[1])
                        shape_factors.loc[1, input_var] = shape
                        scale_factors = pd.DataFrame(columns=[input_var], index=[1])
                        scale_factors.loc[1, input_var] = scale
                        loc_factors = pd.DataFrame(columns=[input_var], index=[1])
                        loc_factors.loc[1, input_var] = loc
                        forcing_orig = system_data[[input_var]].copy()
                        transformed = transform_inputs(shape_factors, scale_factors, loc_factors, 
                                                        system_data.index, forcing_orig)
                        transformed_inputs = pd.concat((transformed_inputs, transformed[[input_var + "_tr_1"]]), axis='columns')
                    """
                    # SINDY way
                    transformed = transform_inputs(
                        shape_factors,
                        scale_factors,
                        loc_factors,
                        system_data.index,
                        forcing_orig,
                    )
                    transformed_inputs = pd.concat(
                        (transformed_inputs, transformed[[forcing_col + "_tr_1"]]),
                        axis="columns",
                    )
                    # build a sindy model with these inputs
                    # feature_names = [output_variable] + candidate_inputs
                    feature_names = [dep_col, str(forcing_col + "_tr_1")]
                    model = ps.SINDy(
                        differentiation_method=ps.FiniteDifference(
                            order=10, drop_endpoints=True
                        ),
                        feature_library=ps.PolynomialLibrary(
                            degree=2, include_bias=False, include_interaction=False
                        ),
                        optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
                    )
                    # fit = model.fit(x = system_data.loc[:,output_variable] ,t = np.arange(0,len(system_data.index),1) , u = transformed_inputs,
                    #        feature_names = feature_names)
                    fit = model.fit(
                        x=system_data.loc[:, dep_col],
                        u=transformed_inputs,
                        t=np.arange(0, len(system_data.index), 1),
                        feature_names=feature_names,
                    )
                    r2 = fit.score(
                        x=system_data.loc[:, dep_col],
                        u=transformed_inputs,
                        t=np.arange(0, len(system_data.index), 1),
                    )
                    # model.print(precision=5)

                    """
                    # polynomial regression way (might be faster than sindy, doesn't consider autocorrelation)
                    forcing = np.array(transformed[forcing_col + "_tr_1"].values)
                    

                    # fourth order polynomial regression between transformed forcing and derivative of response
                    coeffs = np.polyfit(forcing, np.gradient(response), 4)
                    r_value_poly = np.corrcoef(np.polyval(coeffs, forcing), np.gradient(response))[0, 1]

                    # r^2 likely makes more sense as our criterion.
                    r2 = sklearn.metrics.r2_score(np.gradient(response), np.polyval(coeffs, forcing))
                    """

                    return -r2  # Negative because minimize
                except Exception as e:
                    # if e contains any letters or numbers, print it for debugging
                    if any(c.isalnum() for c in str(e)):
                        if verbose:
                            print(f"Exception in objective function: {e}")

                    return 1e10  # Large penalty for invalid parameters

            # Initial guess and bounds
            x0 = [2.0, 2.0, 0.0]
            bounds = [(1.0, 300.0), (1e-5, 300.0), (0, 300.0)]  # shape, scale, loc

            # Optimize
            # result = minimize(objective, x0, method='Nelder-Mead',
            #                    options={'maxiter': 5, 'disp': verbose})
            # optimize using a method that supports bounds
            result = minimize(
                objective,
                x0,
                method="Nelder-Mead",
                bounds=bounds,
                options={"maxiter": max_iterations, "disp": verbose, "fatol": 1e-4},
            )
            # can use 'fatol' keyword argument to set convergence tolerance if speedup is desired.
            # I'm worried about losing accuracy with that though.

            # result = minimize(objective, x0, method='trust-constr', bounds=bounds,
            #                    options={'maxiter': max_iterations, 'disp': verbose})
            # L-BFGS-B did not get nearly as good of results as Nelder-Mead in testing. maybe there are local minima in the objective.
            # trust-constr was also worse than nelder-mead.
            # Store best results
            best_shape, best_scale, best_loc = result.x

            # Compute final correlation and p_value with best parameters
            shape_factors = pd.DataFrame(columns=[forcing_col], index=[1])
            shape_factors.loc[1, forcing_col] = best_shape
            scale_factors = pd.DataFrame(columns=[forcing_col], index=[1])
            scale_factors.loc[1, forcing_col] = best_scale
            loc_factors = pd.DataFrame(columns=[forcing_col], index=[1])
            loc_factors.loc[1, forcing_col] = best_loc

            transformed = transform_inputs(
                shape_factors,
                scale_factors,
                loc_factors,
                system_data.index,
                forcing_orig,
            )
            forcing = np.array(transformed[forcing_col + "_tr_1"].values)
            feature_names = [dep_col, forcing_col]
            model = ps.SINDy(
                differentiation_method=ps.FiniteDifference(
                    order=10, drop_endpoints=True
                ),
                feature_library=ps.PolynomialLibrary(
                    degree=2, include_bias=False, include_interaction=False
                ),
                optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
            )
            fit = model.fit(
                x=system_data.loc[:, dep_col],
                t=np.arange(0, len(system_data.index), 1),
                u=transformed,
                feature_names=feature_names,
            )
            # evaluate the r2 score
            r2 = fit.score(
                x=system_data.loc[:, dep_col],
                t=np.arange(0, len(system_data.index), 1),
                u=transformed,
            )
            try:
                model.print()
            except Exception as e:
                print(e)

            r2_values.loc[dep_col, forcing_col] = r2

            # Compute cross-correlation lag between forcing and response
            # Use max_lag of 1/4 of the data length, capped at 100
            max_lag = min(len(system_data) // 4, 100)
            best_lag, best_xcorr = cross_correlation_lag(
                system_data[forcing_col], system_data[dep_col], max_lag
            )
            lead_lag.loc[dep_col, forcing_col] = best_lag

            print(f"\nOptimizing transformation for {forcing_col} -> {dep_col}")
            print(
                f"  BEST: shape={best_shape:.2f}, scale={best_scale:.2f}, loc={best_loc:.2f}"
            )
            print(f"  Cross-correlation: lag={best_lag}, corr={best_xcorr:.4f}")
            # save the best parameters
            best_params.loc[dep_col, forcing_col] = (best_shape, best_scale, best_loc)

            print("R2 Values:")
            print(r2_values)
            print("\n")

    print("Final SISO R2 Values:")
    print(r2_values)
    current_best_r2 = pd.Series(index=dependent_columns, dtype=float, data=0.0)
    print("Lead/Lag Matrix: (positive lag means forcing leads response)")
    print(lead_lag)

    # OPTION A: Mask r2 values by nonnegative lead/lag (forcing must lead response)
    # This is applied AFTER SISO optimization - use this if not skipping early
    # r2_values = r2_values.mask(lead_lag < 0, 0)
    # print("Masked R2 Values (only forcing leads response):")
    # print(r2_values)

    # OPTION B: Early skip is done above in the SISO loop - r2_values already has 0s for skipped pairs

    # first identify the maximum r^2 value in each row. we know these will be included in the final topology
    # with an exception: if we form a cycle with these initial edges, remove the lowest r^2 edge in the cycle
    # for dep_col in dependent_columns:
    #    forcing_col = r2_values.loc[dep_col,:].idxmax()
    #    edges.loc[dep_col,forcing_col] = 1
    #    current_best_r2[dep_col] = r2_values.loc[dep_col,forcing_col]

    # try a different method of picking initial edges
    # find the n_columns edges in r2_values with the highest r^2 values
    # if they are the maximum in their row and column, include them
    sorted_r2 = r2_values.stack().sort_values(ascending=False)
    for idx in sorted_r2.index:
        dep_col = idx[0]
        forcing_col = idx[1]
        r2 = r2_values.loc[dep_col, forcing_col]
        # is this the maximum in its row and column? (strongest connection for giver and receiver)
        if (
            r2 == r2_values.loc[dep_col, :].max()
            and r2 == r2_values.loc[:, forcing_col].max()
        ):
            edges.loc[dep_col, forcing_col] = 1
            current_best_r2[dep_col] = r2_values.loc[dep_col, forcing_col]
            print(f"Initial edge added: {forcing_col} -> {dep_col} with r^2 = {r2:.4f}")

    # check for cycles and remove them iteratively
    G = nx.from_pandas_adjacency(edges, create_using=nx.DiGraph)
    while True:
        try:
            # find_cycle returns a list of edges forming ONE cycle: [(u, v, dir), (v, w, dir), ...]
            cycle_edges = list(nx.find_cycle(G, orientation="original"))
            if len(cycle_edges) == 0:
                break

            print(
                f"Found cycle with {len(cycle_edges)} edges. Removing lowest r^2 edge."
            )
            print(f"Cycle edges: {[(e[0], e[1]) for e in cycle_edges]}")

            # find the edge with the lowest r^2 in the cycle
            min_r2 = float("inf")
            edge_to_remove = None
            for edge in cycle_edges:
                from_node = edge[0]  # source node
                to_node = edge[1]  # target node
                # In our adjacency matrix, edges.loc[row, col] = 1 means col -> row
                # So we need r2_values.loc[to_node, from_node] for edge from_node -> to_node
                r2 = r2_values.loc[to_node, from_node]
                print(f"  Edge {from_node} -> {to_node}: r^2 = {r2:.4f}")
                if r2 < min_r2:
                    min_r2 = r2
                    edge_to_remove = (from_node, to_node)

            # remove this edge from our edges DataFrame
            # edges.loc[row, col] = 1 means col -> row, so to remove from_node -> to_node:
            edges.loc[edge_to_remove[1], edge_to_remove[0]] = 0
            print(
                f"  Removed edge {edge_to_remove[0]} -> {edge_to_remove[1]} with r^2 = {min_r2:.4f}"
            )

            # rebuild the graph for next iteration
            G = nx.from_pandas_adjacency(edges, create_using=nx.DiGraph)

        except nx.NetworkXNoCycle:
            # No cycle found, we're done
            print("No cycles detected in initial edges.")
            break
        except Exception as e:
            print(f"Error during cycle detection: {e}")
            break

    # Helper function to update correlation-weighted R² scores for a single output variable
    def update_corr_weighted_r2(dep_col):
        """Update corr_wted_r2 for all potential inputs to dep_col based on current edges."""
        selected_inputs = list(edges.loc[dep_col, edges.loc[dep_col, :] == 1].index)
        for forcing_col in system_data.columns:
            if forcing_col in selected_inputs or forcing_col == dep_col:
                continue  # skip already selected inputs / autocorrelation

            if len(selected_inputs) > 0:
                correlations = []
                for sel_input in selected_inputs:
                    # compute correlation between transformed versions of forcing_col and sel_input
                    shape_factors_1 = pd.DataFrame(columns=[forcing_col], index=[1])
                    shape_factors_1.loc[1, forcing_col] = best_params.loc[
                        dep_col, forcing_col
                    ][0]
                    scale_factors_1 = pd.DataFrame(columns=[forcing_col], index=[1])
                    scale_factors_1.loc[1, forcing_col] = best_params.loc[
                        dep_col, forcing_col
                    ][1]
                    loc_factors_1 = pd.DataFrame(columns=[forcing_col], index=[1])
                    loc_factors_1.loc[1, forcing_col] = best_params.loc[
                        dep_col, forcing_col
                    ][2]
                    transformed_1 = transform_inputs(
                        shape_factors_1,
                        scale_factors_1,
                        loc_factors_1,
                        system_data.index,
                        system_data[[forcing_col]],
                    )

                    shape_factors_2 = pd.DataFrame(columns=[sel_input], index=[1])
                    shape_factors_2.loc[1, sel_input] = best_params.loc[
                        dep_col, sel_input
                    ][0]
                    scale_factors_2 = pd.DataFrame(columns=[sel_input], index=[1])
                    scale_factors_2.loc[1, sel_input] = best_params.loc[
                        dep_col, sel_input
                    ][1]
                    loc_factors_2 = pd.DataFrame(columns=[sel_input], index=[1])
                    loc_factors_2.loc[1, sel_input] = best_params.loc[
                        dep_col, sel_input
                    ][2]
                    transformed_2 = transform_inputs(
                        shape_factors_2,
                        scale_factors_2,
                        loc_factors_2,
                        system_data.index,
                        system_data[[sel_input]],
                    )

                    together = pd.DataFrame(index=system_data.index)
                    together[forcing_col] = transformed_1[str(forcing_col + "_tr_1")]
                    together[sel_input] = transformed_2[str(sel_input + "_tr_1")]

                    # Check for zero variance before computing correlation
                    if (
                        together[forcing_col].std() == 0
                        or together[sel_input].std() == 0
                    ):
                        corr = 2.0  # constant variable, exclude it
                    else:
                        corr = np.corrcoef(together[forcing_col], together[sel_input])[
                            0, 1
                        ]
                        if np.isnan(corr):
                            corr = 0.0
                    correlations.append(abs(corr))
                max_corr = np.max(correlations)
            else:
                max_corr = 0.0

            corr_wted_r2.loc[dep_col, forcing_col] = (
                r2_values.loc[dep_col, forcing_col] * 1
            )  # ((1 - max_corr)) # was **10

    # Initialize correlation-weighted R² scores
    corr_wted_r2 = r2_values.copy(deep=True)
    for dep_col in dependent_columns:
        update_corr_weighted_r2(dep_col)

    sorted_r2 = r2_values.stack().sort_values(ascending=False)
    if verbose:
        print("Sorted R2 values:")
        print(sorted_r2)

    # Use a while loop so we can re-sort after each edge addition
    # This ensures we always pick the best remaining candidate after correlation weights are updated
    evaluated_pairs = (
        set()
    )  # Track pairs we've already evaluated to avoid infinite loops

    while True:
        sorted_corr_wted_r2 = corr_wted_r2.stack().sort_values(ascending=False)
        # Find the best candidate we haven't evaluated yet
        idx = None
        for candidate_idx in sorted_corr_wted_r2.index:
            if (
                candidate_idx not in evaluated_pairs
                and edges.loc[candidate_idx[0], candidate_idx[1]] != 1
            ):
                idx = candidate_idx
                break

        if idx is None:
            print("No more candidate edges to evaluate.")
            break

        evaluated_pairs.add(idx)
        output_variable = idx[0]
        forcing_variable = idx[1]
        r2 = r2_values.loc[output_variable, forcing_variable]

        non_rain_edges = edges.loc[
            ~edges.index.str.contains("rain"), ~edges.columns.str.contains("rain")
        ]

        # would adding this edge reduce the number of components in the graph? (not considering rain)
        non_rain_edges_if_added = non_rain_edges.copy(deep=True)
        non_rain_edges_if_added.loc[output_variable, forcing_variable] = 1

        n_components_now = nx.number_weakly_connected_components(
            nx.from_pandas_adjacency(non_rain_edges, create_using=nx.DiGraph)
        )
        if n_components_now == 1:
            print("graph is weakly connected.")
            # done
            break

        n_components = nx.number_weakly_connected_components(
            nx.from_pandas_adjacency(non_rain_edges_if_added, create_using=nx.DiGraph)
        )
        if "rain" not in forcing_variable.lower():  # always allow rain edges
            if n_components >= n_components_now:
                print(
                    f"Skipping addition of {forcing_variable} -> {output_variable} as it does not improve connectivity"
                )
                continue  # skip this addition as it doesn't improve connectivity

        print(
            f"Evaluating edge {forcing_variable} -> {output_variable} with r2 = {r2:.4f}"
        )
        print("current best r2 values:")
        print(current_best_r2)
        # build the candidate input set
        selected_inputs = list(
            edges.loc[output_variable, edges.loc[output_variable, :] == 1].index
        )
        candidate_inputs = selected_inputs + [forcing_variable]

        # optimize the transformations for all candidate inputs together, using siso best params as initial guesses
        def joint_objective(params, debug=False):
            # params is a flat list of shape, scale, loc for each candidate input
            transformed_inputs = pd.DataFrame(index=system_data.index)
            for i, input_var in enumerate(candidate_inputs):
                shape = params[i * 3]
                scale = params[i * 3 + 1]
                loc = params[i * 3 + 2]
                shape_factors = pd.DataFrame(columns=[input_var], index=[1])
                shape_factors.loc[1, input_var] = shape
                scale_factors = pd.DataFrame(columns=[input_var], index=[1])
                scale_factors.loc[1, input_var] = scale
                loc_factors = pd.DataFrame(columns=[input_var], index=[1])
                loc_factors.loc[1, input_var] = loc
                forcing_orig = system_data[[input_var]].copy()
                transformed = transform_inputs(
                    shape_factors,
                    scale_factors,
                    loc_factors,
                    system_data.index,
                    forcing_orig,
                )
                # Include BOTH original and transformed columns, consistent with SISO phase
                transformed_inputs = pd.concat(
                    (transformed_inputs, transformed), axis="columns"
                )
            # build and fit the sindy model
            feature_names = [output_variable] + list(transformed_inputs.columns)
            model = ps.SINDy(
                differentiation_method=ps.FiniteDifference(
                    order=10, drop_endpoints=True
                ),
                feature_library=ps.PolynomialLibrary(
                    degree=2, include_bias=False, include_interaction=False
                ),
                optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
            )
            fit = model.fit(
                x=system_data.loc[:, output_variable],
                t=np.arange(0, len(system_data.index), 1),
                u=transformed_inputs,
                feature_names=feature_names,
            )
            r2 = fit.score(
                x=system_data.loc[:, output_variable],
                t=np.arange(0, len(system_data.index), 1),
                u=transformed_inputs,
            )
            if debug:
                print(
                    f"    DEBUG joint_objective: inputs={list(transformed_inputs.columns)}, r2={r2:.4f}"
                )
                try:
                    model.print()
                except:
                    pass
            return -r2  # Negative because minimize

        # initial guesses from SISO optimization
        x0 = []
        for input_var in candidate_inputs:
            shape, scale, loc = best_params.loc[output_variable, input_var]
            x0.extend([shape, scale, loc])
        bounds = []
        for input_var in candidate_inputs:
            bounds.extend(
                [(1.0, 300.0), (1e-5, 300.0), (0.0, 300.0)]
            )  # shape, scale, loc

        # First, compute baseline R² using SISO-optimized params (x0)
        # This ensures we never do worse than the initial guess
        baseline_r2 = -joint_objective(x0, debug=True)
        print(f"  Baseline R² with SISO params: {baseline_r2:.4f}")

        # optimize
        multivariable_iterations = max_iterations * len(candidate_inputs)
        result = minimize(
            joint_objective,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": multivariable_iterations, "disp": verbose},
        )
        # result = minimize(joint_objective, x0, method='L-BFGS-B', bounds=bounds,
        #            options={'maxiter': multivariable_iterations, 'disp': verbose})
        optimized_r2 = -result.fun

        # Use optimized params only if they improve on baseline, otherwise keep SISO params
        if optimized_r2 >= baseline_r2:
            optimized_params = result.x
            print(f"  Optimizer improved R² to {optimized_r2:.4f}")
        else:
            optimized_params = x0
            print(
                f"  Optimizer found worse R² ({optimized_r2:.4f}), keeping SISO params (R² = {baseline_r2:.4f})"
            )

        # extract best params
        for i, input_var in enumerate(candidate_inputs):
            shape = optimized_params[i * 3]
            scale = optimized_params[i * 3 + 1]
            loc = optimized_params[i * 3 + 2]
            best_params.loc[output_variable, input_var] = (shape, scale, loc)
        # compute final r2 with optimized params
        transformed_inputs = pd.DataFrame(index=system_data.index)
        for i, input_var in enumerate(candidate_inputs):
            shape = optimized_params[i * 3]
            scale = optimized_params[i * 3 + 1]
            loc = optimized_params[i * 3 + 2]
            shape_factors = pd.DataFrame(columns=[input_var], index=[1])
            shape_factors.loc[1, input_var] = shape
            scale_factors = pd.DataFrame(columns=[input_var], index=[1])
            scale_factors.loc[1, input_var] = scale
            loc_factors = pd.DataFrame(columns=[input_var], index=[1])
            loc_factors.loc[1, input_var] = loc
            forcing_orig = system_data[[input_var]].copy()
            transformed = transform_inputs(
                shape_factors,
                scale_factors,
                loc_factors,
                system_data.index,
                forcing_orig,
            )
            # Include BOTH original and transformed columns, consistent with SISO phase
            transformed_inputs = pd.concat(
                (transformed_inputs, transformed), axis="columns"
            )
        feature_names = [output_variable] + list(transformed_inputs.columns)
        model = ps.SINDy(
            differentiation_method=ps.FiniteDifference(order=10, drop_endpoints=True),
            feature_library=ps.PolynomialLibrary(
                degree=2, include_bias=False, include_interaction=False
            ),
            optimizer=ps.optimizers.STLSQ(threshold=0, alpha=0),
        )
        fit = model.fit(
            x=system_data.loc[:, output_variable],
            t=np.arange(0, len(system_data.index), 1),
            u=transformed_inputs,
            feature_names=feature_names,
        )
        r2 = fit.score(
            x=system_data.loc[:, output_variable],
            t=np.arange(0, len(system_data.index), 1),
            u=transformed_inputs,
        )

        print(
            f"Testing inputs {candidate_inputs} for output {output_variable} -> r2 = {r2:.4f}"
        )
        if (
            r2 > current_best_r2[output_variable] + 0.01
        ):  # only keep it if it improves the r2 by at least 1%
            # add a conditional here for reducing the number of components in the graph. if it doesn't connect things that were previously unconnected, we don't want it.
            selected_inputs = candidate_inputs
            current_best_r2[output_variable] = r2
            print(
                f"  Accepted new input {forcing_variable}, updated r2 = {current_best_r2[output_variable]:.4f}"
            )
            edges.loc[output_variable, forcing_variable] = 1

            # Update correlation-weighted R² for this output since we added a new input
            # The while loop will re-sort at the next iteration
            update_corr_weighted_r2(output_variable)

        else:
            print(f"  Rejected new input {forcing_variable}, r2 would be {r2:.4f}")

    # transpose edges to have from -> to convention
    edges = edges.T
    # earlier in the code we have dependent variables on the rows and independent on columns.
    # that arrangement makes comparing the effect of potential inputs on each output easier.
    # but for output, it's more intuitive to have from -> to convention, so we transpose before returning.

    return {
        "edges": edges,
        "best_params": best_params,
        "r2_values": r2_values,
        "lead_lag": lead_lag,
    }


# this function takes in the system data,
# which columns are dependent and which are independent,
# as well as an optional constraint on the topology of the digraph
# we will return a digraph (not multidigraph as there are no parallel edges) as defined in https://networkx.org/documentation/stable/reference/classes/digraph.html
# we'll assume there are always self-loops (the derivative always depends on the current value of the variable)
# this will also be returned as an adjacency matrix
# this doesn't go all the way to turning the data into an LTI system. that will be another function that uses this one
def infer_causative_topology(
    system_data,
    dependent_columns,
    independent_columns,
    graph_type="Weak-Conn",
    verbose=False,
    max_iter=250,
    swmm=False,
    method="sindy",  # Changed default from "granger" to "sindy"
    derivative=False,
    sensor_locations=None,
    init_neighbors=3,
):
    """
    Infer causative topology from time series data using SINDy-based optimization.

    Args:
        system_data: pd.DataFrame with time series data
        dependent_columns: list of column names that are dependent variables
        independent_columns: list of column names that are independent/forcing variables
        graph_type: type of graph connectivity requirement ('Weak-Conn' or 'Strong-Conn')
        verbose: whether to print detailed output
        max_iter: maximum iterations for optimization
        swmm: whether this is for SWMM/pystorms data
        method: inference method ('sindy' is the only supported method now)
        derivative: whether to use derivative of response
        sensor_locations: optional dict mapping column names to {"lat": float, "lon": float}
        init_neighbors: initial number of nearest neighbors to evaluate when sensor_locations is provided (default: 3)

    Returns:
        dict with keys: "edges", "best_params", "r2_values", "lead_lag"
        - edges: DataFrame adjacency matrix (from -> to convention)
        - best_params: DataFrame of transformation parameters (shape, scale, loc)
        - r2_values: DataFrame of R^2 values for each potential edge
        - lead_lag: DataFrame of lead/lag values (positive = forcing leads response)
    """
    import warnings

    # Handle deprecated methods
    if method in ("granger", "ccm", "transfer_entropy"):
        warnings.warn(
            f"Method '{method}' is deprecated. The Granger causality, CCM, and "
            "Transfer Entropy methods have been replaced by the improved SINDy-based "
            "topology inference (method='sindy'), which provides significantly better "
            "results. Please use method='sindy' (the new default).",
            DeprecationWarning,
            stacklevel=2,
        )
        # Fall back to new method
        method = "sindy"

    if swmm:
        # do the same for dependent_columns and independent_columns
        dependent_columns = [str(col) for col in dependent_columns]
        independent_columns = [str(col) for col in independent_columns]
        # do the same for the columns of system_data
        system_data.columns = system_data.columns.astype(str)

    # Import and use the new SINDy-based topology inference
    # (using our local implementation)
    result = find_topology_no_geo(
        system_data=system_data,
        dependent_columns=dependent_columns,
        independent_columns=independent_columns,
        sensor_locations=sensor_locations,
        max_iterations=max_iter,
        graph_type=graph_type,
        verbose=verbose,
        init_neighbors=init_neighbors,
    )
    # Convert result to match expected return format for backward compatibility
    # The new method returns edges in from->to convention (transposed from old)
    edges = result["edges"]
    best_params = result["best_params"]
    r2_values = result["r2_values"]
    lead_lag = result["lead_lag"]

    # For backward compatibility with code expecting (causative_topo, total_graph) tuple
    # causative_topo: 'd' for directed edge, 'n' for no edge
    # total_graph: numeric weights (R² values)
    causative_topo = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns
    ).fillna("n")
    total_graph = pd.DataFrame(
        index=dependent_columns, columns=system_data.columns, dtype=float
    ).fillna(0.0)

    # Fill in the edges from the result
    # edges is in from->to convention (row=from, col=to)
    # causative_topo expects row=dependent (to), col=forcing (from)
    for dep_col in dependent_columns:
        for forcing_col in system_data.columns:
            if edges.loc[forcing_col, dep_col] == 1:  # from forcing_col -> to dep_col
                causative_topo.loc[dep_col, forcing_col] = "d"
                total_graph.loc[dep_col, forcing_col] = r2_values.loc[
                    dep_col, forcing_col
                ]

    return causative_topo, total_graph


def find_topology(
    system_data,
    dependent_columns,
    independent_columns,
    method="ccm",
    graph_type="Weak-Conn",
    verbose=False,
):
    """
    DEPRECATED: This function has been replaced by the improved SINDy-based
    topology inference via the local SINDy-based implementation.

    Please use infer_causative_topology(method='sindy') or directly import
    use infer_causative_topology().
    """
    import warnings

    warnings.warn(
        "find_topology() is deprecated. Use infer_causative_topology(method='sindy') "
        "or use infer_causative_topology() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Delegate to the new implementation
    return infer_causative_topology(
        system_data=system_data,
        dependent_columns=dependent_columns,
        independent_columns=independent_columns,
        graph_type=graph_type,
        verbose=verbose,
        method="sindy",
    )
