import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.gaussian_process.kernels import Matern  # type: ignore

from .kernels import ConvolutionKernel, get_kernel, list_kernels
from ._logging import Verbosity, _normalize_verbose, configure_verbosity
from .model import SINDY_delays_MI
from .transforms import _expected_improvement, _propose_location, _transform_cache, make_kernel_params, params_vector_to_dataframe

logger = logging.getLogger(__name__)


def _run_scipy_optimizer(
    optimization_method: str,
    objective_function,
    bounds: np.ndarray,
    max_iter: int,
    verbose: Verbosity,
    optimizer_kwargs: dict,
) -> np.ndarray:
    """Dispatch to scipy.optimize methods for global optimization."""
    import scipy.optimize as opt

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
            "maxiter": max_iter * 4,
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

    defaults = cast(dict[str, Any], method_defaults.get(optimization_method, {}))
    params = {**defaults, **optimizer_kwargs}
    optimizer = getattr(opt, optimization_method, None)
    if optimizer is None:
        raise ValueError(
            f"Unknown optimization_method: '{optimization_method}'. "
            f"Supported scipy.optimize methods: {list(method_defaults.keys())}, "
            f"or 'bayesian' for built-in Bayesian optimization."
        )

    if verbose:
        print(f"  Running scipy.optimize.{optimization_method} with params: {params}")

    result = optimizer(objective_function, bounds, **params)

    if _normalize_verbose(verbose) != "warnings":
        logger.info(
            "Optimization complete. Success: %s, Message: %s",
            result.success,
            result.message,
        )
        logger.info("Best value: %.6f (R²)", -result.fun)

    return result.x  # type: ignore[no-any-return]


def _train_single_kernel(
    kernel: ConvolutionKernel,
    system_data,
    dependent_columns,
    independent_columns,
    windup_timesteps,
    init_transforms,
    max_transforms,
    max_iter,
    poly_order,
    transform_dependent,
    verbose,
    extra_verbose,
    include_bias,
    include_interaction,
    bibo_stable,
    transform_only,
    forcing_coef_constraints,
    early_stopping_threshold,
    optimization_method,
    optimizer_kwargs,
):
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)

    """Train modpods with a single kernel type.

    Returns:
        results dict keyed by num_transforms, each entry containing
        'final_model', 'kernel_type', 'kernel_params', etc.
    """
    forcing = system_data[independent_columns].copy(deep=True)

    response = system_data[dependent_columns].copy(deep=True)

    results = dict()  # to store the optimized models for each number of transformations

    if transform_dependent:
        columns = system_data.columns
    elif transform_only is not None:
        columns = transform_only
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
    for num_transforms in range(init_transforms, max_transforms + 1):
        logger.debug("num_transforms %s", num_transforms)
        if not num_transforms == init_transforms:  # if we're not starting right now
            # start dull
            shape_factors.iloc[num_transforms - 1, :] = 10 * (
                num_transforms - 1
            )  # start with a broad peak centered at ten timesteps
            scale_factors.iloc[num_transforms - 1, :] = 1
            loc_factors.iloc[num_transforms - 1, :] = 0
            if _normalize_verbose(verbose) != "warnings":
                logger.debug(
                    "starting factors for additional transformation\nshape\nscale\nlocation"
                )
                logger.debug("%s", shape_factors)
                logger.debug("%s", scale_factors)
                logger.debug("%s", loc_factors)

        # Choose optimization method
        if optimization_method == "bayesian":
            if _normalize_verbose(verbose) != "warnings":
                logger.info(
                    "Using Bayesian optimization for %s transforms...", num_transforms
                )

            # Determine which columns to transform
            if transform_dependent:
                transform_columns = system_data.columns.tolist()
            elif transform_only is not None:
                transform_columns = transform_only
            else:
                transform_columns = independent_columns

        if optimization_method == "bayesian":
            if verbose:
                print(f"Using Bayesian optimization for {num_transforms} transforms...")

            bounds = np.tile(kernel.default_bounds, (num_transforms * len(transform_columns), 1))

            def objective_function(params_vector):
                try:
                    opt_params = params_vector_to_dataframe(
                        kernel, params_vector, transform_columns,
                        init_transforms, num_transforms
                    )
                    result = SINDY_delays_MI(
                        kernel,
                        opt_params,
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
                        verbose=verbose,
                    )
                    r2 = result["error_metrics"]["r2"]
                    if _normalize_verbose(verbose) != "warnings":
                        logger.debug("  R² = %.6f", r2)
                    return r2
                except Exception as e:
                    if _normalize_verbose(verbose) != "warnings":
                        logger.debug("  Evaluation failed: %s", e)
                    return -1.0

            bayesian_max_iter = min(max_iter * 4, 200)
            n_initial = min(30, max(20, int(bayesian_max_iter * 0.6)))
            X_sample_list: list[Any] = []

            for i in range(n_initial):
                x = np.random.uniform(bounds[:, 0], bounds[:, 1])
                y = objective_function(x)
                X_sample_list.append(x)
                if _normalize_verbose(verbose) != "warnings":
                    logger.debug("Initial sample %s/%s: R² = %.6f", i + 1, n_initial, y)

            X_sample = np.array(X_sample_list)
            Y_sample = np.array([objective_function(x) for x in X_sample_list]).reshape(-1, 1)

            best_r2 = float(np.max(Y_sample))
            best_params = X_sample[np.argmax(Y_sample)].copy()

            gpr_kernel = Matern(length_scale=1.0, nu=1.5)
            gpr = GaussianProcessRegressor(
                kernel=gpr_kernel,
                alpha=1e-3,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42,
            )

            for iteration in range(bayesian_max_iter - n_initial):
                gpr.fit(X_sample, Y_sample.ravel())
                next_x = _propose_location(
                    _expected_improvement, X_sample, Y_sample, gpr, bounds
                )
                next_x = next_x.flatten()
                next_y = objective_function(next_x)

                if verbose:
                    print(
                        f"  BO iteration {iteration+1}/{bayesian_max_iter-n_initial}: R² = {next_y:.6f}"
                    )

                X_sample = np.append(X_sample, [next_x], axis=0)
                Y_sample = np.append(Y_sample, next_y)

                if next_y > best_r2:
                    best_r2 = next_y
                    best_params = next_x.copy()
                    if _normalize_verbose(verbose) != "warnings":
                        logger.debug("New best R² = %.6f", best_r2)

            # Convert best parameters back to DataFrames
            idx = 0
            for transform in range(1, num_transforms + 1):
                for col in transform_columns:
                    shape_factors.loc[transform, col] = best_params[idx]
                    scale_factors.loc[transform, col] = best_params[idx + 1]
                    loc_factors.loc[transform, col] = best_params[idx + 2]
                    idx += 3

        else:
            if _normalize_verbose(verbose) != "warnings":
                logger.info(
                    "Using %s optimization for %s transforms...",
                    optimization_method,
                    num_transforms,
                )

            bounds = np.tile(kernel.default_bounds, (num_transforms * len(transform_columns), 1))

            def objective_function(params_vector):
                try:
                    opt_params = params_vector_to_dataframe(
                        kernel, params_vector, transform_columns,
                        init_transforms, num_transforms
                    )
                    result = SINDY_delays_MI(
                        kernel,
                        opt_params,
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
                        verbose=verbose,
                    )
                    r2 = result["error_metrics"]["r2"]
                    if _normalize_verbose(verbose) != "warnings":
                        logger.debug("  R² = %.6f", r2)
                    return -r2
                except Exception as e:
                    if _normalize_verbose(verbose) != "warnings":
                        logger.debug("  Evaluation failed: %s", e)
                    return 1.0

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

        # For bayesian and scipy.optimize methods, we're done with optimization
        logger.info(
            "Optimization complete. Using optimized parameters for final model."
        )
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
            verbose=verbose,
        )
        logger.info("Final model:")
        try:
            logger.info("%s", final_model["model"].print(precision=5))
        except Exception as e:
            logger.warning("%s", e)
        logger.info("R^2")
        logger.info("%s", final_model["error_metrics"]["r2"])
        logger.info("shape factors")
        logger.info("%s", shape_factors)
        logger.info("scale factors")
        logger.info("%s", scale_factors)
        logger.info("location factors")
        logger.info("%s", loc_factors)
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
            logger.warning(
                "Last transformation added less than %s %% to R2 score. Terminating early.",
                early_stopping_threshold * 100,
            )
            break

    return results
