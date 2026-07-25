from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .model import SINDY_delays_MI
from .transforms import _expected_improvement, _propose_location, _transform_cache


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
    defaults = cast(dict[str, Any], method_defaults.get(optimization_method, {}))

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

    return result.x  # type: ignore[no-any-return]


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
    # first transformation is [1,1,0] for each input
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
            bounds_list: list[list[float]] = []  # type: ignore[no-redef]
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

            rounder_shape = shape_factors.copy(deep=True)
            rounder_shape.loc[tuning_line, tuning_input] = shape_factors.loc[
                tuning_line, tuning_input
            ] * (speed * 1.01)
            rounder_scale = scale_factors.copy(deep=True)
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

            sharper_shape = shape_factors.copy(deep=True)
            sharper_shape.loc[tuning_line, tuning_input] = (
                shape_factors.loc[tuning_line, tuning_input] / speed
            )
            if sharper_shape[tuning_input][tuning_line] < 1:
                sharper = {
                    "error_metrics": {"r2": -1}
                }  # lower bound on shape to avoid inf
            else:
                sharper_scale = scale_factors.copy(deep=True)
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
