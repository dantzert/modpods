import logging
from typing import Any, cast

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
from sklearn.gaussian_process.kernels import Matern  # type: ignore

from ._logging import Verbosity, _normalize_verbose, configure_verbosity
from .kernels import ConvolutionKernel, get_kernel, list_kernels
from .model import SINDY_delays_MI
from .transforms import (
    _expected_improvement,
    _propose_location,
    _transform_cache,
    make_kernel_params,
    params_vector_to_dataframe,
)

logger = logging.getLogger(__name__)


def _run_scipy_optimizer(
    optimization_method: str,
    objective_function,
    bounds: np.ndarray,
    max_iter: int,
    verbose: Verbosity,
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

    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)
        logger.info(
            "Running scipy.optimize.%s with params: %s", optimization_method, params
        )

    # Run the optimizer
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
    windup_timesteps=0,
    init_transforms=1,
    max_transforms=4,
    max_iter=250,
    poly_order=3,
    transform_dependent=False,
    verbose: Verbosity = "warnings",
    extra_verbose=False,
    include_bias=False,
    include_interaction=False,
    bibo_stable=False,
    transform_only=None,
    forcing_coef_constraints=None,
    early_stopping_threshold=0.005,
    optimization_method="bayesian",
    seed=None,
    **optimizer_kwargs,
):
    """Train modpods with a single kernel type."""
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)

    rng = np.random.default_rng(seed) if seed is not None else None

    scipy_optimizer_kwargs = dict(optimizer_kwargs)
    if seed is not None and "seed" not in scipy_optimizer_kwargs:
        scipy_optimizer_kwargs["seed"] = seed

    forcing = system_data[independent_columns].copy(deep=True)
    response = system_data[dependent_columns].copy(deep=True)

    if transform_dependent:
        columns = system_data.columns
    elif transform_only is not None:
        columns = transform_only
    else:
        columns = forcing.columns

    kernel_params = make_kernel_params(kernel, columns, init_transforms, max_transforms)

    results = dict()

    for num_transforms in range(init_transforms, max_transforms + 1):
        logger.debug("num_transforms %s", num_transforms)
        if not num_transforms == init_transforms:
            init_vals = kernel.default_init * (num_transforms - 1)
            for t in range(init_transforms, num_transforms):
                for col in columns:
                    for i, p_name in enumerate(kernel.param_names):
                        kernel_params.loc[(t, p_name), col] = init_vals[i]
            if _normalize_verbose(verbose) != "warnings":
                logger.debug(
                    "starting factors for additional transformation\nshape\nscale\nlocation"
                )
                logger.debug("%s", kernel_params)

        if transform_dependent:
            transform_columns = system_data.columns.tolist()
        elif transform_only is not None:
            transform_columns = transform_only
        else:
            transform_columns = independent_columns

        if optimization_method == "bayesian":
            if _normalize_verbose(verbose) != "warnings":
                logger.info(
                    "Using Bayesian optimization for %s transforms...", num_transforms
                )

            bounds = np.tile(
                kernel.default_bounds, (num_transforms * len(transform_columns), 1)
            )

            def objective_function(params_vector):
                try:
                    opt_params = params_vector_to_dataframe(
                        kernel,
                        params_vector,
                        transform_columns,
                        init_transforms,
                        num_transforms,
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
            Y_sample_list: list[Any] = []

            for i in range(n_initial):
                if rng is not None:
                    x = rng.uniform(bounds[:, 0], bounds[:, 1])
                else:
                    x = np.random.uniform(bounds[:, 0], bounds[:, 1])
                y = objective_function(x)
                X_sample_list.append(x)
                Y_sample_list.append(y)
                if _normalize_verbose(verbose) != "warnings":
                    logger.debug("Initial sample %s/%s: R² = %.6f", i + 1, n_initial, y)

            X_sample: np.ndarray = np.array(X_sample_list)
            Y_sample: np.ndarray = np.array(Y_sample_list).reshape(-1, 1)

            best_r2 = np.max(Y_sample)
            best_params: np.ndarray = X_sample[np.argmax(Y_sample)]

            # Gaussian Process setup
            gpr_kernel = Matern(length_scale=1.0, nu=1.5)
            gpr_random_state = seed if seed is not None else 42
            gpr = GaussianProcessRegressor(
                kernel=gpr_kernel,
                alpha=1e-3,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=gpr_random_state,
            )

            for iteration in range(bayesian_max_iter - n_initial):
                gpr.fit(X_sample, Y_sample.ravel())
                next_x = _propose_location(
                    _expected_improvement, X_sample, Y_sample, gpr, bounds, rng=rng
                )
                next_x = next_x.flatten()
                next_y = objective_function(next_x)

                if _normalize_verbose(verbose) != "warnings":
                    logger.debug(
                        "BO iteration %s/%s: R² = %.6f",
                        iteration + 1,
                        bayesian_max_iter - n_initial,
                        next_y,
                    )

                X_sample = np.append(X_sample, [next_x], axis=0)
                Y_sample = np.append(Y_sample, next_y)

                if next_y > best_r2:
                    best_r2 = next_y
                    best_params = next_x
                    if _normalize_verbose(verbose) != "warnings":
                        logger.debug("New best R² = %.6f", best_r2)

            idx = 0
            for transform in range(1, num_transforms + 1):
                for col in transform_columns:
                    for p_name in kernel.param_names:
                        kernel_params.loc[(transform, p_name), col] = best_params[idx]
                        idx += 1

        else:
            if _normalize_verbose(verbose) != "warnings":
                logger.info(
                    "Using %s optimization for %s transforms...",
                    optimization_method,
                    num_transforms,
                )

            bounds = np.tile(
                kernel.default_bounds, (num_transforms * len(transform_columns), 1)
            )

            def objective_function(params_vector):
                try:
                    opt_params = params_vector_to_dataframe(
                        kernel,
                        params_vector,
                        transform_columns,
                        init_transforms,
                        num_transforms,
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
                optimizer_kwargs=scipy_optimizer_kwargs,
            )

            idx = 0
            for transform in range(1, num_transforms + 1):
                for col in transform_columns:
                    for p_name in kernel.param_names:
                        kernel_params.loc[(transform, p_name), col] = best_params[idx]
                        idx += 1

        logger.info(
            "Optimization complete. Using optimized parameters for final model."
        )
        final_model = SINDY_delays_MI(
            kernel,
            kernel_params,
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
        logger.info("kernel params")
        logger.info("%s", kernel_params)
        results[num_transforms] = {
            "final_model": final_model.copy(),
            "kernel_type": kernel.name,
            "kernel_params": kernel_params.copy(deep=True),
            "windup_timesteps": windup_timesteps,
            "dependent_columns": dependent_columns,
            "independent_columns": independent_columns,
            "transform_cache": _transform_cache,
        }

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
    verbose: Verbosity = "warnings",
    include_bias=False,
    include_interaction=False,
    bibo_stable=False,
    transform_only=None,
    forcing_coef_constraints=None,
    early_stopping_threshold=0.005,
    optimization_method="bayesian",
    kernel="gamma",
    **optimizer_kwargs,
):
    """Train a delay-io model with pluggable convolution kernels.

    Args:
        kernel: ConvolutionKernel instance, kernel name string, "try-all", or "run-all".
            - "try-all": cheap fit all kernels, pick best R², refit expensively.
            - "run-all": expensive fit all kernels, return all results.
            - default "gamma" preserves backward compatibility.

    Returns:
        dict keyed by num_transforms.
    """
    if kernel in ("try-all", "run-all"):
        all_results = dict()
        cheap = kernel == "try-all"
        for name in list_kernels():
            if _normalize_verbose(verbose) != "warnings":
                mode = "cheap" if cheap else "expensive"
                logger.info("Running %s fit with kernel: %s", mode, name)
            k = get_kernel(name)
            if cheap:
                cheap_kwargs = dict(optimizer_kwargs)
                cheap_max_iter = max(5, max_iter // 10)
                cheap_results = _train_single_kernel(
                    kernel=k,
                    system_data=system_data,
                    dependent_columns=dependent_columns,
                    independent_columns=independent_columns,
                    windup_timesteps=windup_timesteps,
                    init_transforms=init_transforms,
                    max_transforms=max_transforms,
                    max_iter=cheap_max_iter,
                    poly_order=poly_order,
                    transform_dependent=transform_dependent,
                    verbose=verbose,
                    extra_verbose=False,
                    include_bias=include_bias,
                    include_interaction=include_interaction,
                    bibo_stable=bibo_stable,
                    transform_only=transform_only,
                    forcing_coef_constraints=forcing_coef_constraints,
                    early_stopping_threshold=early_stopping_threshold,
                    optimization_method=optimization_method,
                    optimizer_kwargs=cheap_kwargs,
                )
                all_results[name] = cheap_results
            else:
                full_results = _train_single_kernel(
                    kernel=k,
                    system_data=system_data,
                    dependent_columns=dependent_columns,
                    independent_columns=independent_columns,
                    windup_timesteps=windup_timesteps,
                    init_transforms=init_transforms,
                    max_transforms=max_transforms,
                    max_iter=max_iter,
                    poly_order=poly_order,
                    transform_dependent=transform_dependent,
                    verbose=verbose,
                    extra_verbose=False,
                    include_bias=include_bias,
                    include_interaction=include_interaction,
                    bibo_stable=bibo_stable,
                    transform_only=transform_only,
                    forcing_coef_constraints=forcing_coef_constraints,
                    early_stopping_threshold=early_stopping_threshold,
                    optimization_method=optimization_method,
                    optimizer_kwargs=optimizer_kwargs,
                )
                all_results[name] = full_results

        if cheap:
            best_kernel_name = None
            best_r2 = -float("inf")
            for name, res in all_results.items():
                for nt, entry in res.items():
                    r2 = entry["final_model"]["error_metrics"]["r2"]
                    if r2 > best_r2:
                        best_r2 = r2
                        best_kernel_name = name
            if _normalize_verbose(verbose) != "warnings":
                logger.info(
                    "Best kernel from cheap pass: %s (R² = %.4f)",
                    best_kernel_name,
                    best_r2,
                )
            if best_kernel_name is None:
                raise RuntimeError("No kernel produced a valid model in try-all mode.")
            return _train_single_kernel(
                kernel=get_kernel(best_kernel_name),
                system_data=system_data,
                dependent_columns=dependent_columns,
                independent_columns=independent_columns,
                windup_timesteps=windup_timesteps,
                init_transforms=init_transforms,
                max_transforms=max_transforms,
                max_iter=max_iter,
                poly_order=poly_order,
                transform_dependent=transform_dependent,
                verbose=verbose,
                extra_verbose=False,
                include_bias=include_bias,
                include_interaction=include_interaction,
                bibo_stable=bibo_stable,
                transform_only=transform_only,
                forcing_coef_constraints=forcing_coef_constraints,
                early_stopping_threshold=early_stopping_threshold,
                optimization_method=optimization_method,
                optimizer_kwargs=optimizer_kwargs,
            )
        else:
            return all_results

    kernel = get_kernel(kernel)
    return _train_single_kernel(
        kernel=kernel,
        system_data=system_data,
        dependent_columns=dependent_columns,
        independent_columns=independent_columns,
        windup_timesteps=windup_timesteps,
        init_transforms=init_transforms,
        max_transforms=max_transforms,
        max_iter=max_iter,
        poly_order=poly_order,
        transform_dependent=transform_dependent,
        verbose=verbose,
        extra_verbose=False,
        include_bias=include_bias,
        include_interaction=include_interaction,
        bibo_stable=bibo_stable,
        transform_only=transform_only,
        forcing_coef_constraints=forcing_coef_constraints,
        early_stopping_threshold=early_stopping_threshold,
        optimization_method=optimization_method,
        optimizer_kwargs=optimizer_kwargs,
    )
