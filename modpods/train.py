import logging
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import pandas as pd
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


class OptimizerStrategy(ABC):
    """Abstract base class for optimization strategies."""

    @abstractmethod
    def optimize(
        self,
        objective_function,
        bounds: np.ndarray,
        max_iter: int,
        verbose: Verbosity,
        optimizer_kwargs: dict,
    ) -> np.ndarray:
        """Run optimization and return best parameter vector.

        Args:
            objective_function: Callable that takes parameter vector and
                returns scalar to minimize.
            bounds: Array of [min, max] bounds for each parameter.
            max_iter: Maximum iterations.
            verbose: Verbosity level.
            optimizer_kwargs: Additional keyword arguments for the optimizer.

        Returns:
            Best parameter vector found.
        """
        ...


class BayesianOptimizer(OptimizerStrategy):
    """Bayesian optimization using Gaussian Process and Expected Improvement."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def optimize(
        self,
        objective_function,
        bounds: np.ndarray,
        max_iter: int,
        verbose: Verbosity,
        optimizer_kwargs: dict,
    ) -> np.ndarray:
        if _normalize_verbose(verbose) != "warnings":
            configure_verbosity(verbose)
            logger.info("Using Bayesian optimization...")

        bayesian_max_iter = min(max_iter * 4, 200)
        n_initial = min(30, max(20, int(bayesian_max_iter * 0.6)))

        rng = np.random.default_rng(self.seed) if self.seed is not None else None
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

        gpr_kernel = Matern(length_scale=1.0, nu=1.5)
        gpr_random_state = self.seed if self.seed is not None else 42
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

        return best_params


class ScipyOptimizer(OptimizerStrategy):
    """Wrapper for scipy.optimize global optimization methods."""

    def __init__(self, method: str = "differential_evolution") -> None:
        self.method = method

    def optimize(
        self,
        objective_function,
        bounds: np.ndarray,
        max_iter: int,
        verbose: Verbosity,
        optimizer_kwargs: dict,
    ) -> np.ndarray:
        def negated_objective(x):
            return -objective_function(x)

        return _run_scipy_optimizer(
            optimization_method=self.method,
            objective_function=negated_objective,
            bounds=bounds,
            max_iter=max_iter,
            verbose=verbose,
            optimizer_kwargs=optimizer_kwargs,
        )


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

    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)
        logger.info(
            "Running scipy.optimize.%s with params: %s", optimization_method, params
        )

    result = optimizer(objective_function, bounds, **params)

    if _normalize_verbose(verbose) != "warnings":
        logger.info(
            "Optimization complete. Success: %s, Message: %s",
            result.success,
            result.message,
        )
        logger.info("Best value: %.6f (R²)", -result.fun)

    return result.x  # type: ignore[no-any-return]


def _auto_max_transforms(kernel: ConvolutionKernel, max_transforms: int) -> int:
    """Auto-adjust max_transforms based on kernel type.

    Gamma-like kernels use cascades of first-order systems, needing many transforms.
    Underdamped/2nd-order kernels naturally represent the dynamics in 1 transform.
    """
    if kernel.name == "underdamped":
        return min(max_transforms, 1)
    return max_transforms


class SingleKernelTrainer:
    """Train a modpods model with a single kernel type."""

    def __init__(
        self,
        kernel: ConvolutionKernel,
        system_data: pd.DataFrame,
        dependent_columns: list[str],
        independent_columns: list[str],
        windup_timesteps: int = 0,
        init_transforms: int = 1,
        max_transforms: int = 4,
        max_iter: int = 250,
        poly_order: int = 3,
        transform_dependent: bool = False,
        verbose: Verbosity = "warnings",
        include_bias: bool = False,
        include_interaction: bool = False,
        bibo_stable: bool = False,
        transform_only: list[str] | None = None,
        forcing_coef_constraints: Any = None,
        constraints: Any = None,
        early_stopping_threshold: float = 0.005,
        optimization_method: str = "bayesian",
        seed: int | None = None,
        optimizer_kwargs: dict | None = None,
    ) -> None:
        self.kernel = kernel
        self.system_data = system_data
        self.dependent_columns = dependent_columns
        self.independent_columns = independent_columns
        self.windup_timesteps = windup_timesteps
        self.init_transforms = init_transforms
        self.max_transforms = _auto_max_transforms(kernel, max_transforms)
        self.max_iter = max_iter
        self.poly_order = poly_order
        self.transform_dependent = transform_dependent
        self.verbose = verbose
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.bibo_stable = bibo_stable
        self.transform_only = transform_only
        self.forcing_coef_constraints = forcing_coef_constraints
        self.constraints = constraints
        self.early_stopping_threshold = early_stopping_threshold
        self.optimization_method = optimization_method
        self.seed = seed
        self.optimizer_kwargs = optimizer_kwargs or {}

        if transform_dependent:
            self.columns = system_data.columns.tolist()
        elif transform_only is not None:
            self.columns = transform_only
        else:
            self.columns = system_data[independent_columns].columns.tolist()

        self.kernel_params = make_kernel_params(
            kernel, self.columns, init_transforms, self.max_transforms
        )
        self.results: dict[int, dict[str, Any]] = {}

    def _get_transform_columns(self) -> list[str]:
        if self.transform_dependent:
            return list(self.system_data.columns)
        if self.transform_only is not None:
            return self.transform_only
        return self.independent_columns

    def _create_objective(self, transform_columns: list[str], num_transforms: int):
        def objective_function(params_vector):
            try:
                opt_params = params_vector_to_dataframe(
                    self.kernel,
                    params_vector,
                    transform_columns,
                    self.init_transforms,
                    num_transforms,
                )

                # For unstable kernels, optimize for full system prediction accuracy (NSE)
                # instead of just immediate SINDy regression R²
                is_unstable = self.kernel.is_unstable_params(*params_vector)

                if is_unstable:
                    # Use full system simulation for unstable kernels
                    result = SINDY_delays_MI(
                        self.kernel,
                        opt_params,
                        self.system_data.index,
                        self.system_data[self.independent_columns],
                        self.system_data[self.dependent_columns],
                        True,  # final_run=True: compute full system simulation metrics
                        self.poly_order,
                        self.include_bias,
                        self.include_interaction,
                        self.windup_timesteps,
                        self.bibo_stable,
                        self.transform_dependent,
                        self.transform_only,
                        self.forcing_coef_constraints,
                        self.constraints,
                        transform_cache=_transform_cache,
                        verbose=self.verbose,
                    )
                    # Use NSE (Nash-Sutcliffe Efficiency) as the metric for full system accuracy
                    # NSE = 1 - (sum of squared errors / sum of squared deviations from mean)
                    # NSE = 1 is perfect, NSE = 0 is as good as mean, NSE < 0 is worse than mean
                    nse = result["error_metrics"].get("nse", -1.0)

                    # Get the identified model to check eigenvalues
                    model = result.get("model")
                    eigenval_penalty = 0.0
                    if model is not None and hasattr(model, "A"):
                        try:
                            A = np.array(model.A)
                            eigvals = np.linalg.eigvals(A)
                            max_real = np.max(np.real(eigvals))
                            # Penalize extreme eigenvalues (true unstable pole is ~4.35)
                            # Penalize both too large (>50) and too small (<0.1) unstable poles
                            if max_real > 50.0:
                                eigenval_penalty = (
                                    max_real - 50.0
                                ) / 50.0  # Linear penalty for too large
                            elif max_real > 0 and max_real < 0.1:
                                eigenval_penalty = (
                                    0.1 - max_real
                                ) / 0.1  # Penalty for too small
                        except Exception:
                            pass

                    # Penalized NSE: reward good fit, penalize extreme eigenvalues
                    penalized_nse = nse - eigenval_penalty

                    if _normalize_verbose(self.verbose) != "warnings":
                        logger.debug(
                            "  NSE = %.6f, eigval_penalty = %.6f, penalized = %.6f",
                            nse,
                            eigenval_penalty,
                            penalized_nse,
                        )
                    return penalized_nse
                else:
                    # Stable kernels: use immediate SINDy regression R² (fast)
                    result = SINDY_delays_MI(
                        self.kernel,
                        opt_params,
                        self.system_data.index,
                        self.system_data[self.independent_columns],
                        self.system_data[self.dependent_columns],
                        False,
                        self.poly_order,
                        self.include_bias,
                        self.include_interaction,
                        self.windup_timesteps,
                        self.bibo_stable,
                        self.transform_dependent,
                        self.transform_only,
                        self.forcing_coef_constraints,
                        self.constraints,
                        transform_cache=_transform_cache,
                        verbose=self.verbose,
                    )
                    r2 = result["error_metrics"]["r2"]
                    if _normalize_verbose(self.verbose) != "warnings":
                        logger.debug("  R² = %.6f", r2)
                    return r2

            except Exception as e:
                if _normalize_verbose(self.verbose) != "warnings":
                    logger.debug("  Evaluation failed: %s", e)
                return -1.0

        return objective_function

    def _get_optimizer(self) -> OptimizerStrategy:
        if self.optimization_method == "bayesian":
            return BayesianOptimizer(seed=self.seed)
        return ScipyOptimizer(method=self.optimization_method)

    def _initialize_transform_params(self, num_transforms: int) -> None:
        if num_transforms == self.init_transforms:
            return
        init_vals = self.kernel.default_init * (num_transforms - 1)
        for t in range(self.init_transforms, num_transforms):
            for col in self.columns:
                for i, p_name in enumerate(self.kernel.param_names):
                    self.kernel_params.loc[(t, p_name), col] = init_vals[i]
        if _normalize_verbose(self.verbose) != "warnings":
            logger.debug(
                "starting factors for additional transformation\nshape\nscale\nlocation"
            )
            logger.debug("%s", self.kernel_params)

    def _optimize_params(self, num_transforms: int) -> np.ndarray:
        transform_columns = self._get_transform_columns()
        bounds = np.tile(
            self.kernel.default_bounds, (num_transforms * len(transform_columns), 1)
        )
        objective = self._create_objective(transform_columns, num_transforms)
        optimizer = self._get_optimizer()
        return optimizer.optimize(
            objective_function=objective,
            bounds=bounds,
            max_iter=self.max_iter,
            verbose=self.verbose,
            optimizer_kwargs=self.optimizer_kwargs,
        )

    def _update_kernel_params(
        self, best_params: np.ndarray, num_transforms: int
    ) -> None:
        transform_columns = self._get_transform_columns()
        idx = 0
        for transform in range(1, num_transforms + 1):
            for col in transform_columns:
                for p_name in self.kernel.param_names:
                    self.kernel_params.loc[(transform, p_name), col] = best_params[idx]
                    idx += 1

    def _train_single_transform_count(self, num_transforms: int) -> dict[str, Any]:
        self._initialize_transform_params(num_transforms)

        if _normalize_verbose(self.verbose) != "warnings":
            logger.info(
                "Using %s optimization for %s transforms...",
                self.optimization_method,
                num_transforms,
            )

        best_params = self._optimize_params(num_transforms)
        self._update_kernel_params(best_params, num_transforms)

        if _normalize_verbose(self.verbose) != "warnings":
            logger.info(
                "Optimization complete. Using optimized parameters for final model."
            )

        final_model = SINDY_delays_MI(
            self.kernel,
            self.kernel_params,
            self.system_data.index,
            self.system_data[self.independent_columns],
            self.system_data[self.dependent_columns],
            True,
            self.poly_order,
            self.include_bias,
            self.include_interaction,
            self.windup_timesteps,
            self.bibo_stable,
            self.transform_dependent,
            self.transform_only,
            self.forcing_coef_constraints,
            self.constraints,
            transform_cache=_transform_cache,
            verbose=self.verbose,
        )
        if _normalize_verbose(self.verbose) != "warnings":
            logger.info("Final model:")
            try:
                logger.info("%s", final_model["model"].print(precision=5))
            except Exception as e:
                logger.warning("%s", e)
            logger.info("R^2")
            logger.info("%s", final_model["error_metrics"]["r2"])
            logger.info("kernel params")
            logger.info("%s", self.kernel_params)

        return {
            "final_model": final_model.copy(),
            "kernel_type": self.kernel.name,
            "kernel_params": self.kernel_params.copy(deep=True),
            "windup_timesteps": self.windup_timesteps,
            "dependent_columns": self.dependent_columns,
            "independent_columns": self.independent_columns,
            "transform_cache": _transform_cache,
        }

    def train(self) -> dict[int, dict[str, Any]]:
        for num_transforms in range(self.init_transforms, self.max_transforms + 1):
            if _normalize_verbose(self.verbose) != "warnings":
                logger.debug("num_transforms %s", num_transforms)

            self.results[num_transforms] = self._train_single_transform_count(
                num_transforms
            )

            if (
                num_transforms > self.init_transforms
                and self.results[num_transforms]["final_model"]["error_metrics"]["r2"]
                - self.results[num_transforms - 1]["final_model"]["error_metrics"]["r2"]
                < self.early_stopping_threshold
            ):
                logger.warning(
                    "Last transformation added less than %s %% to R2 score."
                    " Terminating early.",
                    self.early_stopping_threshold * 100,
                )
                break

        return self.results


class MultiKernelTrainer:
    """Train models with multiple kernels."""

    def __init__(
        self,
        system_data: pd.DataFrame,
        dependent_columns: list[str],
        independent_columns: list[str],
        mode: str,
        windup_timesteps: int = 0,
        init_transforms: int = 1,
        max_transforms: int = 4,
        max_iter: int = 250,
        poly_order: int = 3,
        transform_dependent: bool = False,
        verbose: Verbosity = "warnings",
        include_bias: bool = False,
        include_interaction: bool = False,
        bibo_stable: bool = False,
        transform_only: list[str] | None = None,
        forcing_coef_constraints: Any = None,
        constraints: Any = None,
        early_stopping_threshold: float = 0.005,
        optimization_method: str = "bayesian",
        seed: int | None = None,
        optimizer_kwargs: dict | None = None,
    ) -> None:
        self.system_data = system_data
        self.dependent_columns = dependent_columns
        self.independent_columns = independent_columns
        self.mode = mode
        self.windup_timesteps = windup_timesteps
        self.init_transforms = init_transforms
        self.max_transforms = max_transforms
        self.max_iter = max_iter
        self.poly_order = poly_order
        self.transform_dependent = transform_dependent
        self.verbose = verbose
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.bibo_stable = bibo_stable
        self.transform_only = transform_only
        self.forcing_coef_constraints = forcing_coef_constraints
        self.constraints = constraints
        self.early_stopping_threshold = early_stopping_threshold
        self.optimization_method = optimization_method
        self.seed = seed
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.all_results: dict[str, dict[int, dict[str, Any]]] = {}

    def _train_kernel(
        self, kernel: ConvolutionKernel, max_iter: int
    ) -> dict[int, dict[str, Any]]:
        trainer = SingleKernelTrainer(
            kernel=kernel,
            system_data=self.system_data,
            dependent_columns=self.dependent_columns,
            independent_columns=self.independent_columns,
            windup_timesteps=self.windup_timesteps,
            init_transforms=self.init_transforms,
            max_transforms=self.max_transforms,
            max_iter=max_iter,
            poly_order=self.poly_order,
            transform_dependent=self.transform_dependent,
            verbose=self.verbose,
            include_bias=self.include_bias,
            include_interaction=self.include_interaction,
            bibo_stable=self.bibo_stable,
            transform_only=self.transform_only,
            forcing_coef_constraints=self.forcing_coef_constraints,
            constraints=self.constraints,
            early_stopping_threshold=self.early_stopping_threshold,
            optimization_method=self.optimization_method,
            seed=self.seed,
            optimizer_kwargs=self.optimizer_kwargs,
        )
        return trainer.train()

    def _find_best_kernel(self) -> tuple[str, float]:
        best_kernel_name = None
        best_r2 = -float("inf")
        for name, res in self.all_results.items():
            for nt, entry in res.items():
                r2 = entry["final_model"]["error_metrics"]["r2"]
                if r2 > best_r2:
                    best_r2 = r2
                    best_kernel_name = name
        if best_kernel_name is None:
            raise RuntimeError("No kernel produced a valid model in try-all mode.")
        return best_kernel_name, best_r2

    def train(self) -> Any:
        cheap = self.mode == "try-all"

        for name in list_kernels():
            if _normalize_verbose(self.verbose) != "warnings":
                mode = "cheap" if cheap else "expensive"
                logger.info("Running %s fit with kernel: %s", mode, name)
            k = get_kernel(name)
            if cheap:
                cheap_max_iter = max(5, self.max_iter // 10)
                self.all_results[name] = self._train_kernel(k, cheap_max_iter)
            else:
                self.all_results[name] = self._train_kernel(k, self.max_iter)

        if cheap:
            best_kernel_name, best_r2 = self._find_best_kernel()
            if _normalize_verbose(self.verbose) != "warnings":
                logger.info(
                    "Best kernel from cheap pass: %s (R² = %.4f)",
                    best_kernel_name,
                    best_r2,
                )
            return self._train_kernel(get_kernel(best_kernel_name), self.max_iter)

        return self.all_results


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
    constraints=None,
    early_stopping_threshold=0.005,
    optimization_method="bayesian",
    kernel="gamma",
    max_states=5,
    seed=None,
    **optimizer_kwargs,
):
    """Train a delay-IO model with pluggable convolution kernels.

    Args:
        kernel: ConvolutionKernel instance, kernel name string, "try-all", "run-all",
            "canonical_lti", or "canonical_lti_incremental".
            - "try-all": cheap fit all kernels, pick best R², refit expensively.
            - "run-all": expensive fit all kernels, return all results.
            - "canonical_lti": single canonical LTI with fixed max_states.
            - "canonical_lti_incremental": incremental state dimension canonical LTI.
            - default "gamma" preserves backward compatibility.

        max_transforms: Maximum number of transforms. For underdamped kernel,
            this is automatically limited to 1 (since underdamped oscillator
            naturally represents a 2nd-order system in a single transform).
            For gamma/lognormal/bimodal_gamma/exponential_growth, cascades
            of first-order systems are used, so more transforms may be needed.

        max_states: Maximum state dimension for canonical LTI kernels (default 5).

    Returns:
        dict keyed by num_transforms.
    """
    if kernel in ("try-all", "run-all"):
        trainer = MultiKernelTrainer(
            system_data=system_data,
            dependent_columns=dependent_columns,
            independent_columns=independent_columns,
            mode=kernel,
            windup_timesteps=windup_timesteps,
            init_transforms=init_transforms,
            max_transforms=max_transforms,
            max_iter=max_iter,
            poly_order=poly_order,
            transform_dependent=transform_dependent,
            verbose=verbose,
            include_bias=include_bias,
            include_interaction=include_interaction,
            bibo_stable=bibo_stable,
            transform_only=transform_only,
            forcing_coef_constraints=forcing_coef_constraints,
            constraints=constraints,
            early_stopping_threshold=early_stopping_threshold,
            optimization_method=optimization_method,
            seed=seed,
            optimizer_kwargs=optimizer_kwargs,
        )
        return trainer.train()

    if kernel in ("canonical_lti", "canonical_lti_incremental"):
        max_states = optimizer_kwargs.get("max_states", 5)
        if kernel == "canonical_lti_incremental":
            k = get_kernel("canonical_lti_incremental")
            if hasattr(k, "max_states"):
                k.max_states = max_states
        else:
            k = get_kernel("canonical_lti")
            if hasattr(k, "max_states"):
                k.max_states = max_states

        auto_max_transforms = 1  # Canonical LTI doesn't use multiple transforms
        if _normalize_verbose(verbose) != "warnings":
            logger.info(
                "Using canonical LTI kernel with max_states=%s (no transforms needed)",
                max_states,
            )

        single_trainer = SingleKernelTrainer(
            kernel=k,
            system_data=system_data,
            dependent_columns=dependent_columns,
            independent_columns=independent_columns,
            windup_timesteps=windup_timesteps,
            init_transforms=1,
            max_transforms=1,
            max_iter=max_iter,
            poly_order=poly_order,
            transform_dependent=transform_dependent,
            verbose=verbose,
            include_bias=include_bias,
            include_interaction=include_interaction,
            bibo_stable=bibo_stable,
            transform_only=transform_only,
            forcing_coef_constraints=forcing_coef_constraints,
            constraints=constraints,
            early_stopping_threshold=early_stopping_threshold,
            optimization_method=optimization_method,
            seed=seed,
            optimizer_kwargs=optimizer_kwargs,
        )
        return single_trainer.train()

    if kernel == "decoupled_lti":
        max_states = optimizer_kwargs.get("max_states", 5)
        k = get_kernel("decoupled_lti")
        if hasattr(k, "max_states"):
            k.max_states = max_states

        return decoupled_lti_train(
            system_data=system_data,
            dependent_columns=dependent_columns,
            independent_columns=independent_columns,
            windup_timesteps=windup_timesteps,
            max_states=max_states,
            max_iter=max_iter,
            verbose=verbose,
            optimization_method=optimization_method,
            seed=seed,
            **optimizer_kwargs,
        )

    k = get_kernel(kernel)
    # Auto-limit transforms for underdamped kernel
    auto_max_transforms = _auto_max_transforms(k, max_transforms)
    if (
        auto_max_transforms != max_transforms
        and _normalize_verbose(verbose) != "warnings"
    ):
        logger.info(
            "Auto-limiting max_transforms from %s to %s for '%s' kernel "
            "(2nd-order systems don't need cascades)",
            max_transforms,
            auto_max_transforms,
            k.name,
        )

    single_trainer = SingleKernelTrainer(
        kernel=k,
        system_data=system_data,
        dependent_columns=dependent_columns,
        independent_columns=independent_columns,
        windup_timesteps=windup_timesteps,
        init_transforms=init_transforms,
        max_transforms=auto_max_transforms,
        max_iter=max_iter,
        poly_order=poly_order,
        transform_dependent=transform_dependent,
        verbose=verbose,
        include_bias=include_bias,
        include_interaction=include_interaction,
        bibo_stable=bibo_stable,
        transform_only=transform_only,
        forcing_coef_constraints=forcing_coef_constraints,
        constraints=constraints,
        early_stopping_threshold=early_stopping_threshold,
        optimization_method=optimization_method,
        seed=seed,
        optimizer_kwargs=optimizer_kwargs,
    )
    return single_trainer.train()


class DecoupledLTITrainer:
    """Direct LTI optimization bypassing delay-model architecture.

    Optimizes A, B, C, D matrices directly for each output using
    controllable canonical form. Uses NSE (full system simulation accuracy)
    as objective instead of SINDy R².
    """

    def __init__(
        self,
        system_data: pd.DataFrame,
        dependent_columns: list[str],
        independent_columns: list[str],
        windup_timesteps: int = 0,
        max_states: int = 5,
        max_iter: int = 250,
        verbose: Verbosity = "warnings",
        optimization_method: str = "bayesian",
        seed: int | None = None,
        optimizer_kwargs: dict | None = None,
    ) -> None:
        self.system_data = system_data
        self.dependent_columns = dependent_columns
        self.independent_columns = independent_columns
        self.windup_timesteps = windup_timesteps
        self.max_states = max_states
        self.max_iter = max_iter
        self.verbose = verbose
        self.optimization_method = optimization_method
        self.seed = seed
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.index = system_data.index
        self.n_samples = len(system_data)

        if hasattr(self.index, "dtype") and np.issubdtype(
            self.index.dtype, np.datetime64  # type: ignore[arg-type]
        ):
            self.dt = float((self.index[1] - self.index[0]) / np.timedelta64(1, "s"))
        elif hasattr(self.index, "dtype") and hasattr(
            self.index[1] - self.index[0], "total_seconds"
        ):
            self.dt = float((self.index[1] - self.index[0]).total_seconds())
        else:
            self.dt = (
                float(self.index[1] - self.index[0]) if self.n_samples > 1 else 1.0
            )
        self.t_vec = np.arange(0, self.n_samples) * self.dt

    def _build_lti(self, params: np.ndarray, n_states: int):
        a = params[:n_states]
        params[n_states : 2 * n_states]
        params[2 * n_states]

        A = np.zeros((n_states, n_states))
        A[-1, :] = -np.array(a)
        for i in range(n_states - 1):
            A[i, i + 1] = 1.0

        B = np.zeros((n_states, 1))
        B[-1, 0] = 1.0

        C = np.array([params[n_states : 2 * n_states]])
        D = np.array([[params[2 * n_states]]])

        return A, B, C, D

    def _simulate_with_divergence_handling(
        self, A, B, C, D, u: np.ndarray
    ) -> np.ndarray | None:
        """Simulate step-by-step with divergence detection."""
        n_steps = len(u)
        n_states = A.shape[0]
        n_outputs = C.shape[0]

        Ad = np.eye(n_states) + A * self.dt
        Bd = B * self.dt

        x0 = np.zeros(n_states)
        x = x0.copy()
        y_sim = np.zeros((n_steps, n_outputs))
        y_sim[0] = (C @ x0 + D @ u[0]).flatten()

        divergence_threshold = 1e10

        for i in range(1, n_steps):
            x = Ad @ x + Bd @ u[i]
            y = C @ x + D @ u[i]
            y_sim[i] = y.flatten()

            if np.any(np.abs(x) > divergence_threshold) or not np.all(np.isfinite(x)):
                logger.warning(f"Divergence detected at step {i}, stopping simulation")
                return y_sim[: i + 1]

        return y_sim

    def _objective_single_output(
        self,
        params: np.ndarray,
        y_true: np.ndarray,
        u: np.ndarray,
        n_states: int,
    ) -> float:
        A, B, C, D = self._build_lti(params, n_states)

        eigvals = np.linalg.eigvals(A)
        penalty = 0.0
        penalty += np.sum(np.maximum(np.real(eigvals) - 50.0, 0.0) ** 2) * 1e6
        penalty += np.sum(np.maximum(0.1 - np.real(eigvals), 0.0) ** 2) * 1e6

        try:
            y_sim = self._simulate_with_divergence_handling(A, B, C, D, u)
            if y_sim is None or len(y_sim) == 0:
                return 1e6

            y_sim = y_sim.flatten()
            y_eval = y_true[: len(y_sim)]
            nse = 1.0 - np.sum((y_eval - y_sim) ** 2) / np.sum(
                (y_eval - np.mean(y_eval)) ** 2
            )
            if np.isnan(nse):
                nse = -1e6
            return float(-nse + penalty)  # type: ignore[no-any-return]
        except Exception:
            return 1e6

    def _train_single_output(self, output_col: str) -> dict[str, Any]:
        n_states = self.max_states
        y_true = np.asarray(
            self.system_data[output_col].values[self.windup_timesteps :], dtype=float
        )
        u = np.asarray(
            self.system_data[self.independent_columns].values[self.windup_timesteps :],
            dtype=float,
        )

        n_params = 2 * n_states + 1
        bounds = np.array([[-50.0, 50.0]] * n_params)

        init = np.zeros(n_params)
        for i in range(n_states):
            init[i] = -0.5 * (0.5**i)
        init[n_states] = 1.0
        init[-1] = 0.0

        import scipy.optimize as opt

        def objective(params):
            return self._objective_single_output(params, y_true, u, n_states)

        result = opt.differential_evolution(
            objective,
            bounds=bounds,
            maxiter=self.max_iter,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42 if self.seed is None else self.seed,
            updating="deferred",
        )

        best_params = result.x
        A, B, C, D = self._build_lti(best_params, n_states)

        try:
            y_sim = self._simulate_with_divergence_handling(A, B, C, D, u)
            if y_sim is None or len(y_sim) == 0:
                y_sim = np.zeros_like(y_true)
                nse = -1.0
                r2 = -1.0
            else:
                y_sim = y_sim.flatten()
                y_eval = y_true[: len(y_sim)]
                nse = 1.0 - np.sum((y_eval - y_sim) ** 2) / np.sum(
                    (y_eval - np.mean(y_eval)) ** 2
                )
                r2 = nse
                if len(y_sim) < len(y_true):
                    padded = np.full_like(y_true, np.nan)
                    padded[: len(y_sim)] = y_sim
                    y_sim = padded
        except Exception:
            y_sim = np.zeros_like(y_true)
            nse = -1.0
            r2 = -1.0

        return {
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "params": best_params,
            "nse": nse,
            "r2": r2,
            "simulated": y_sim,
            "response": self.system_data[[output_col]],
            "forcing": self.system_data[self.independent_columns],
            "index": self.index,
            "diverged": False,
        }

    def train(self) -> dict[int, dict[str, Any]]:
        results = {}
        for idx, output_col in enumerate(self.dependent_columns, start=1):
            if _normalize_verbose(self.verbose) != "warnings":
                logger.info("Training decoupled LTI for output: %s", output_col)
            results[idx] = self._train_single_output(output_col)
        return results


def decoupled_lti_train(
    system_data,
    dependent_columns,
    independent_columns,
    windup_timesteps=0,
    max_states=5,
    max_iter=250,
    verbose: Verbosity = "warnings",
    optimization_method="differential_evolution",
    seed=None,
    **optimizer_kwargs,
):
    """Train a direct LTI model bypassing the delay-model architecture.

    Directly optimizes A, B, C, D matrices for each output using
    controllable canonical form. Uses NSE (full system simulation accuracy)
    as objective instead of SINDy R².

    Args:
        system_data: DataFrame with time index, inputs, and outputs.
        dependent_columns: Output column names.
        independent_columns: Input column names.
        windup_timesteps: Number of initial timesteps to discard.
        max_states: State dimension for each output's LTI system.
        max_iter: Maximum optimization iterations.
        verbose: Verbosity level.
        optimization_method: scipy.optimize method name.
        seed: Random seed for reproducibility.

    Returns:
        dict keyed by output index (1-based), each containing:
            - A, B, C, D matrices
            - nse, r2 scores
            - simulated output
    """
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)

    trainer = DecoupledLTITrainer(
        system_data=system_data,
        dependent_columns=dependent_columns,
        independent_columns=independent_columns,
        windup_timesteps=windup_timesteps,
        max_states=max_states,
        max_iter=max_iter,
        verbose=verbose,
        optimization_method=optimization_method,
        seed=seed,
        optimizer_kwargs=optimizer_kwargs,
    )
    return trainer.train()
