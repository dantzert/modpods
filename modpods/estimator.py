from __future__ import annotations

from typing import Any

import pandas as pd

from ._logging import Verbosity
from ._validation import validate_columns, validate_system_data


class DelayIOModel:
    """A single fitted delay-io model for a given number of transforms."""

    def __init__(
        self,
        n_transforms: int,
        kernel_type: str,
        final_model: dict[str, Any],
        kernel_params: pd.DataFrame,
        windup_timesteps: int,
        dependent_columns: list[str],
        independent_columns: list[str],
        transform_cache: Any,
    ) -> None:
        self.n_transforms_ = n_transforms
        self.kernel_type_ = kernel_type
        self.final_model_ = final_model
        self.kernel_params_ = kernel_params
        self.windup_timesteps_ = windup_timesteps
        self.dependent_columns_ = dependent_columns
        self.independent_columns_ = independent_columns
        self.transform_cache_ = transform_cache
        self.kernel_name_: str | None = None

    @classmethod
    def from_dict(cls, n_transforms: int, entry: dict[str, Any]) -> DelayIOModel:
        return cls(
            n_transforms=n_transforms,
            kernel_type=entry["kernel_type"],
            final_model=entry["final_model"],
            kernel_params=entry["kernel_params"],
            windup_timesteps=entry["windup_timesteps"],
            dependent_columns=entry["dependent_columns"],
            independent_columns=entry["independent_columns"],
            transform_cache=entry["transform_cache"],
        )

    def predict(
        self,
        system_data: pd.DataFrame,
        evaluation: bool = False,
        windup_timesteps: int | None = None,
        verbose: Verbosity = "warnings",
    ) -> dict[str, Any]:
        from .predict import delay_io_predict

        old_format = {
            self.n_transforms_: {
                "final_model": self.final_model_,
                "kernel_type": self.kernel_type_,
                "kernel_params": self.kernel_params_,
                "windup_timesteps": self.windup_timesteps_,
                "dependent_columns": self.dependent_columns_,
                "independent_columns": self.independent_columns_,
                "transform_cache": self.transform_cache_,
            }
        }
        return delay_io_predict(  # type: ignore[no-any-return]
            old_format,
            system_data,
            num_transforms=self.n_transforms_,
            evaluation=evaluation,
            windup_timesteps=windup_timesteps,
            verbose=verbose,
        )

    @property
    def error_metrics_(self) -> dict[str, Any]:
        return self.final_model_["error_metrics"]  # type: ignore[no-any-return]

    @property
    def r2_(self) -> float:
        return float(self.final_model_["error_metrics"]["r2"])

    def __repr__(self) -> str:
        return f"DelayIOModel(n_transforms={self.n_transforms_}, " f"r2={self.r2_:.4f})"


class DelayIO:
    """Delay-IO estimator following scikit-learn conventions."""

    def __init__(
        self,
        dependent_columns: list[str],
        independent_columns: list[str],
        windup_timesteps: int = 0,
        init_transforms: int = 1,
        max_transforms: int = 4,
        max_iter: int = 250,
        poly_order: int = 3,
        transform_dependent: bool = False,
        transform_only: list[str] | None = None,
        verbose: Verbosity = "warnings",
        include_bias: bool = False,
        include_interaction: bool = False,
        bibo_stable: bool = False,
        forcing_coef_constraints: Any = None,
        constraints: Any = None,
        early_stopping_threshold: float = 0.005,
        optimization_method: str = "bayesian",
        kernel: str | Any = "gamma",
        random_state: int | None = None,
    ) -> None:
        self.dependent_columns = dependent_columns
        self.independent_columns = independent_columns
        self.windup_timesteps = windup_timesteps
        self.init_transforms = init_transforms
        self.max_transforms = max_transforms
        self.max_iter = max_iter
        self.poly_order = poly_order
        self.transform_dependent = transform_dependent
        self.transform_only = transform_only
        self.verbose = verbose
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.bibo_stable = bibo_stable
        self.forcing_coef_constraints = forcing_coef_constraints
        self.constraints = constraints
        self.early_stopping_threshold = early_stopping_threshold
        self.optimization_method = optimization_method
        self.kernel = kernel
        self.random_state = random_state
        self.estimators_: list[DelayIOModel] = []

    def fit(self, system_data: pd.DataFrame, **kwargs: Any) -> list[DelayIOModel]:
        validate_system_data(system_data)
        validate_columns(system_data, self.dependent_columns, "dependent_columns")
        validate_columns(system_data, self.independent_columns, "independent_columns")

        from .train import delay_io_train

        results = delay_io_train(
            system_data=system_data,
            dependent_columns=self.dependent_columns,
            independent_columns=self.independent_columns,
            windup_timesteps=self.windup_timesteps,
            init_transforms=self.init_transforms,
            max_transforms=self.max_transforms,
            max_iter=self.max_iter,
            poly_order=self.poly_order,
            transform_dependent=self.transform_dependent,
            transform_only=self.transform_only,
            verbose=self.verbose,
            include_bias=self.include_bias,
            include_interaction=self.include_interaction,
            bibo_stable=self.bibo_stable,
            forcing_coef_constraints=self.forcing_coef_constraints,
            constraints=self.constraints,
            early_stopping_threshold=self.early_stopping_threshold,
            optimization_method=self.optimization_method,
            kernel=self.kernel,
            seed=self.random_state,
            **kwargs,
        )

        estimators: list[DelayIOModel] = []
        first_key = next(iter(results))
        first_val = results[first_key]
        if isinstance(first_val, dict) and "final_model" in first_val:
            for nt, entry in results.items():
                estimators.append(DelayIOModel.from_dict(nt, entry))
        else:
            for kernel_name, kernel_results in results.items():
                for nt, entry in kernel_results.items():
                    model = DelayIOModel.from_dict(nt, entry)
                    model.kernel_name_ = kernel_name
                    estimators.append(model)

        self.estimators_ = estimators
        self.best_estimator_ = self._select_best()
        return self.estimators_

    def predict(
        self,
        system_data: pd.DataFrame,
        n_transforms: int | None = None,
        evaluation: bool = False,
        windup_timesteps: int | None = None,
        verbose: Verbosity = "warnings",
    ) -> dict[str, Any]:
        if not self.estimators_:
            raise RuntimeError("Estimator has not been fitted yet.")
        if n_transforms is None:
            model = self.best_estimator_
        else:
            model = next(
                (e for e in self.estimators_ if e.n_transforms_ == n_transforms),
                None,
            )
            if model is None:
                raise ValueError(
                    f"No model with n_transforms={n_transforms}. "
                    f"Available: {[e.n_transforms_ for e in self.estimators_]}"
                )
        return model.predict(
            system_data,
            evaluation=evaluation,
            windup_timesteps=windup_timesteps,
            verbose=verbose,
        )

    def _select_best(self) -> DelayIOModel:
        return max(self.estimators_, key=lambda e: e.r2_)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "dependent_columns": self.dependent_columns,
            "independent_columns": self.independent_columns,
            "windup_timesteps": self.windup_timesteps,
            "init_transforms": self.init_transforms,
            "max_transforms": self.max_transforms,
            "max_iter": self.max_iter,
            "poly_order": self.poly_order,
            "transform_dependent": self.transform_dependent,
            "transform_only": self.transform_only,
            "verbose": self.verbose,
            "include_bias": self.include_bias,
            "include_interaction": self.include_interaction,
            "bibo_stable": self.bibo_stable,
            "forcing_coef_constraints": self.forcing_coef_constraints,
            "constraints": self.constraints,
            "early_stopping_threshold": self.early_stopping_threshold,
            "optimization_method": self.optimization_method,
            "kernel": self.kernel,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> DelayIO:
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)
        return self
