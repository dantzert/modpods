from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from ._logging import Verbosity, _normalize_verbose, configure_verbosity
from ._system_id import SystemIdModel, _polynomial_feature_names
from .kernels import ConvolutionKernel, get_kernel
from .metrics import compute_detailed_metrics
from .transforms import transform_inputs

logger = logging.getLogger(__name__)


def _build_constraint_matrices(
    feature_names: list[str],
    forcing_coef_constraints: dict[str, Any] | None,
    constraints: list[dict[str, Any]] | None,
    n_targets: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Build constraint matrices for SINDy optimization.

    Args:
        feature_names: List of feature names.
        forcing_coef_constraints: Dict mapping forcing names to constraint specs.
        constraints: List of custom constraint dicts.
        n_targets: Number of target variables.

    Returns:
        Tuple of (constraint_lhs, constraint_rhs, all_inequality).
    """
    n_features = len(feature_names)
    constraint_rows: list[np.ndarray] = []
    constraint_rhs_values: list[float] = []
    all_inequality = True

    if forcing_coef_constraints is not None:
        for key, value in forcing_coef_constraints.items():
            row = np.zeros(n_targets * n_features)
            if isinstance(value, dict):
                lhs = float(value.get("lhs", -1))
                rhs = float(value.get("rhs", 0))
                inequality = value.get("inequality", True)
            else:
                lhs = -float(value)
                rhs = 0.0
                inequality = True
            for i, col in enumerate(feature_names):
                if key in col:
                    row[i] = lhs
            constraint_rows.append(row)
            constraint_rhs_values.append(rhs)
            all_inequality = all_inequality and inequality

    if constraints is not None:
        for constraint in constraints:
            row = np.zeros(n_targets * n_features)
            features = constraint["features"]
            coefficients = constraint["coefficients"]
            rhs = float(constraint.get("rhs", 0))
            inequality = constraint.get("inequality", True)
            for feature, coeff in zip(features, coefficients):
                for i, col in enumerate(feature_names):
                    if col == feature:
                        row[i] = float(coeff)
            constraint_rows.append(row)
            constraint_rhs_values.append(rhs)
            all_inequality = all_inequality and inequality

    if not constraint_rows:
        return np.zeros((0, n_targets * n_features)), np.zeros((0,)), True

    constraint_lhs = np.vstack(constraint_rows)
    constraint_rhs = np.array(constraint_rhs_values)
    return constraint_lhs, constraint_rhs, all_inequality


class SINDYBuilder(ABC):
    """Abstract base class for system-identification model builders."""

    @abstractmethod
    def build(
        self,
        feature_names: list[str],
        poly_degree: int,
        include_bias: bool,
        include_interaction: bool,
    ) -> SystemIdModel:
        """Build an unfitted model.

        Args:
            feature_names: Names for the feature columns.
            poly_degree: Polynomial degree for the feature library.
            include_bias: Whether to include a bias term.
            include_interaction: Whether to include interaction terms.

        Returns:
            An unfitted model instance.
        """
        ...


class StandardSINDYBuilder(SINDYBuilder):
    """Build a standard model with ordinary least squares."""

    def build(
        self,
        feature_names: list[str],
        poly_degree: int,
        include_bias: bool,
        include_interaction: bool,
    ) -> SystemIdModel:
        return SystemIdModel(
            poly_degree=poly_degree,
            include_bias=include_bias,
            include_interaction=include_interaction,
        )


class ConstrainedSINDYBuilder(SINDYBuilder):
    """Build a model with constrained least squares."""

    def __init__(
        self,
        constraint_lhs: np.ndarray,
        constraint_rhs: np.ndarray,
        inequality_constraints: bool,
    ) -> None:
        self.constraint_lhs = constraint_lhs
        self.constraint_rhs = constraint_rhs
        self.inequality_constraints = inequality_constraints

    def build(
        self,
        feature_names: list[str],
        poly_degree: int,
        include_bias: bool,
        include_interaction: bool,
    ) -> SystemIdModel:
        return SystemIdModel(
            poly_degree=poly_degree,
            include_bias=include_bias,
            include_interaction=include_interaction,
            constraint_lhs=self.constraint_lhs,
            constraint_rhs=self.constraint_rhs,
            inequality_constraints=self.inequality_constraints,
        )


class SINDYModelFactory:
    """Factory for training SINDy delay-IO models."""

    def __init__(
        self,
        kernel: ConvolutionKernel,
        kernel_params,
        index,
        forcing: pd.DataFrame,
        response: pd.DataFrame,
        poly_degree: int,
        include_bias: bool,
        include_interaction: bool,
        windup_timesteps: int,
        bibo_stable: bool = False,
        transform_dependent: bool = False,
        transform_only: list[str] | None = None,
        forcing_coef_constraints: Any = None,
        constraints: list[dict[str, Any]] | None = None,
    ) -> None:
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.index = index
        self.forcing = forcing
        self.response = response
        self.poly_degree = poly_degree
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.windup_timesteps = windup_timesteps
        self.bibo_stable = bibo_stable
        self.transform_dependent = transform_dependent
        self.transform_only = transform_only
        self.forcing_coef_constraints = forcing_coef_constraints
        self.constraints = constraints

    def _transform_forcing(self) -> pd.DataFrame:
        """Apply kernel convolution transformations to forcing inputs."""
        if self.transform_only is not None:
            transformed_forcing = transform_inputs(
                self.kernel,
                self.kernel_params,
                self.index,
                self.forcing.loc[:, self.transform_only],
            )
            transformed_forcing = transformed_forcing.drop(columns=self.transform_only)
            untransformed_forcing = self.forcing.drop(columns=self.transform_only)
            return pd.concat(  # type: ignore[no-any-return]
                (untransformed_forcing, transformed_forcing), axis="columns"
            )
        return transform_inputs(  # type: ignore[no-any-return]
            self.kernel,
            self.kernel_params,
            self.index,
            self.forcing,
        )

    def _build_constraint_matrices(
        self, feature_names: list[str], n_targets: int
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        return _build_constraint_matrices(
            feature_names,
            self.forcing_coef_constraints,
            self.constraints,
            n_targets,
        )

    def _create_model_and_feature_names(
        self, forcing: pd.DataFrame
    ) -> tuple[SystemIdModel, list[str], pd.DataFrame]:
        """Create the model and determine feature names for fitting."""
        if self.transform_dependent:
            return self._build_transform_dependent_model(forcing)

        feature_names = self.response.columns.tolist() + forcing.columns.tolist()

        if self.bibo_stable or self.forcing_coef_constraints or self.constraints:
            poly_feature_names = _polynomial_feature_names(
                feature_names,
                self.poly_degree,
                self.include_bias,
                self.include_interaction,
            )
            n_targets = len(self.response.columns)
            custom_lhs, custom_rhs, custom_inequality = self._build_constraint_matrices(
                poly_feature_names, n_targets
            )
            if custom_lhs.shape[0] > 0:
                constraint_rhs = np.zeros((n_targets + custom_lhs.shape[0],))
                constraint_lhs = np.zeros(
                    (
                        n_targets + custom_lhs.shape[0],
                        n_targets * len(poly_feature_names),
                    )
                )
                for j in range(n_targets):
                    constraint_lhs[
                        j,
                        j * len(poly_feature_names)
                        + (j + 1) * len(poly_feature_names)
                        - n_targets
                        + j,
                    ] = 1
                constraint_lhs = np.vstack([constraint_lhs, custom_lhs])
                constraint_rhs = np.concatenate([constraint_rhs, custom_rhs])
                all_inequality = custom_inequality
            else:
                constraint_rhs = np.zeros((n_targets, 1))
                constraint_lhs = np.zeros((n_targets, len(poly_feature_names)))
                constraint_lhs[
                    :,
                    -len(forcing.columns)
                    - len(self.response.columns) : -len(forcing.columns),
                ] = 1
                all_inequality = True

            builder = ConstrainedSINDYBuilder(
                constraint_lhs, constraint_rhs, all_inequality
            )
            model = builder.build(
                poly_feature_names,
                self.poly_degree,
                self.include_bias,
                self.include_interaction,
            )
            return model, poly_feature_names, forcing

        std_builder = StandardSINDYBuilder()
        model = std_builder.build(
            feature_names,
            self.poly_degree,
            self.include_bias,
            self.include_interaction,
        )
        return model, feature_names, forcing

    def _build_transform_dependent_model(
        self, forcing: pd.DataFrame
    ) -> tuple[SystemIdModel, list[str], pd.DataFrame]:
        """Build model for transform_dependent mode."""
        total_train = pd.concat((self.response, forcing), axis="columns")
        total_train = transform_inputs(
            self.kernel,
            self.kernel_params,
            self.index,
            total_train,
        )
        total_train = total_train.drop(columns=self.response.columns)
        feature_names = self.response.columns.tolist() + total_train.columns.tolist()

        n_targets = self.response.shape[1]
        poly_feature_names = _polynomial_feature_names(
            feature_names,
            self.poly_degree,
            self.include_bias,
            self.include_interaction,
        )
        n_features = len(poly_feature_names)

        constraint_rhs = np.zeros((n_targets,))
        constraint_lhs = np.zeros((n_targets, n_features * n_targets))
        if self.bibo_stable:
            initial_guess = np.zeros((n_targets, n_features))
            for idx in range(n_targets):
                initial_guess[idx, idx] = -1
        else:
            initial_guess = None

        for idx in range(n_targets):
            constraint_lhs[idx, (idx + 1) * n_features - n_targets + idx] = 1

        model = SystemIdModel(
            poly_degree=self.poly_degree,
            include_bias=self.include_bias,
            include_interaction=self.include_interaction,
            constraint_lhs=constraint_lhs,
            constraint_rhs=constraint_rhs,
            inequality_constraints=False,
            initial_guess=initial_guess,
        )
        return model, feature_names, total_train

    def _fit_and_score(
        self,
        model: SystemIdModel,
        forcing: pd.DataFrame,
        feature_names: list[str],
    ) -> tuple[float, Exception | None]:
        """Fit the model and compute R² score."""
        try:
            model.fit(
                self.response.values[self.windup_timesteps :, :],
                t=np.arange(0, len(self.index), 1)[self.windup_timesteps :],
                u=forcing.values[self.windup_timesteps :, :],
                feature_names=feature_names,
            )
            r2 = model.score(
                self.response.values[self.windup_timesteps :, :],
                t=np.arange(0, len(self.index), 1)[self.windup_timesteps :],
                u=forcing.values[self.windup_timesteps :, :],
            )
            return r2, None
        except Exception as e:
            logger.warning("Exception in model fitting, returning r2=-1")
            logger.warning("%s", e)
            return -1.0, e

    def _error_result(
        self, model: SystemIdModel | None, r2: float = -1.0
    ) -> dict[str, Any]:
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
        return {
            "error_metrics": error_metrics,
            "model": model,
            "simulated": False,
            "response": self.response,
            "forcing": self.forcing,
            "index": self.index,
            "diverged": False,
        }

    def train(self, final_run: bool = False) -> dict[str, Any]:
        """Train the SINDy model.

        Args:
            final_run: If True, simulate and compute detailed metrics.

        Returns:
            Dict with keys: error_metrics, model, simulated, response,
            forcing, index, diverged.
        """
        forcing = self._transform_forcing()
        model, feature_names, fit_forcing = self._create_model_and_feature_names(
            forcing
        )

        if self.transform_dependent:
            try:
                model.fit(
                    self.response.values[self.windup_timesteps :, :],
                    t=np.arange(0, len(self.index), 1)[self.windup_timesteps :],
                    u=fit_forcing.values[self.windup_timesteps :, :],
                    feature_names=feature_names,
                )
                r2 = model.score(
                    self.response.values[self.windup_timesteps :, :],
                    t=np.arange(0, len(self.index), 1)[self.windup_timesteps :],
                    u=fit_forcing.values[self.windup_timesteps :, :],
                )
            except Exception as e:
                logger.warning("Exception in model fitting, returning r2=-1")
                logger.warning("%s", e)
                return self._error_result(model, r2=-1)
        else:
            r2, err = self._fit_and_score(model, fit_forcing, feature_names)
            if err is not None:
                return self._error_result(model, r2=-1)

        if not final_run:
            return {
                "error_metrics": {"r2": r2},
                "model": model,
                "simulated": False,
                "response": self.response,
                "forcing": forcing,
                "index": self.index,
                "diverged": False,
            }

        simulated: Any = False
        try:
            if self.transform_dependent:
                simulated = model.simulate(
                    self.response.values[self.windup_timesteps, :],
                    t=np.arange(0, len(self.index), 1)[self.windup_timesteps :],
                    u=fit_forcing.values[self.windup_timesteps :, :],
                )
            else:
                simulated = model.simulate(
                    self.response.values[self.windup_timesteps, :],
                    t=np.arange(0, len(self.index), 1)[self.windup_timesteps :],
                    u=fit_forcing.values[self.windup_timesteps :, :],
                )
            error_metrics = compute_detailed_metrics(
                self.response.values[self.windup_timesteps + 1 :, :],
                simulated,
                self.index,
                self.windup_timesteps,
            )
            error_metrics["r2"] = r2
        except Exception as e:
            logger.warning("Exception in simulation")
            logger.warning("%s", e)
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
                "simulated": self.response[1:],
                "response": self.response,
                "forcing": forcing,
                "index": self.index,
                "diverged": True,
            }

        return {
            "error_metrics": error_metrics,
            "model": model,
            "simulated": simulated,
            "response": self.response,
            "forcing": forcing,
            "index": self.index,
            "diverged": False,
        }


def SINDY_delays_MI(
    kernel: ConvolutionKernel | str,
    kernel_params,
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
    constraints=None,
    transform_cache=None,
    verbose: Verbosity = "warnings",
):
    """Train a SINDy delay-IO model.

    .. deprecated::
        Use :class:`SINDYModelFactory` for new code. This function is preserved
        for backward compatibility.
    """
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)

    kernel = get_kernel(kernel)
    factory = SINDYModelFactory(
        kernel=kernel,
        kernel_params=kernel_params,
        index=index,
        forcing=forcing,
        response=response,
        poly_degree=poly_degree,
        include_bias=include_bias,
        include_interaction=include_interaction,
        windup_timesteps=windup_timesteps,
        bibo_stable=bibo_stable,
        transform_dependent=transform_dependent,
        transform_only=transform_only,
        forcing_coef_constraints=forcing_coef_constraints,
        constraints=constraints,
    )
    return factory.train(final_run=final_run)
