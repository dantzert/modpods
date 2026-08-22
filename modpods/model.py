import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import pysindy as ps  # type: ignore
from pysindy.optimizers._constrained_sr3 import (  # type: ignore[import-untyped]
    ConstrainedSR3 as _ConstrainedSR3,
)

from ._logging import Verbosity, _normalize_verbose, configure_verbosity
from .kernels import ConvolutionKernel, get_kernel
from .metrics import compute_basic_metrics, compute_detailed_metrics
from .transforms import transform_inputs

logger = logging.getLogger(__name__)


class SINDYBuilder(ABC):
    """Abstract base class for SINDy model builders."""

    @abstractmethod
    def build(
        self,
        feature_names: list[str],
        poly_degree: int,
        include_bias: bool,
        include_interaction: bool,
    ) -> ps.SINDy:
        """Build an unfitted SINDy model.

        Args:
            feature_names: Names for the feature columns.
            poly_degree: Polynomial degree for the feature library.
            include_bias: Whether to include a bias term.
            include_interaction: Whether to include interaction terms.

        Returns:
            An unfitted SINDy model instance.
        """
        ...


class StandardSINDYBuilder(SINDYBuilder):
    """Build a standard SINDy model with STLSQ optimizer."""

    def build(
        self,
        feature_names: list[str],
        poly_degree: int,
        include_bias: bool,
        include_interaction: bool,
    ) -> ps.SINDy:
        return ps.SINDy(
            differentiation_method=ps.FiniteDifference(),
            feature_library=ps.PolynomialLibrary(
                degree=poly_degree,
                include_bias=include_bias,
                include_interaction=include_interaction,
            ),
            optimizer=ps.STLSQ(threshold=0),
        )


class ConstrainedSINDYBuilder(SINDYBuilder):
    """Build a SINDy model with constrained SR3 optimizer."""

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
    ) -> ps.SINDy:
        return ps.SINDy(
            differentiation_method=ps.FiniteDifference(),
            feature_library=ps.PolynomialLibrary(
                degree=poly_degree,
                include_bias=include_bias,
                include_interaction=include_interaction,
            ),
            optimizer=_ConstrainedSR3(
                reg_weight_lam=0,
                regularizer="l2",
                constraint_lhs=self.constraint_lhs,
                constraint_rhs=self.constraint_rhs,
                inequality_constraints=self.inequality_constraints,
            ),
        )


class TransformDependentSINDYBuilder(SINDYBuilder):
    """Build a SINDy model for transform_dependent mode."""

    def __init__(self, bibo_stable: bool = False) -> None:
        self.bibo_stable = bibo_stable

    def build(
        self,
        feature_names: list[str],
        poly_degree: int,
        include_bias: bool,
        include_interaction: bool,
    ) -> ps.SINDy:
        library = ps.PolynomialLibrary(
            degree=poly_degree,
            include_bias=include_bias,
            include_interaction=include_interaction,
        )
        n_targets = len(feature_names) - len(
            [f for f in feature_names if f in ["MAE", "RMSE", "NSE", "alpha", "beta", "HFV", "HFV10", "LFV", "FDC"]]
        )
        # The n_targets will be set properly by the factory
        # This builder needs additional context from the factory
        return ps.SINDy(  # type: ignore[no-any-return]
            differentiation_method=ps.FiniteDifference(),
            feature_library=library,
            optimizer=_ConstrainedSR3(
                reg_weight_lam=0,
                regularizer="l0",
                relax_coeff_nu=10e9,
                constraint_lhs=np.zeros((0, 1)),
                constraint_rhs=np.zeros((0,)),
                inequality_constraints=False,
                max_iter=10000,
            ),
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
            return pd.concat(
                (untransformed_forcing, transformed_forcing), axis="columns"
            )
        return transform_inputs(
            self.kernel,
            self.kernel_params,
            self.index,
            self.forcing,
        )

    def _build_constraint_matrices(
        self, feature_names: list[str], n_targets: int
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        n_features = len(feature_names)
        constraint_rows: list[np.ndarray] = []
        constraint_rhs_values: list[float] = []
        all_inequality = True

        if self.forcing_coef_constraints is not None:
            for key, value in self.forcing_coef_constraints.items():
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

        if self.constraints is not None:
            for constraint in self.constraints:
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
            return (
                np.zeros((0, n_targets * n_features)),
                np.zeros((0,)),
                True,
            )

        constraint_lhs = np.vstack(constraint_rows)
        constraint_rhs = np.array(constraint_rhs_values)
        return constraint_lhs, constraint_rhs, all_inequality

    def _create_builder(self) -> SINDYBuilder:
        if self.transform_dependent:
            return TransformDependentSINDYBuilder(bibo_stable=self.bibo_stable)
        if self.bibo_stable or self.forcing_coef_constraints or self.constraints:
            feature_names = self.response.columns.tolist() + self.forcing.columns.tolist()
            n_targets = len(self.response.columns)
            custom_lhs, custom_rhs, custom_inequality = self._build_constraint_matrices(
                feature_names, n_targets
            )
            if custom_lhs.shape[0] > 0:
                constraint_rhs = np.zeros((n_targets + custom_lhs.shape[0],))
                constraint_lhs = np.zeros(
                    (n_targets + custom_lhs.shape[0], n_targets * len(feature_names))
                )
                for j in range(n_targets):
                    constraint_lhs[
                        j, j * len(feature_names) + (j + 1) * len(feature_names) - n_targets + j
                    ] = 1
                constraint_lhs = np.vstack([constraint_lhs, custom_lhs])
                constraint_rhs = np.concatenate([constraint_rhs, custom_rhs])
                all_inequality = custom_inequality
            else:
                constraint_rhs = np.zeros((n_targets, 1))
                constraint_lhs = np.zeros((n_targets, len(feature_names)))
                constraint_lhs[
                    :,
                    -len(self.forcing.columns)
                    - len(self.response.columns) : -len(self.forcing.columns),
                ] = 1
                all_inequality = True
            return ConstrainedSINDYBuilder(constraint_lhs, constraint_rhs, all_inequality)
        return StandardSINDYBuilder()

    def _build_transform_dependent_model(
        self, forcing: pd.DataFrame
    ) -> tuple[ps.SINDy, pd.DataFrame]:
        total_train = pd.concat((self.response, forcing), axis="columns")
        total_train = transform_inputs(
            self.kernel,
            self.kernel_params,
            self.index,
            total_train,
        )
        total_train = total_train.drop(columns=self.response.columns)
        feature_names = self.response.columns.tolist() + total_train.columns.tolist()

        library = ps.PolynomialLibrary(
            degree=self.poly_degree,
            include_bias=self.include_bias,
            include_interaction=self.include_interaction,
        )
        library_terms = pd.concat((total_train, self.response), axis="columns")
        library.fit([ps.AxesArray(library_terms, {"ax_sample": 0, "ax_coord": 1})])
        n_features = library.n_output_features_
        n_targets = self.response.shape[1]

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
        return model, total_train

    def _fit_and_score(
        self,
        model: ps.SINDy,
        forcing: pd.DataFrame,
    ) -> tuple[float, Any]:
        try:
            model.fit(
                self.response.values[self.windup_timesteps:, :],
                t=np.arange(0, len(self.index), 1)[self.windup_timesteps:],
                u=forcing.values[self.windup_timesteps:, :],
                feature_names=self.response.columns.tolist() + forcing.columns.tolist(),
            )
            r2 = model.score(
                self.response.values[self.windup_timesteps:, :],
                t=np.arange(0, len(self.index), 1)[self.windup_timesteps:],
                u=forcing.values[self.windup_timesteps:, :],
            )
            return r2, None
        except Exception as e:
            logger.warning("Exception in model fitting, returning r2=-1")
            logger.warning("%s", e)
            return -1.0, e

    def _error_result(self, model: ps.SINDy | None, r2: float = -1.0) -> dict[str, Any]:
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
            Dict with keys: error_metrics, model, simulated, response, forcing, index, diverged.
        """
        forcing = self._transform_forcing()
        feature_names = self.response.columns.tolist() + forcing.columns.tolist()

        if self.transform_dependent:
            model, total_train = self._build_transform_dependent_model(forcing)
            try:
                model.fit(
                    self.response.values[self.windup_timesteps:, :],
                    t=np.arange(0, len(self.index), 1)[self.windup_timesteps:],
                    u=total_train.values[self.windup_timesteps:, :],
                    feature_names=feature_names,
                )
                r2 = model.score(
                    self.response.values[self.windup_timesteps:, :],
                    t=np.arange(0, len(self.index), 1)[self.windup_timesteps:],
                    u=total_train.values[self.windup_timesteps:, :],
                )
            except Exception as e:
                logger.warning("Exception in model fitting, returning r2=-1")
                logger.warning("%s", e)
                return self._error_result(model, r2=-1)
        else:
            builder = self._create_builder()
            model = builder.build(
                feature_names=feature_names,
                poly_degree=self.poly_degree,
                include_bias=self.include_bias,
                include_interaction=self.include_interaction,
            )
            r2, err = self._fit_and_score(model, forcing)
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

        simulated = False
        try:
            if self.transform_dependent:
                simulated = model.simulate(
                    self.response.values[self.windup_timesteps, :],
                    t=np.arange(0, len(self.index), 1)[self.windup_timesteps:],
                    u=total_train.values[self.windup_timesteps:, :],
                )
            else:
                simulated = model.simulate(
                    self.response.values[self.windup_timesteps, :],
                    t=np.arange(0, len(self.index), 1)[self.windup_timesteps:],
                    u=forcing.values[self.windup_timesteps:, :],
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
