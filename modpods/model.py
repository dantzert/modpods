import logging

import numpy as np
import pandas as pd
import pysindy as ps  # type: ignore
from pysindy.optimizers._constrained_sr3 import (  # type: ignore[import-untyped]
    ConstrainedSR3 as _ConstrainedSR3,
)

from ._logging import Verbosity, _normalize_verbose, configure_verbosity
from .kernels import ConvolutionKernel, get_kernel
from .metrics import compute_basic_metrics
from .transforms import transform_inputs

logger = logging.getLogger(__name__)


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
    transform_cache=None,
    verbose: Verbosity = "warnings",
):
    kernel = get_kernel(kernel)
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)

    if transform_only is not None:
        transformed_forcing = transform_inputs(
            kernel,
            kernel_params,
            index,
            forcing.loc[:, transform_only],
            cache=transform_cache,
        )
        untransformed_forcing = forcing.drop(columns=transform_only)
        forcing = pd.concat(
            (untransformed_forcing, transformed_forcing), axis="columns"
        )
    else:
        forcing = transform_inputs(
            kernel,
            kernel_params,
            index,
            forcing,
            cache=transform_cache,
        )

    feature_names = response.columns.tolist() + forcing.columns.tolist()

    # SINDy
    if not bibo_stable and forcing_coef_constraints is None:
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
        constraint_rhs = np.zeros((n_features,))
        constraint_lhs = np.zeros((n_features, n_targets * n_features))

        for i, col in enumerate(feature_names):
            for key in forcing_coef_constraints.keys():
                if key in col:
                    constraint_lhs[i, i] = -forcing_coef_constraints[key]

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
    elif bibo_stable:
        library = ps.PolynomialLibrary(
            degree=poly_degree,
            include_bias=include_bias,
            include_interaction=include_interaction,
        )
        total_train = pd.concat((response, forcing), axis="columns")
        library.fit([ps.AxesArray(total_train, {"ax_sample": 0, "ax_coord": 1})])
        n_features = library.n_output_features_
        feature_names = library.get_feature_names(input_features=total_train.columns)
        n_targets = total_train.shape[1]
        constraint_rhs = np.zeros((len(response.columns), 1))
        constraint_lhs = np.zeros((len(response.columns), n_features))

        constraint_lhs[
            :, -len(forcing.columns) - len(response.columns) : -len(forcing.columns)
        ] = 1

        if forcing_coef_constraints is not None:
            n_targets = len(response.columns)
            constraint_rhs = np.zeros((n_features,))
            constraint_lhs = np.zeros((n_features, n_targets * n_features))
            highest_power_col_idx = 0
            for i, col in enumerate(feature_names):
                if response.columns[0] in col:
                    highest_power_col_idx = i
            constraint_lhs[0, highest_power_col_idx] = 1

            for i, col in enumerate(feature_names):
                for key in forcing_coef_constraints.keys():
                    if key in col:
                        constraint_lhs[i, i] = -forcing_coef_constraints[key]

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
        total_train = pd.concat((response, forcing), axis="columns")
        total_train = transform_inputs(
            kernel,
            kernel_params,
            index,
            total_train,
        )
        total_train = total_train.drop(columns=response.columns)
        feature_names = response.columns.tolist() + total_train.columns.tolist()

        library = ps.PolynomialLibrary(
            degree=poly_degree,
            include_bias=include_bias,
            include_interaction=include_interaction,
        )
        library_terms = pd.concat((total_train, response), axis="columns")
        library.fit([ps.AxesArray(library_terms, {"ax_sample": 0, "ax_coord": 1})])
        n_features = library.n_output_features_
        n_targets = response.shape[1]

        constraint_rhs = np.zeros((n_targets,))
        constraint_lhs = np.zeros((n_targets, n_features * n_targets))
        if bibo_stable:
            initial_guess = np.zeros((n_targets, n_features))
            for idx in range(0, n_targets):
                initial_guess[idx, idx] = -1
        else:
            initial_guess = None

        for idx in range(0, n_targets):
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

        try:
            model.fit(
                response.values[windup_timesteps:, :],
                t=np.arange(0, len(index), 1)[windup_timesteps:],
                u=total_train.values[windup_timesteps:, :],
            )
            r2 = model.score(
                response.values[windup_timesteps:, :],
                t=np.arange(0, len(index), 1)[windup_timesteps:],
                u=total_train.values[windup_timesteps:, :],
            )
        except Exception as e:
            logger.warning("Exception in model fitting, returning r2=-1")
            logger.warning("%s", e)
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
            model.fit(
                response.values[windup_timesteps:, :],
                t=np.arange(0, len(index), 1)[windup_timesteps:],
                u=forcing.values[windup_timesteps:, :],
            )
            r2 = model.score(
                response.values[windup_timesteps:, :],
                t=np.arange(0, len(index), 1)[windup_timesteps:],
                u=forcing.values[windup_timesteps:, :],
            )
        except Exception as e:
            logger.warning("Exception in model fitting, returning r2=-1")
            logger.warning("%s", e)
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
    if final_run:
        try:
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
            for col_idx in range(0, len(response.columns)):
                basic = compute_basic_metrics(
                    response.values[windup_timesteps + 1 :, col_idx],
                    simulated[:, col_idx],
                )
                mae.append(basic["mae"])
                rmse.append(basic["rmse"])
                nse.append(basic["nse"])
                alpha.append(basic["alpha"])
                beta.append(basic["beta"])

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

            logger.info("MAE = %s", mae)
            logger.info("RMSE = %s", rmse)
            logger.info("NSE = %s", nse)
            logger.info("alpha = %s", alpha)
            logger.info("beta = %s", beta)
            logger.info("HFV = %s", hfv)
            logger.info("HFV10 = %s", hfv10)
            logger.info("LFV = %s", lfv)
            logger.info("FDC = %s", fdc)
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
