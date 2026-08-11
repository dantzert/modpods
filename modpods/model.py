import logging

import numpy as np
import pandas as pd
import pysindy as ps
from pysindy.optimizers._constrained_sr3 import ConstrainedSR3 as _ConstrainedSR3

from ._logging import _normalize_verbose, configure_verbosity, Verbosity
from .metrics import compute_basic_metrics
from .transforms import transform_inputs

logger = logging.getLogger(__name__)


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
    verbose: Verbosity = "warnings",
):
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)
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

            # constrain the highest order output autocorrelation to be negative
            # this indexing is only right for include_interaction=False, include_bias=False, and pure polynomial library
            # for more complex libraries, some conditional logic will be needed to grab the right column
            # constraint_lhs[:n_targets,-len(forcing.columns)-len(response.columns):-len(forcing.columns)] = 1

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
            # alpha nse decomposition due to gupta et al 2009
            logger.info("alpha = %s", alpha)
            logger.info("beta = %s", beta)
            # top 2% peak flow bias (HFV) due to yilmaz et al 2008
            logger.info("HFV = %s", hfv)
            # top 10% peak flow bias (HFV) due to yilmaz et al 2008
            logger.info("HFV10 = %s", hfv10)
            # 30% low flow bias (LFV) due to yilmaz et al 2008
            logger.info("LFV = %s", lfv)
            # bias of FDC midsegment slope due to yilmaz et al 2008
            logger.info("FDC = %s", fdc)
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
    # return [r2, model, mae, rmse, index, simulated , response , forcing]
