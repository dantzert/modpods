import logging

import numpy as np

from ._logging import Verbosity, _normalize_verbose, configure_verbosity
from .kernels import get_kernel
from .metrics import compute_basic_metrics
from .transforms import transform_inputs

logger = logging.getLogger(__name__)


def delay_io_predict(
    delay_io_model,
    system_data,
    num_transforms=1,
    evaluation=False,
    windup_timesteps=None,
    verbose: Verbosity = "warnings",
):
    if _normalize_verbose(verbose) != "warnings":
        configure_verbosity(verbose)
    if windup_timesteps is None:
        windup_timesteps = delay_io_model[num_transforms]["windup_timesteps"]
    forcing = system_data[delay_io_model[num_transforms]["independent_columns"]].copy(
        deep=True
    )
    response = system_data[delay_io_model[num_transforms]["dependent_columns"]].copy(
        deep=True
    )

    kernel = get_kernel(delay_io_model[num_transforms]["kernel_type"])
    kernel_params = delay_io_model[num_transforms]["kernel_params"]

    transform_cache = delay_io_model[num_transforms].get("transform_cache", None)
    transformed_forcing = transform_inputs(
        kernel,
        kernel_params,
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
    except Exception as e:
        logger.warning("Exception in simulation")
        logger.warning("%s", e)
        logger.warning("diverged.")
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
            for col_idx in range(0, len(response.columns)):
                error = (
                    response.values[windup_timesteps + 1 :, col_idx]
                    - prediction[:, col_idx]
                )

                initial_error_length = len(error)
                error = error[~np.isnan(error)]
                if len(error) < 0.75 * initial_error_length:
                    logger.warning(
                        "WARNING: More than 25%% of the entries in error were NaN"
                    )

                basic = compute_basic_metrics(
                    response.values[windup_timesteps + 1 :, col_idx],
                    prediction[:, col_idx],
                )
                mae.append(basic["mae"])
                rmse.append(basic["rmse"])
                nse.append(basic["nse"])
                alpha.append(basic["alpha"])
                beta.append(basic["beta"])

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
            }

            return {
                "prediction": prediction,
                "error_metrics": error_metrics,
                "diverged": False,
            }
        except Exception as e:
            logger.warning("Exception in simulation")
            logger.warning("%s", e)
            logger.warning("Simulation diverged.")
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
