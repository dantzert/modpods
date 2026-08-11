import logging

import numpy as np

from .metrics import compute_basic_metrics
from .transforms import transform_inputs

logger = logging.getLogger(__name__)


def delay_io_predict(
    delay_io_model,
    system_data,
    num_transforms=1,
    evaluation=False,
    windup_timesteps=None,
    verbose=False,
):
    if verbose:
        logger.setLevel(logging.INFO)
    if (
        windup_timesteps is None
    ):  # user didn't specify windup timesteps, use what the model trained with.
        windup_timesteps = delay_io_model[num_transforms]["windup_timesteps"]
    forcing = system_data[delay_io_model[num_transforms]["independent_columns"]].copy(
        deep=True
    )
    response = system_data[delay_io_model[num_transforms]["dependent_columns"]].copy(
        deep=True
    )

    # Use cache from model if available, otherwise create a new one
    transform_cache = delay_io_model[num_transforms].get("transform_cache", None)
    transformed_forcing = transform_inputs(
        shape_factors=delay_io_model[num_transforms]["shape_factors"],
        scale_factors=delay_io_model[num_transforms]["scale_factors"],
        loc_factors=delay_io_model[num_transforms]["loc_factors"],
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
    except Exception as e:  # and print the exception:
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

    # return all the error metrics if the prediction is being evaluated against known measurements
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
            for col_idx in range(
                0, len(response.columns)
            ):  # univariate performance metrics
                error = (
                    response.values[windup_timesteps + 1 :, col_idx]
                    - prediction[:, col_idx]
                )

                initial_error_length = len(error)
                error = error[~np.isnan(error)]
                if len(error) < 0.75 * initial_error_length:
                    logger.warning(
                        "More than 25%% of the entries in error were NaN"
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
            }
            # omit r2 here because it doesn't mean the same thing as it does for training, would be misleading.
            # r2 in training expresses how much of the derivative is predicted by the model, whereas in evaluation it expresses how much of the response is predicted by the model

            return {
                "prediction": prediction,
                "error_metrics": error_metrics,
                "diverged": False,
            }
        except Exception as e:  # and print the exception:
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
