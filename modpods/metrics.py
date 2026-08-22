import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from ._logging import Verbosity

logger = logging.getLogger(__name__)


def compute_basic_metrics(y_true, y_pred):
    """Compute common error metrics between true and predicted values.

    Args:
        y_true: array of observed values
        y_pred: array of predicted values

    Returns:
        dict with keys: "mae", "rmse", "nse", "alpha", "beta"
    """
    error = y_true - y_pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    nse = float(1 - np.sum(error**2) / np.sum((y_true - np.mean(y_true)) ** 2))
    alpha = float(np.std(y_pred) / np.std(y_true))
    beta = float(np.mean(y_pred) / np.mean(y_true))
    return {
        "mae": mae,
        "rmse": rmse,
        "nse": nse,
        "alpha": alpha,
        "beta": beta,
    }


def compute_detailed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index,
    windup_timesteps: int,
) -> dict[str, Any]:
    """Compute detailed error metrics for multi-output models.

    Computes per-column metrics including MAE, RMSE, NSE, alpha, beta,
    HFV, HFV10, LFV, and FDC.

    Args:
        y_true: Array of observed values, shape (n_timesteps, n_outputs).
        y_pred: Array of predicted values, shape (n_timesteps, n_outputs).
        index: Time index for the full dataset.
        windup_timesteps: Number of initial timesteps skipped during warm-up.

    Returns:
        Dict with keys: MAE, RMSE, NSE, alpha, beta, HFV, HFV10, LFV, FDC.
    """
    n_cols = y_true.shape[1]
    mae = []
    rmse = []
    nse = []
    alpha = []
    beta = []
    hfv = []
    hfv10 = []
    lfv = []
    fdc = []

    for col_idx in range(n_cols):
        basic = compute_basic_metrics(y_true[:, col_idx], y_pred[:, col_idx])
        mae.append(basic["mae"])
        rmse.append(basic["rmse"])
        nse.append(basic["nse"])
        alpha.append(basic["alpha"])
        beta.append(basic["beta"])

        hfv.append(
            100
            * np.sum(
                np.sort(y_pred[:, col_idx])[-int(0.02 * len(index)) :]
                - np.sort(y_true[:, col_idx])[-int(0.02 * len(index)) :]
            )
            / np.sum(np.sort(y_true[:, col_idx])[-int(0.02 * len(index)) :])
        )
        hfv10.append(
            100
            * np.sum(
                np.sort(y_pred[:, col_idx])[-int(0.1 * len(index)) :]
                - np.sort(y_true[:, col_idx])[-int(0.1 * len(index)) :]
            )
            / np.sum(np.sort(y_true[:, col_idx])[-int(0.1 * len(index)) :])
        )
        lfv.append(
            100
            * np.sum(
                np.sort(y_pred[:, col_idx])[-int(0.3 * len(index)) :]
                - np.sort(y_true[:, col_idx])[-int(0.3 * len(index)) :]
            )
            / np.sum(np.sort(y_true[:, col_idx])[-int(0.3 * len(index)) :])
        )
        fdc.append(
            100
            * (
                np.log10(np.sort(y_pred[:, col_idx])[int(0.2 * len(y_pred))])
                - np.log10(np.sort(y_pred[:, col_idx])[int(0.7 * len(y_pred))])
                - np.log10(np.sort(y_true[:, col_idx])[int(0.2 * len(y_true))])
                + np.log10(np.sort(y_true[:, col_idx])[int(0.7 * len(y_true))])
            )
            / np.log10(np.sort(y_true[:, col_idx])[int(0.2 * len(y_true))])
            - np.log10(np.sort(y_true[:, col_idx])[int(0.7 * len(y_true))])
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

    return {
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
