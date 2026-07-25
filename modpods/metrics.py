import numpy as np


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
