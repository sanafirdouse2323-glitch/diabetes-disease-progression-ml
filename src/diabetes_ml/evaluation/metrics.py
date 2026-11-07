"""Standard regression evaluation metrics and error analysis."""

from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> Dict[str, float]:
    """Calculate standard regression metrics: MAE, RMSE, and R2.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary containing MAE, RMSE, and R2.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    # Pearson correlation coefficient between true and predicted
    if np.std(y_p) > 1e-8 and np.std(y_t) > 1e-8:
        pearson_r = float(np.corrcoef(y_t, y_p)[0, 1])
    else:
        pearson_r = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
    }
