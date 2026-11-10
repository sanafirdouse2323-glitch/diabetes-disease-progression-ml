"""Standard regression evaluation metrics and error analysis."""

from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _validated_targets(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned finite one-dimensional target arrays."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if y_t.ndim != 1 or y_p.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")
    if y_t.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if y_t.shape != y_p.shape:
        raise ValueError("y_true and y_pred must have identical shapes.")
    if not np.isfinite(y_t).all() or not np.isfinite(y_p).all():
        raise ValueError("y_true and y_pred must contain only finite values.")
    return y_t, y_p


def calculate_regression_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> Dict[str, float]:
    """Calculate MAE, RMSE, R2, and Pearson correlation for aligned targets."""
    y_t, y_p = _validated_targets(y_true, y_pred)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    if np.std(y_p) > 1e-8 and np.std(y_t) > 1e-8:
        pearson_r = float(np.corrcoef(y_t, y_p)[0, 1])
    else:
        pearson_r = 0.0

    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson_r": pearson_r}


def residual_summary(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> Dict[str, float]:
    """Summarise residuals defined consistently as actual minus predicted."""
    y_t, y_p = _validated_targets(y_true, y_pred)
    residuals = y_t - y_p
    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_min": float(np.min(residuals)),
        "residual_max": float(np.max(residuals)),
    }


# Backwards-compatible public name retained for existing downstream users.
regression_metrics = calculate_regression_metrics
