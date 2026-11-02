"""Regression metrics and residual diagnostics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute standard regression metrics after validating input arrays."""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.size == 0 or truth.size != pred.size:
        raise ValueError("y_true and y_pred must be non-empty and have equal length.")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("y_true and y_pred must contain only finite values.")
    mse = mean_squared_error(truth, pred)
    return {
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(truth, pred)),
    }


def residual_summary(y_true, y_pred) -> dict[str, float]:
    """Summarise signed residuals (actual minus predicted)."""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.size == 0 or truth.size != pred.size:
        raise ValueError("y_true and y_pred must be non-empty and have equal length.")
    residuals = truth - pred
    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=0)),
        "residual_min": float(np.min(residuals)),
        "residual_max": float(np.max(residuals)),
    }
