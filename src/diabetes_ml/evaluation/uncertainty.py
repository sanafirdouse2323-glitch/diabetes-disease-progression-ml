"""Split-conformal prediction intervals for regression.

The implementation uses absolute residual scores and the finite-sample
order-statistic correction. Coverage is marginal over exchangeable future
observations; it is not a patient-specific probability statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


def _as_vector(
    values: Any,
    name: str,
    *,
    allow_infinite: bool = False,
) -> np.ndarray:
    """Convert a vector or single-column array to a validated float vector."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional or a single-column array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.isnan(array).any():
        raise ValueError(f"{name} must not contain NaN values.")
    if not allow_infinite and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_features(X: Any, name: str) -> np.ndarray:
    """Validate a non-empty, finite two-dimensional feature matrix."""
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional feature matrix.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class PredictionInterval:
    """Container for point predictions and their calibrated prediction intervals."""

    point_prediction: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float
    margin: float

    def __post_init__(self) -> None:
        point = _as_vector(self.point_prediction, "point_prediction")
        lower = _as_vector(self.lower_bound, "lower_bound", allow_infinite=True)
        upper = _as_vector(self.upper_bound, "upper_bound", allow_infinite=True)
        if point.shape != lower.shape or point.shape != upper.shape:
            raise ValueError("Point predictions and interval bounds must have identical shapes.")
        if np.any(lower > upper):
            raise ValueError("Every lower bound must be less than or equal to its upper bound.")

        confidence = float(self.confidence_level)
        if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be strictly between 0 and 1.")
        margin = float(self.margin)
        if np.isnan(margin) or margin < 0.0:
            raise ValueError("margin must be non-negative and must not be NaN.")

        object.__setattr__(self, "point_prediction", point)
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "margin", margin)

    @property
    def interval_width(self) -> np.ndarray:
        return self.upper_bound - self.lower_bound

    def contains(self, y_true: np.ndarray) -> np.ndarray:
        """Return a row-aligned mask indicating which targets lie inside the bounds."""
        y = _as_vector(y_true, "y_true")
        if y.shape != self.point_prediction.shape:
            raise ValueError("y_true and prediction intervals must have identical shapes.")
        return (y >= self.lower_bound) & (y <= self.upper_bound)


class ConformalRegressor:
    """Split-conformal regression using a separately held-out calibration subset.

    The base estimator is fitted only on the model-fitting data. Calibration
    residuals are then computed on observations that must not have been used
    for fitting, preprocessing choices, or early-stopping/model selection.
    Under exchangeability of calibration and future test observations, the
    resulting interval has finite-sample marginal coverage of at least 1-alpha.
    """

    def __init__(self, base_estimator: Any, alpha: float = 0.10):
        self.base_estimator = base_estimator
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1.")
        self.q_hat_: float | None = None
        self.cal_scores_: np.ndarray | None = None

    def fit_and_calibrate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> ConformalRegressor:
        """Fit on the model-fitting subset, then calibrate on a disjoint subset."""
        X_train_array = _validate_features(X_train, "X_train")
        X_cal_array = _validate_features(X_cal, "X_cal")
        y_train_array = _as_vector(y_train, "y_train")
        y_cal_array = _as_vector(y_cal, "y_cal")
        if X_train_array.shape[0] != y_train_array.shape[0]:
            raise ValueError("X_train and y_train must contain the same number of rows.")
        if X_cal_array.shape[0] != y_cal_array.shape[0]:
            raise ValueError("X_cal and y_cal must contain the same number of rows.")
        if X_train_array.shape[1] != X_cal_array.shape[1]:
            raise ValueError("X_train and X_cal must have the same number of features.")

        self.base_estimator.fit(X_train, y_train_array)
        return self.calibrate(X_cal, y_cal_array)

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> ConformalRegressor:
        """Calibrate the residual quantile on an independent calibration subset."""
        X_cal_array = _validate_features(X_cal, "X_cal")
        y_cal_array = _as_vector(y_cal, "y_cal")
        if X_cal_array.shape[0] != y_cal_array.shape[0]:
            raise ValueError("X_cal and y_cal must contain the same number of rows.")

        predictions = _as_vector(
            self.base_estimator.predict(X_cal),
            "calibration predictions",
        )
        if predictions.shape != y_cal_array.shape:
            raise ValueError("Calibration targets and predictions must have identical shapes.")

        scores = np.abs(y_cal_array - predictions)
        if not np.isfinite(scores).all():
            raise ValueError("Calibration residual scores must be finite.")
        self.cal_scores_ = scores

        # Exact finite-sample rank: k = ceil((n + 1) * (1 - alpha)).
        # If k > n, the valid split-conformal threshold is positive infinity.
        n_cal = scores.size
        rank = int(np.ceil((n_cal + 1) * (1.0 - self.alpha)))
        if rank > n_cal:
            self.q_hat_ = float("inf")
        else:
            self.q_hat_ = float(np.sort(scores)[rank - 1])
        return self

    def predict_interval(self, X: np.ndarray) -> PredictionInterval:
        """Generate point predictions and symmetric split-conformal intervals."""
        if self.q_hat_ is None:
            raise ValueError("ConformalRegressor is not calibrated. Call calibrate() first.")
        X_array = _validate_features(X, "X")
        predictions = _as_vector(self.base_estimator.predict(X), "predictions")
        if predictions.shape[0] != X_array.shape[0]:
            raise ValueError("X and predictions must contain the same number of rows.")

        return PredictionInterval(
            point_prediction=predictions,
            lower_bound=predictions - self.q_hat_,
            upper_bound=predictions + self.q_hat_,
            confidence_level=1.0 - self.alpha,
            margin=self.q_hat_,
        )


def evaluate_conformal_coverage(
    intervals: PredictionInterval,
    y_true: np.ndarray,
) -> Dict[str, float]:
    """Evaluate empirical marginal coverage, interval width, and Winkler score."""
    y = _as_vector(y_true, "y_true")
    lower = _as_vector(intervals.lower_bound, "lower_bound", allow_infinite=True)
    upper = _as_vector(intervals.upper_bound, "upper_bound", allow_infinite=True)
    widths = upper - lower
    if y.shape != lower.shape or y.shape != upper.shape:
        raise ValueError("y_true and prediction intervals must have identical shapes.")
    if np.isnan(widths).any() or np.any(widths < 0.0):
        raise ValueError("Interval widths must be non-negative and must not be NaN.")

    alpha = 1.0 - intervals.confidence_level
    contained = (y >= lower) & (y <= upper)
    empirical_coverage = float(np.mean(contained))
    mean_width = float(np.mean(widths))

    # Winkler interval score: width plus an alpha-scaled penalty for misses.
    penalties = widths.copy()
    below = y < lower
    above = y > upper
    penalties[below] += (2.0 / alpha) * (lower[below] - y[below])
    penalties[above] += (2.0 / alpha) * (y[above] - upper[above])
    mean_winkler = float(np.mean(penalties))

    return {
        "empirical_coverage": empirical_coverage,
        "nominal_coverage": intervals.confidence_level,
        "mean_interval_width": mean_width,
        "winkler_score": mean_winkler,
        "coverage_gap": empirical_coverage - intervals.confidence_level,
    }
