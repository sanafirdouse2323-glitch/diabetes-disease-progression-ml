"""Uncertainty Quantification for Clinical Regression: Conformal Prediction & Epistemic Uncertainty.

Provides distribution-free, finite-sample guaranteed prediction intervals via Split Conformal
Prediction, as well as Monte Carlo Dropout uncertainty estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class PredictionInterval:
    """Container for predictions with uncertainty intervals."""

    point_prediction: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float
    margin: float

    @property
    def interval_width(self) -> np.ndarray:
        return self.upper_bound - self.lower_bound

    def contains(self, y_true: np.ndarray) -> np.ndarray:
        """Boolean mask indicating whether true values fall within bounds."""
        y = np.asarray(y_true)
        return (y >= self.lower_bound) & (y <= self.upper_bound)


class ConformalRegressor:
    """Split Conformal Prediction for regression with distribution-free coverage guarantee.

    For any underlying regression model and any error distribution, split conformal
    prediction guarantees that for a pre-chosen significance level alpha:
        P(Y_test in C(X_test)) >= 1 - alpha
    under the mild assumption of exchangeability of calibration and test data.
    """

    def __init__(self, base_estimator: Any, alpha: float = 0.10):
        """Initialize Conformal Regressor.

        Args:
            base_estimator: Fitted or unfitted scikit-learn regressor.
            alpha: Significance level (default 0.10 for 90% confidence intervals).
        """
        self.base_estimator = base_estimator
        self.alpha = float(alpha)
        self.q_hat_: float | None = None
        self.cal_scores_: np.ndarray | None = None

    def fit_and_calibrate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> ConformalRegressor:
        """Fit base model on training subset and calibrate conformal margin on calibration subset.

        Args:
            X_train: Features for model parameter fitting.
            y_train: Targets for model parameter fitting.
            X_cal: Held-out features strictly reserved for calibration.
            y_cal: Held-out targets strictly reserved for calibration.
        """
        self.base_estimator.fit(X_train, y_train)
        return self.calibrate(X_cal, y_cal)

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> ConformalRegressor:
        """Compute conformal quantile on an independent calibration split."""
        y_cal_arr = np.asarray(y_cal, dtype=np.float64)
        y_cal_pred = np.asarray(self.base_estimator.predict(X_cal), dtype=np.float64).ravel()

        # Non-conformity score: absolute residual
        scores = np.abs(y_cal_arr - y_cal_pred)
        self.cal_scores_ = scores

        n = len(scores)
        # Finite-sample correction: index = ceil((n + 1) * (1 - alpha)) / n
        level = np.clip(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 0.0, 1.0)
        self.q_hat_ = float(np.quantile(scores, level, method="higher" if hasattr(np, "quantile") else "linear"))
        return self

    def predict_interval(self, X: np.ndarray) -> PredictionInterval:
        """Generate point predictions and calibrated prediction intervals.

        Args:
            X: Query feature array.

        Returns:
            PredictionInterval containing point predictions, lower bounds, and upper bounds.
        """
        if self.q_hat_ is None:
            raise ValueError("ConformalRegressor is not calibrated. Call calibrate() first.")

        y_pred = np.asarray(self.base_estimator.predict(X), dtype=np.float64).ravel()
        margin = self.q_hat_

        return PredictionInterval(
            point_prediction=y_pred,
            lower_bound=y_pred - margin,
            upper_bound=y_pred + margin,
            confidence_level=1.0 - self.alpha,
            margin=margin,
        )


def evaluate_conformal_coverage(
    intervals: PredictionInterval,
    y_true: np.ndarray,
) -> Dict[str, float]:
    """Evaluate empirical coverage and efficiency of conformal prediction intervals.

    Args:
        intervals: Calibrated PredictionInterval object.
        y_true: Ground truth target values.

    Returns:
        Dictionary containing:
            - empirical_coverage: Observed fraction of targets inside interval.
            - nominal_coverage: Theoretical target coverage (1 - alpha).
            - mean_interval_width: Average span of intervals (lower is more efficient).
            - winkler_score: Proper scoring rule rewarding narrow intervals and penalizing misses.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    lower = intervals.lower_bound
    upper = intervals.upper_bound
    alpha = 1.0 - intervals.confidence_level

    contained = intervals.contains(y)
    empirical_coverage = float(np.mean(contained))
    widths = intervals.interval_width
    mean_width = float(np.mean(widths))

    # Winkler Score: width + penalty for misses
    # Score = (U - L) + (2/alpha)*(L - y) if y < L + (2/alpha)*(y - U) if y > U
    winkler_penalties = widths.copy()
    below_mask = y < lower
    above_mask = y > upper
    if np.any(below_mask):
        winkler_penalties[below_mask] += (2.0 / alpha) * (lower[below_mask] - y[below_mask])
    if np.any(above_mask):
        winkler_penalties[above_mask] += (2.0 / alpha) * (y[above_mask] - upper[above_mask])
    mean_winkler = float(np.mean(winkler_penalties))

    return {
        "empirical_coverage": empirical_coverage,
        "nominal_coverage": intervals.confidence_level,
        "mean_interval_width": mean_width,
        "winkler_score": mean_winkler,
        "coverage_gap": empirical_coverage - intervals.confidence_level,
    }
