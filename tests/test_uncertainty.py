"""Unit tests for conformal prediction shape handling and finite-sample ranks."""

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from diabetes_ml.data import create_experiment_splits
from diabetes_ml.evaluation.uncertainty import (
    ConformalRegressor,
    PredictionInterval,
    evaluate_conformal_coverage,
)


def test_conformal_regressor_calibration_and_coverage():
    splits = create_experiment_splits(random_state=42)
    base_model = Ridge()

    # Use disjoint rows from the outer training partition.
    X_fit = splits.X_train.iloc[:250].values
    y_fit = splits.y_train.iloc[:250].values
    X_cal = splits.X_train.iloc[250:].values
    # Single-column targets explicitly exercise protection against n-by-n broadcasting.
    y_cal = splits.y_train.iloc[250:].values.reshape(-1, 1)

    cr = ConformalRegressor(base_model, alpha=0.10)
    cr.fit_and_calibrate(X_fit, y_fit, X_cal, y_cal)

    assert cr.q_hat_ is not None and np.isfinite(cr.q_hat_)
    assert cr.q_hat_ > 0.0
    assert cr.cal_scores_.shape == (len(y_cal),)

    intervals = cr.predict_interval(splits.X_test.values)
    assert intervals.point_prediction.shape == (len(splits.y_test),)
    assert np.all(intervals.lower_bound <= intervals.point_prediction)
    assert np.all(intervals.point_prediction <= intervals.upper_bound)

    metrics = evaluate_conformal_coverage(intervals, splits.y_test.values)
    assert 0.0 <= metrics["empirical_coverage"] <= 1.0
    assert metrics["nominal_coverage"] == pytest.approx(0.90)
    assert metrics["mean_interval_width"] > 0.0
    assert np.isfinite(metrics["winkler_score"])


def test_finite_sample_quantile_uses_exact_order_statistic():
    X_fit = np.arange(4, dtype=float).reshape(-1, 1)
    y_fit = np.zeros(4, dtype=float)
    X_cal = np.arange(4, 8, dtype=float).reshape(-1, 1)
    y_cal = np.array([[1.0], [2.0], [3.0], [4.0]])

    model = DummyRegressor(strategy="constant", constant=0.0)
    conformal = ConformalRegressor(model, alpha=0.40)
    conformal.fit_and_calibrate(X_fit, y_fit, X_cal, y_cal)

    # k = ceil((4 + 1) * (1 - 0.40)) = 3; the third sorted score is 3.
    assert conformal.q_hat_ == pytest.approx(3.0)
    assert conformal.cal_scores_.shape == (4,)


def test_insufficient_calibration_size_returns_unbounded_valid_interval():
    X_fit = np.arange(4, dtype=float).reshape(-1, 1)
    y_fit = np.zeros(4, dtype=float)
    X_cal = np.arange(4, 8, dtype=float).reshape(-1, 1)
    y_cal = np.arange(1, 5, dtype=float)

    conformal = ConformalRegressor(
        DummyRegressor(strategy="constant", constant=0.0),
        alpha=0.10,
    )
    conformal.fit_and_calibrate(X_fit, y_fit, X_cal, y_cal)

    # k = ceil((4 + 1) * 0.90) = 5, which exceeds n=4.
    assert np.isinf(conformal.q_hat_)
    intervals = conformal.predict_interval(np.array([[10.0], [11.0]]))
    assert np.isneginf(intervals.lower_bound).all()
    assert np.isposinf(intervals.upper_bound).all()


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, np.nan, np.inf])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError):
        ConformalRegressor(DummyRegressor(), alpha=alpha)


@pytest.mark.parametrize(
    "X_cal,y_cal",
    [
        (np.empty((0, 1)), np.empty(0)),
        (np.ones((3, 1)), np.ones(2)),
        (np.ones((3, 1)), np.array([1.0, np.nan, 2.0])),
        (np.ones((3, 1)), np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])),
    ],
)
def test_invalid_calibration_inputs_are_rejected(X_cal, y_cal):
    model = DummyRegressor(strategy="constant", constant=0.0)
    with pytest.raises(ValueError):
        ConformalRegressor(model).fit_and_calibrate(
            np.ones((3, 1)),
            np.ones(3),
            X_cal,
            y_cal,
        )


def test_interval_evaluation_rejects_misaligned_targets():
    intervals = PredictionInterval(
        point_prediction=np.array([1.0, 2.0]),
        lower_bound=np.array([0.0, 1.0]),
        upper_bound=np.array([2.0, 3.0]),
        confidence_level=0.90,
        margin=1.0,
    )
    with pytest.raises(ValueError):
        evaluate_conformal_coverage(intervals, np.array([1.0]))
