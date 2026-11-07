"""Unit tests for Conformal Prediction uncertainty quantification."""

import numpy as np
from sklearn.linear_model import Ridge
from diabetes_ml.data import create_experiment_splits
from diabetes_ml.evaluation.uncertainty import ConformalRegressor, evaluate_conformal_coverage


def test_conformal_regressor_calibration_and_coverage():
    splits = create_experiment_splits(random_state=42)
    base_model = Ridge()

    # Split training partition into fit (250) and calibration (103)
    X_fit = splits.X_train.iloc[:250].values
    y_fit = splits.y_train.iloc[:250].values
    X_cal = splits.X_train.iloc[250:].values
    y_cal = splits.y_train.iloc[250:].values

    cr = ConformalRegressor(base_model, alpha=0.10)  # 90% target coverage
    cr.fit_and_calibrate(X_fit, y_fit, X_cal, y_cal)

    assert cr.q_hat_ is not None
    assert cr.q_hat_ > 0.0

    intervals = cr.predict_interval(splits.X_test.values)
    assert len(intervals.point_prediction) == len(splits.y_test)
    assert np.all(intervals.lower_bound <= intervals.point_prediction)
    assert np.all(intervals.point_prediction <= intervals.upper_bound)

    metrics = evaluate_conformal_coverage(intervals, splits.y_test.values)
    assert "empirical_coverage" in metrics
    assert "mean_interval_width" in metrics
    assert "winkler_score" in metrics

    # Empirical coverage should be close to 0.90 (e.g. >= 0.80 on finite sample)
    assert 0.80 <= metrics["empirical_coverage"] <= 1.00
