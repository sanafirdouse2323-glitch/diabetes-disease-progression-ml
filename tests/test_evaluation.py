import numpy as np
import pytest

from diabetes_ml.evaluation import regression_metrics, residual_summary


def test_metrics_are_computed_correctly():
    result = regression_metrics([1, 2, 3], [1, 2, 4])
    assert result["mae"] == pytest.approx(1 / 3)
    assert result["rmse"] == pytest.approx((1 / 3) ** 0.5)
    assert result["r2"] == pytest.approx(0.5)


def test_residual_summary_uses_actual_minus_predicted():
    result = residual_summary([3, 5], [2, 7])
    assert result["residual_mean"] == pytest.approx(-0.5)
    assert result["residual_min"] == -2
    assert result["residual_max"] == 1


@pytest.mark.parametrize("truth,pred", [([], []), ([1], [1, 2]), ([np.nan], [0])])
def test_invalid_metric_inputs_are_rejected(truth, pred):
    with pytest.raises(ValueError):
        regression_metrics(truth, pred)
