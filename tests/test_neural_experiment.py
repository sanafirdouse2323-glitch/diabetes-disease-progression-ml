"""Tests for complete two-framework neural comparison preflight."""

import pytest

from diabetes_ml.neural.neural_experiment import _require_both_neural_frameworks


@pytest.mark.parametrize(
    "torch_available,tensorflow_available,missing",
    [
        (False, True, "PyTorch"),
        (True, False, "TensorFlow/Keras"),
        (False, False, "PyTorch, TensorFlow/Keras"),
    ],
)
def test_full_comparison_rejects_missing_frameworks_before_running(
    torch_available, tensorflow_available, missing
):
    with pytest.raises(ImportError, match=missing):
        _require_both_neural_frameworks(torch_available, tensorflow_available)


def test_full_comparison_accepts_both_frameworks():
    _require_both_neural_frameworks(True, True)
