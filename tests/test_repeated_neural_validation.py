"""Tests for neural repeated-seed validation parameter checks."""

import numpy as np
import pytest

from diabetes_ml.repeated_neural_validation import _validated_training_parameters


def test_validated_training_parameters_returns_normalized_values():
    assert _validated_training_parameters(100, 12, 32, 0.001, 0.1) == (
        100, 12, 32, 0.001, 0.1
    )


@pytest.mark.parametrize(
    "args",
    [
        (0, 12, 32, 0.001, 0.1),
        (100, 0, 32, 0.001, 0.1),
        (100, 12, 0, 0.001, 0.1),
        (True, 12, 32, 0.001, 0.1),
        (100, 12, 32, 0.0, 0.1),
        (100, 12, 32, np.nan, 0.1),
        (100, 12, 32, 0.001, -0.1),
        (100, 12, 32, 0.001, 1.0),
    ],
)
def test_invalid_training_parameters_are_rejected(args):
    with pytest.raises(ValueError):
        _validated_training_parameters(*args)
