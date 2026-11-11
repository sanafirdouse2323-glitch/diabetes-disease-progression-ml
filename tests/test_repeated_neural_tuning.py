"""Tests for validation-selected neural tuning safeguards."""
import numpy as np
import pandas as pd
import pytest

from diabetes_ml.repeated_neural_tuning import CANDIDATE_CONFIGS
from diabetes_ml.repeated_validation import summarize_seed_metrics


def test_candidate_grid_includes_original_control_and_is_unique():
    names = [item["config"] for item in CANDIDATE_CONFIGS]
    assert "baseline_default" in names
    assert len(names) == len(set(names))
    assert all(item["batch_size"] > 0 for item in CANDIDATE_CONFIGS)
    assert all(item["learning_rate"] > 0 for item in CANDIDATE_CONFIGS)
    assert all(0 <= item["dropout_p"] < 1 for item in CANDIDATE_CONFIGS)


def test_seed_summary_rejects_missing_or_nonfinite_metrics():
    frame = pd.DataFrame({"seed": [1, 2], "model": ["m", "m"], "test_mae": [1.0, np.nan]})
    with pytest.raises(ValueError):
        summarize_seed_metrics(frame)
