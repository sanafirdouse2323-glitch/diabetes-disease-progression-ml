"""Tests for validation-selected neural tuning safeguards."""
import numpy as np
import pandas as pd
import pytest

from diabetes_ml.repeated_neural_tuning import CANDIDATE_CONFIGS
from diabetes_ml.repeated_validation import summarize_seed_metrics
from diabetes_ml.neural.pytorch_model import TORCH_AVAILABLE, PyTorchTrainer


def test_candidate_grid_includes_control_and_unique_settings():
    names = [item["config"] for item in CANDIDATE_CONFIGS]
    assert "baseline_default" in names
    assert len(names) == len(set(names))
    assert all(item["batch_size"] > 0 for item in CANDIDATE_CONFIGS)
    assert all(item["learning_rate"] > 0 for item in CANDIDATE_CONFIGS)
    assert all(0 <= item["dropout_p"] < 1 for item in CANDIDATE_CONFIGS)
    assert {item["standardize_target"] for item in CANDIDATE_CONFIGS} == {False, True}


def test_seed_summary_rejects_missing_or_nonfinite_metrics():
    frame = pd.DataFrame({"seed": [1, 2], "model": ["m", "m"], "test_mae": [1.0, np.nan]})
    with pytest.raises(ValueError):
        summarize_seed_metrics(frame)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch optional dependency is not installed")
def test_target_standardization_uses_fit_targets_and_predicts_original_units():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 3)).astype(np.float32)
    y = (120.0 + 15.0 * X[:, :1] - 8.0 * X[:, 1:2]).astype(np.float32)
    trainer = PyTorchTrainer(
        in_features=3, epochs=8, patience=4, random_state=7,
        standardize_target=True, dropout_p=0.0,
    ).fit(X[:30], y[:30], X[30:], y[30:])
    predictions = trainer.predict(X[30:])
    assert trainer.target_mean_ == pytest.approx(float(np.mean(y[:30])))
    assert np.isfinite(predictions).all()
    assert np.isfinite(trainer.history[-1]["val_mse"])
    assert predictions.shape == (10,)
