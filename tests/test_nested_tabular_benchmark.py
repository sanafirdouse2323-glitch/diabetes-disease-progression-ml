"""Tests for nested repeated-CV model benchmarking."""

import numpy as np
import pandas as pd
import pytest

from diabetes_ml.nested_tabular_benchmark import (
    _summarize,
    run_nested_tabular_benchmark,
)


def test_summary_rejects_missing_or_duplicate_outer_fold_results():
    duplicate = pd.DataFrame({
        "model": ["A", "A"],
        "outer_fold": [0, 0],
        "test_mae": [1.0, 2.0],
        "test_rmse": [2.0, 3.0],
        "test_r2": [0.1, 0.2],
    })
    with pytest.raises(ValueError):
        _summarize(duplicate)


def test_summary_reports_descriptive_outer_fold_statistics():
    frame = pd.DataFrame({
        "model": ["A", "A", "B", "B"],
        "outer_fold": [0, 1, 0, 1],
        "test_mae": [10.0, 14.0, 8.0, 12.0],
        "test_rmse": [12.0, 16.0, 10.0, 14.0],
        "test_r2": [0.2, 0.4, 0.3, 0.5],
    })
    summary = _summarize(frame).set_index("model")
    assert summary.loc["A", "test_mae_mean"] == pytest.approx(12.0)
    assert summary.loc["A", "test_mae_std"] == pytest.approx(np.sqrt(8.0))
    assert summary.loc["B", "test_r2_mean"] == pytest.approx(0.4)


def test_nested_benchmark_tunes_inside_outer_training_folds(tmp_path):
    per_fold, summary = run_nested_tabular_benchmark(
        output_dir=tmp_path,
        random_state=17,
        outer_folds=2,
        outer_repeats=2,
        inner_folds=2,
        model_names=["Ridge"],
    )
    assert len(per_fold) == 4
    assert per_fold["outer_fold"].nunique() == 4
    assert set(per_fold["model"]) == {"Ridge"}
    assert (per_fold["outer_train_n"] + per_fold["outer_test_n"] == 442).all()
    assert np.isfinite(per_fold[["inner_best_mae", "test_mae", "test_rmse", "test_r2"]]).all().all()
    assert summary.loc[0, "test_mae_mean"] == pytest.approx(per_fold["test_mae"].mean())
    assert (tmp_path / "outer_fold_metrics.csv").exists()
    assert (tmp_path / "model_summary.csv").exists()
    assert (tmp_path / "benchmark_metadata.json").exists()


def test_nested_benchmark_rejects_unknown_models(tmp_path):
    with pytest.raises(ValueError, match="Unknown model names"):
        run_nested_tabular_benchmark(
            output_dir=tmp_path, outer_folds=2, outer_repeats=2,
            inner_folds=2, model_names=["Made-up model"],
        )
