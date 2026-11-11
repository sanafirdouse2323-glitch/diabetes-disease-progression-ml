"""Tests for nested repeated-CV model benchmarking."""

import json

import numpy as np
import pandas as pd
import pytest

from diabetes_ml.nested_tabular_benchmark import (
    _model_specs,
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


def test_summary_rejects_a_model_missing_an_outer_fold():
    missing = pd.DataFrame({
        "model": ["A", "A", "B"],
        "outer_fold": [0, 1, 0],
        "test_mae": [10.0, 14.0, 8.0],
        "test_rmse": [12.0, 16.0, 10.0],
        "test_r2": [0.2, 0.4, 0.3],
    })
    with pytest.raises(ValueError, match="same outer folds"):
        _summarize(missing)


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
    metadata = json.loads((tmp_path / "benchmark_metadata.json").read_text(encoding="utf-8"))
    assert metadata["outer_test_use"] == (
        "Each outer split is evaluated once and its targets are not used for fitting or selection; "
        "because outer cross-validation is repeated, an observation can appear in test partitions "
        "in more than one repeat."
    )
    assert (tmp_path / "family_selection_metrics.csv").exists()
    assert (tmp_path / "family_selection_summary.csv").exists()
    assert (tmp_path / "selected_family_frequency.csv").exists()
    selected = pd.read_csv(tmp_path / "family_selection_metrics.csv")
    assert len(selected) == 4
    assert set(selected["selected_model"]) == {"Ridge"}
    assert np.isfinite(selected[["inner_best_mae", "test_mae", "test_rmse", "test_r2"]]).all().all()


def test_family_selector_uses_inner_cv_and_evaluates_outer_fold(tmp_path):
    per_fold, _ = run_nested_tabular_benchmark(
        output_dir=tmp_path,
        random_state=23,
        outer_folds=2,
        outer_repeats=2,
        inner_folds=2,
        model_names=["Mean Baseline", "Linear Regression", "Ridge"],
    )
    selected = pd.read_csv(tmp_path / "family_selection_metrics.csv")
    assert len(selected) == 4
    assert set(selected["selected_model"]).issubset(
        {"Mean Baseline", "Linear Regression", "Ridge"}
    )
    assert selected["outer_fold"].is_unique
    assert set(selected["outer_fold"]) == set(per_fold["outer_fold"])
    expected = (
        per_fold.loc[per_fold.groupby("outer_fold")["inner_best_mae"].idxmin(),
                     ["outer_fold", "model"]]
        .sort_values("outer_fold")
        .reset_index(drop=True)
    )
    observed = (
        selected[["outer_fold", "selected_model"]]
        .rename(columns={"selected_model": "model"})
        .sort_values("outer_fold")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(observed, expected)
    assert np.isfinite(selected[["inner_best_mae", "test_mae", "test_rmse", "test_r2"]]).all().all()


def test_nested_benchmark_rejects_impossible_fold_counts(tmp_path):
    with pytest.raises(ValueError, match="outer_folds cannot exceed"):
        run_nested_tabular_benchmark(
            output_dir=tmp_path, outer_folds=443, outer_repeats=2,
            inner_folds=2, model_names=["Ridge"],
        )


def test_nested_benchmark_rejects_unknown_models(tmp_path):
    with pytest.raises(ValueError, match="Unknown model names"):
        run_nested_tabular_benchmark(
            output_dir=tmp_path, outer_folds=2, outer_repeats=2,
            inner_folds=2, model_names=["Made-up model"],
        )


def test_model_specs_use_the_same_svr_label_as_published_results():
    specs = _model_specs(random_state=42)
    assert "SVR-RBF" in specs
    assert "RBF-SVR" not in specs
