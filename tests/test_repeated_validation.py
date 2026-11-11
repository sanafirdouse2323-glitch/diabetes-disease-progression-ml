"""Tests for repeated-seed validation input checks and summary statistics."""

import numpy as np
import pandas as pd
import pytest

from diabetes_ml.repeated_validation import (
    _validated_seeds,
    run_repeated_seed_validation,
    summarize_seed_metrics,
)


def test_validated_seeds_preserves_order_and_normalizes_integer_types():
    assert _validated_seeds([42, np.int64(43)]) == (42, 43)


@pytest.mark.parametrize(
    "seeds",
    [[42], [], [42, 42], [42, True], "42 43", [42, -1], [42, 2**32]],
)
def test_invalid_seed_sequences_are_rejected(seeds):
    with pytest.raises(ValueError):
        _validated_seeds(seeds)


def test_summary_reports_mean_sample_standard_deviation_and_range():
    frame = pd.DataFrame({
        "seed": [1, 2, 1, 2],
        "model": ["A", "A", "B", "B"],
        "test_mae": [10.0, 14.0, 20.0, 24.0],
        "test_r2": [0.2, 0.4, 0.1, 0.3],
    })

    summary = summarize_seed_metrics(frame).set_index("model")

    assert summary.loc["A", "test_mae_mean"] == pytest.approx(12.0)
    assert summary.loc["A", "test_mae_std"] == pytest.approx(np.sqrt(8.0))
    assert summary.loc["A", "test_mae_min"] == pytest.approx(10.0)
    assert summary.loc["A", "test_mae_max"] == pytest.approx(14.0)
    assert summary.loc["B", "test_r2_mean"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(columns=["seed", "model", "test_mae"]),
        pd.DataFrame({"seed": [1, 1], "model": ["A", "A"], "test_mae": [1.0, 2.0]}),
        pd.DataFrame({"seed": [1, 2], "model": ["A", "B"], "test_mae": [1.0, 2.0]}),
        pd.DataFrame({"seed": [1, 2], "model": ["A", "A"], "test_mae": [1.0, np.nan]}),
    ],
)
def test_invalid_seed_metric_frames_are_rejected(frame):
    with pytest.raises(ValueError):
        summarize_seed_metrics(frame)


def test_matched_sample_partition_sizes_match_neural_pipeline(tmp_path):
    per_seed, _ = run_repeated_seed_validation(
        output_dir=tmp_path,
        seeds=[42, 43],
        val_fraction=0.20,
        cv_folds=3,
    )

    assert set(per_seed["fit_n"]) == {225}
    assert set(per_seed["validation_n"]) == {57}
    assert set(per_seed["calibration_n"]) == {71}
    assert set(per_seed["test_n"]) == {89}
\n