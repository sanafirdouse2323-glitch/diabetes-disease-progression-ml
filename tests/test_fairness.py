"""Unit tests for validated exploratory subgroup error analysis."""

import numpy as np
import pandas as pd
import pytest

from diabetes_ml.evaluation.fairness import (
    evaluate_subgroup_disparity,
    permutation_test_disparity,
)


def test_evaluate_subgroup_disparity_reports_aligned_group_metrics():
    y_true = np.array([[100.0], [150.0], [200.0], [250.0]])
    y_pred = np.array([110.0, 140.0, 210.0, 240.0])
    groups = pd.Series(["A", "A", "B", "B"])

    result = evaluate_subgroup_disparity(
        y_true,
        y_pred,
        groups,
        subgroup_name="test_group",
    )

    assert result["attribute"] == "test_group"
    assert result["subgroups"]["A"]["n_samples"] == 2
    assert result["subgroups"]["B"]["n_samples"] == 2
    assert result["subgroups"]["A"]["mae"] == pytest.approx(10.0)
    assert result["subgroups"]["B"]["mae"] == pytest.approx(10.0)
    assert result["subgroups"]["A"]["r2"] == 0.0
    assert result["subgroups"]["B"]["r2"] == 0.0
    assert result["max_mae_disparity"] == pytest.approx(0.0)
    assert result["mae_parity_ratio"] == pytest.approx(1.0)


def test_permutation_test_is_reproducible_and_uses_plus_one_correction():
    rng = np.random.default_rng(2026)
    y_true = rng.uniform(50.0, 250.0, size=50)
    y_pred = y_true + rng.normal(0.0, 10.0, size=50)
    groups = np.array(["A"] * 25 + ["B"] * 25)

    p_value = permutation_test_disparity(
        y_true,
        y_pred,
        groups,
        n_permutations=200,
        random_state=42,
    )
    repeated = permutation_test_disparity(
        y_true,
        y_pred,
        groups,
        n_permutations=200,
        random_state=42,
    )

    assert p_value == pytest.approx(repeated)
    assert 1.0 / 201.0 <= p_value <= 1.0


@pytest.mark.parametrize("n_permutations", [0, -1, 1.5, True, np.nan])
def test_permutation_test_rejects_invalid_permutation_counts(n_permutations):
    with pytest.raises(ValueError):
        permutation_test_disparity(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.1, 0.9, 2.2, 2.8]),
            np.array(["A", "A", "B", "B"]),
            n_permutations=n_permutations,
        )


@pytest.mark.parametrize(
    "groups",
    [
        np.array(["A", "A", "A", "A"]),
        np.array(["A", "A", "B", "C"]),
    ],
)
def test_permutation_test_requires_exactly_two_groups(groups):
    with pytest.raises(ValueError, match="exactly two"):
        permutation_test_disparity(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.1, 0.9, 2.2, 2.8]),
            groups,
            n_permutations=10,
        )


@pytest.mark.parametrize(
    "y_true,y_pred,groups",
    [
        (np.array([1.0, 2.0]), np.array([1.0]), np.array(["A", "B"])),
        (np.array([1.0, np.nan]), np.array([1.0, 2.0]), np.array(["A", "B"])),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array(["A", None])),
        (np.ones((2, 2)), np.ones(4), np.array(["A", "A", "B", "B"])),
    ],
)
def test_subgroup_functions_reject_misaligned_or_invalid_inputs(y_true, y_pred, groups):
    with pytest.raises(ValueError):
        evaluate_subgroup_disparity(y_true, y_pred, groups)
    with pytest.raises(ValueError):
        permutation_test_disparity(y_true, y_pred, groups, n_permutations=10)


def test_subgroup_metrics_require_at_least_two_distinct_groups():
    with pytest.raises(ValueError, match="At least two"):
        evaluate_subgroup_disparity(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array(["A", "A", "A"]),
        )
