"""Unit tests for demographic subgroup disparity and permutation testing."""

import numpy as np
import pandas as pd
from diabetes_ml.evaluation.fairness import evaluate_subgroup_disparity, permutation_test_disparity


def test_evaluate_subgroup_disparity():
    y_true = np.array([100.0, 150.0, 200.0, 250.0])
    y_pred = np.array([110.0, 140.0, 210.0, 240.0])
    groups = np.array(["A", "A", "B", "B"])

    res = evaluate_subgroup_disparity(y_true, y_pred, groups, subgroup_name="test_group")
    assert res["attribute"] == "test_group"
    assert "A" in res["subgroups"]
    assert "B" in res["subgroups"]
    assert res["subgroups"]["A"]["mae"] == 10.0
    assert res["subgroups"]["B"]["mae"] == 10.0
    assert res["max_mae_disparity"] == 0.0
    assert res["mae_parity_ratio"] == 1.0


def test_permutation_test_disparity():
    y_true = np.random.uniform(50, 250, size=50)
    y_pred = y_true + np.random.normal(0, 10, size=50)
    groups = np.array(["Female"] * 25 + ["Male"] * 25)

    p_val = permutation_test_disparity(y_true, y_pred, groups, n_permutations=200, random_state=42)
    assert 0.0 <= p_val <= 1.0
