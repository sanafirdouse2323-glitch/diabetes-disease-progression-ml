"""Exploratory subgroup error analysis for regression models.

This module reports descriptive per-group regression errors and a Monte Carlo
permutation test for a difference in mean absolute error. It does not establish
that a model is fair, unbiased, or clinically appropriate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _as_numeric_vector(values: Any, name: str) -> np.ndarray:
    """Return a finite non-empty numeric vector or single-column array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional or a single-column array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_group_vector(values: Any) -> np.ndarray:
    """Return a non-empty one-dimensional vector with no missing group labels."""
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim != 1:
        raise ValueError("subgroup_series must be one-dimensional or a single-column array.")
    if array.size == 0:
        raise ValueError("subgroup_series must not be empty.")
    if pd.isna(array).any():
        raise ValueError("subgroup_series must not contain missing group labels.")
    return array


def _validated_inputs(
    y_true: Any,
    y_pred: Any,
    subgroup_series: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate aligned target, prediction, and subgroup vectors."""
    targets = _as_numeric_vector(y_true, "y_true")
    predictions = _as_numeric_vector(y_pred, "y_pred")
    groups = _as_group_vector(subgroup_series)

    if targets.shape != predictions.shape or targets.shape != groups.shape:
        raise ValueError("y_true, y_pred, and subgroup_series must have identical lengths.")
    return targets, predictions, groups


def evaluate_subgroup_disparity(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    subgroup_series: pd.Series | np.ndarray,
    subgroup_name: str = "cohort",
) -> dict[str, Any]:
    """Calculate descriptive regression metrics for each observed subgroup.

    Groups may be categorical or discrete. At least two non-missing groups are
    required. MAE disparity is the maximum subgroup MAE minus the minimum
    subgroup MAE. The parity ratio is the minimum subgroup MAE divided by the
    maximum subgroup MAE (1.0 indicates equal MAE in this metric only).

    R² is reported as 0.0 for groups with fewer than three observations because
    such a small subgroup does not support a useful R² summary.

    Returns:
        A dictionary containing the subgroup sample count, MAE, RMSE, and R²,
        together with maximum MAE disparity and the MAE parity ratio.
    """
    targets, predictions, groups = _validated_inputs(y_true, y_pred, subgroup_series)
    unique_groups = pd.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two distinct subgroup labels are required.")

    group_results: dict[str, dict[str, float | int]] = {}
    for group in unique_groups:
        mask = groups == group
        group_targets = targets[mask]
        group_predictions = predictions[mask]

        group_results[str(group)] = {
            "n_samples": int(mask.sum()),
            "mae": float(mean_absolute_error(group_targets, group_predictions)),
            "rmse": float(np.sqrt(mean_squared_error(group_targets, group_predictions))),
            "r2": float(r2_score(group_targets, group_predictions))
            if group_targets.size >= 3
            else 0.0,
        }

    group_maes = [metrics["mae"] for metrics in group_results.values()]
    max_mae = max(group_maes)
    min_mae = min(group_maes)
    max_mae_disparity = float(max_mae - min_mae)
    mae_parity_ratio = float(min_mae / max_mae) if max_mae > 0.0 else 1.0

    return {
        "attribute": subgroup_name,
        "subgroups": group_results,
        "max_mae_disparity": max_mae_disparity,
        "mae_parity_ratio": mae_parity_ratio,
    }


def permutation_test_disparity(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    subgroup_series: pd.Series | np.ndarray,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> float:
    """Test a two-group difference in mean absolute error by label permutation.

    The null hypothesis is that absolute prediction errors are exchangeable
    across the two subgroup labels. Group sizes are preserved while labels are
    permuted. A plus-one Monte Carlo correction is used:

        p = (1 + number of permuted disparities >= observed) / (B + 1).

    The returned p-value is not the probability that the null hypothesis is
    true. This test is exploratory and does not by itself establish fairness.
    """
    targets, predictions, groups = _validated_inputs(y_true, y_pred, subgroup_series)
    if isinstance(n_permutations, (bool, np.bool_)) or not isinstance(
        n_permutations, (int, np.integer)
    ):
        raise ValueError("n_permutations must be a positive integer.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer.")

    unique_groups = pd.unique(groups)
    if len(unique_groups) != 2:
        raise ValueError("permutation_test_disparity requires exactly two distinct groups.")

    errors = np.abs(targets - predictions)
    group_a, group_b = unique_groups
    observed = abs(
        float(np.mean(errors[groups == group_a]))
        - float(np.mean(errors[groups == group_b]))
    )

    rng = np.random.default_rng(random_state)
    null_disparities = np.empty(n_permutations, dtype=np.float64)
    for index in range(n_permutations):
        permuted_groups = rng.permutation(groups)
        null_disparities[index] = abs(
            float(np.mean(errors[permuted_groups == group_a]))
            - float(np.mean(errors[permuted_groups == group_b]))
        )

    extreme_count = int(np.count_nonzero(null_disparities >= observed))
    return float((extreme_count + 1) / (n_permutations + 1))
