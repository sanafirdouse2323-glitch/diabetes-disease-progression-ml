"""Subgroup Disparity and Algorithmic Fairness Analysis for Clinical ML.

Evaluates performance disparities across demographic cohorts (sex, age groups) to detect
disparate error rates, differential coverage, and assess statistical significance of disparities.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_subgroup_disparity(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    subgroup_series: pd.Series | np.ndarray,
    subgroup_name: str = "cohort",
) -> Dict[str, Any]:
    """Calculate regression performance metrics disaggregated by subgroup.

    Args:
        y_true: True continuous targets.
        y_pred: Predicted values.
        subgroup_series: Categorical or discrete indicator of cohort membership.
        subgroup_name: Label for the demographic attribute.

    Returns:
        Structured dictionary containing:
            - per_group_metrics: Dict mapping group labels to MAE, RMSE, R2, sample size.
            - max_mae_disparity: Absolute difference between highest and lowest subgroup MAE.
            - mae_ratio: Min MAE / Max MAE (1.0 indicates perfect parity).
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    s = np.asarray(subgroup_series).ravel()

    unique_groups = np.unique(s)
    group_results: Dict[str, Dict[str, float]] = {}

    for grp in unique_groups:
        mask = s == grp
        if np.sum(mask) == 0:
            continue
        g_yt = y_t[mask]
        g_yp = y_p[mask]

        g_mae = float(mean_absolute_error(g_yt, g_yp))
        g_rmse = float(np.sqrt(mean_squared_error(g_yt, g_yp)))
        g_r2 = float(r2_score(g_yt, g_yp)) if len(g_yt) > 2 else 0.0

        group_results[str(grp)] = {
            "n_samples": int(np.sum(mask)),
            "mae": g_mae,
            "rmse": g_rmse,
            "r2": g_r2,
        }

    maes = [m["mae"] for m in group_results.values()]
    max_mae_disparity = float(max(maes) - min(maes)) if len(maes) > 1 else 0.0
    mae_ratio = float(min(maes) / max(maes)) if len(maes) > 1 and max(maes) > 0 else 1.0

    return {
        "attribute": subgroup_name,
        "subgroups": group_results,
        "max_mae_disparity": max_mae_disparity,
        "mae_parity_ratio": mae_ratio,
    }


def permutation_test_disparity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_series: np.ndarray,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> float:
    """Assess whether observed subgroup MAE disparity is statistically significant.

    Under the null hypothesis H0, absolute errors are independent of subgroup labels.
    Permutes subgroup labels to construct the empirical null distribution of MAE disparity.

    Returns:
        p-value (probability of observing a disparity >= observed under H0).
    """
    rng = np.random.default_rng(random_state)
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    s = np.asarray(subgroup_series)
    unique_groups = np.unique(s)

    if len(unique_groups) != 2:
        return 1.0

    mask_a = s == unique_groups[0]
    mask_b = s == unique_groups[1]
    observed_disparity = abs(float(np.mean(errors[mask_a])) - float(np.mean(errors[mask_b])))

    null_disparities = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(s)
        null_a = np.mean(errors[shuffled == unique_groups[0]])
        null_b = np.mean(errors[shuffled == unique_groups[1]])
        null_disparities[i] = abs(null_a - null_b)

    p_value = float(np.mean(null_disparities >= observed_disparity))
    return p_value
