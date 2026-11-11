"""Evaluation subpackage: regression metrics, conformal uncertainty, subgroup errors, and feature attribution."""

from diabetes_ml.evaluation.metrics import (
    calculate_regression_metrics,
    regression_metrics,
    residual_summary,
)
from diabetes_ml.evaluation.uncertainty import (
    ConformalRegressor,
    PredictionInterval,
    evaluate_conformal_coverage,
)
from diabetes_ml.evaluation.fairness import (
    evaluate_subgroup_disparity,
    permutation_test_disparity,
)
from diabetes_ml.evaluation.explainability import (
    get_linear_attribution,
    get_tree_feature_importance,
    compute_patient_explanation,
)

__all__ = [
    "calculate_regression_metrics",
    "regression_metrics",
    "residual_summary",
    "ConformalRegressor",
    "PredictionInterval",
    "evaluate_conformal_coverage",
    "evaluate_subgroup_disparity",
    "permutation_test_disparity",
    "get_linear_attribution",
    "get_tree_feature_importance",
    "compute_patient_explanation",
]
