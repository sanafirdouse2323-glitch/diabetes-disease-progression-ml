"""Evaluation subpackage: metrics, conformal uncertainty, subgroup fairness, and explainability."""

from diabetes_ml.evaluation.metrics import calculate_regression_metrics
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
    "ConformalRegressor",
    "PredictionInterval",
    "evaluate_conformal_coverage",
    "evaluate_subgroup_disparity",
    "permutation_test_disparity",
    "get_linear_attribution",
    "get_tree_feature_importance",
    "compute_patient_explanation",
]
