"""Model-based feature attribution methods for the tabular regression benchmark.

Implemented methods are linear-model coefficients, Random Forest impurity
importance, exact baseline-relative contributions for linear models, and
one-feature-at-a-time perturbations for other regressors.

This module does not implement SHAP, neural input-gradient attribution, causal
inference, or a validated clinical explanation method.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def _feature_names(values: Sequence[str], expected_size: int) -> list[str]:
    """Validate feature names and align them with the feature dimension."""
    names = list(values)
    if not names:
        raise ValueError("feature_names must not be empty.")
    if len(names) != expected_size:
        raise ValueError("feature_names length must match the number of features.")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("feature_names must contain non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique.")
    return names


def _feature_vector(values: Any, name: str) -> np.ndarray:
    """Convert a one-dimensional vector or single-row/column array to float64."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and (array.shape[0] == 1 or array.shape[1] == 1):
        array = array.reshape(-1)
    elif array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional or a single row/column.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _fitted_vector_attribute(model: Any, attribute: str, name: str) -> np.ndarray:
    """Read a finite fitted model attribute as a one-dimensional vector."""
    if not hasattr(model, attribute):
        raise ValueError(f"model must expose a fitted {attribute} attribute.")
    values = np.asarray(getattr(model, attribute), dtype=np.float64)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values.reshape(-1)
    elif values.ndim != 1:
        raise ValueError(f"{name} must represent a single-output model.")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be non-empty and finite.")
    return values


def _single_prediction(model: Any, vector: np.ndarray, name: str) -> float:
    """Predict for one observation and validate a single finite output."""
    if not callable(getattr(model, "predict", None)):
        raise ValueError("model must provide a callable predict method.")
    prediction = np.asarray(model.predict(vector.reshape(1, -1)), dtype=np.float64)
    if prediction.size != 1 or not np.isfinite(prediction).all():
        raise ValueError(f"{name} must return exactly one finite prediction.")
    return float(prediction.reshape(-1)[0])


def get_linear_attribution(
    linear_model: Any,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Return signed coefficients and normalized absolute coefficient magnitudes.

    This is a model-parameter summary, not a causal effect or a SHAP explanation.
    If all coefficients are zero, relative importance is defined as zero for
    every feature rather than producing NaN values.
    """
    coefficients = _fitted_vector_attribute(linear_model, "coef_", "linear coefficients")
    names = _feature_names(feature_names, coefficients.size)
    absolute = np.abs(coefficients)
    total_absolute = float(absolute.sum())
    relative = absolute / total_absolute if total_absolute > 0.0 else np.zeros_like(absolute)

    frame = pd.DataFrame({
        "feature": names,
        "coefficient": coefficients,
        "abs_coefficient": absolute,
        "relative_importance": relative,
    })
    return frame.sort_values(
        by="abs_coefficient",
        ascending=False,
    ).reset_index(drop=True)


def get_tree_feature_importance(
    tree_model: Any,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Return fitted tree-ensemble mean-decrease-in-impurity importances.

    Impurity importances are model-dependent and can be biased by feature
    characteristics and correlations. They are not causal explanations.
    """
    importances = _fitted_vector_attribute(
        tree_model,
        "feature_importances_",
        "tree feature importances",
    )
    if np.any(importances < 0.0):
        raise ValueError("tree feature importances must be non-negative.")
    names = _feature_names(feature_names, importances.size)

    frame = pd.DataFrame({
        "feature": names,
        "importance": importances,
    })
    return frame.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)


def compute_patient_explanation(
    model: Any,
    patient_vector: np.ndarray | pd.Series,
    feature_names: Sequence[str],
    reference_baseline: np.ndarray,
) -> pd.DataFrame:
    """Compute baseline-relative feature contributions for one feature profile.

    For single-output linear models, contributions are calculated as the
    coefficient multiplied by the feature's difference from its reference
    value. Their sum equals the prediction difference between the profile
    and baseline, up to floating-point precision.

    For other regressors, each contribution is the difference between the
    profile prediction and a prediction with just that feature replaced by its
    reference value. These one-feature-at-a-time perturbations are not
    guaranteed to add up to the total prediction change; interactions and
    correlated features can make the individual changes overlap.

    The legacy function name is retained for compatibility. The function is a
    local model explanation, not SHAP, gradient attribution, causal inference,
    or a validated clinical explanation.
    """
    vector = _feature_vector(patient_vector, "patient_vector")
    baseline = _feature_vector(reference_baseline, "reference_baseline")
    names = _feature_names(feature_names, vector.size)
    if baseline.shape != vector.shape:
        raise ValueError("reference_baseline must have the same number of features as patient_vector.")

    difference = vector - baseline

    if hasattr(model, "coef_"):
        coefficients = _fitted_vector_attribute(model, "coef_", "linear coefficients")
        if coefficients.shape != vector.shape:
            raise ValueError("linear coefficient count must match the feature count.")
        contributions = coefficients * difference
    else:
        base_prediction = _single_prediction(model, vector, "model.predict")
        contributions = np.zeros(vector.size, dtype=np.float64)
        for index in range(vector.size):
            perturbed = vector.copy()
            perturbed[index] = baseline[index]
            perturbed_prediction = _single_prediction(
                model,
                perturbed,
                "model.predict",
            )
            contributions[index] = base_prediction - perturbed_prediction

    if not np.isfinite(contributions).all():
        raise ValueError("feature contributions must be finite.")

    frame = pd.DataFrame({
        "feature": names,
        "patient_value": vector,
        "reference_value": baseline,
        "deviation": difference,
        "contribution_to_progression": contributions,
    })
    return frame.sort_values(
        by="contribution_to_progression",
        key=np.abs,
        ascending=False,
    ).reset_index(drop=True)
