"""Clinical Feature Attribution and Model Explainability (XAI).

Provides linear coefficient attribution, tree-based feature importance, neural input-gradient
attribution, and patient-level explanation breakdowns.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd


def get_linear_attribution(
    linear_model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract standardized coefficients and relative impact from linear estimators.

    Args:
        linear_model: Fitted LinearRegression, Ridge, or Lasso model.
        feature_names: List of column names.

    Returns:
        DataFrame sorted by absolute coefficient magnitude.
    """
    coefs = np.asarray(linear_model.coef_, dtype=np.float64).ravel()
    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    })
    df["relative_importance"] = df["abs_coefficient"] / df["abs_coefficient"].sum()
    return df.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)


def get_tree_feature_importance(
    tree_model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract MDI (mean decrease in impurity) feature importances from tree ensembles.

    Args:
        tree_model: Fitted RandomForestRegressor.
        feature_names: Feature names.

    Returns:
        DataFrame sorted by importance score.
    """
    importances = np.asarray(tree_model.feature_importances_, dtype=np.float64)
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)


def compute_patient_explanation(
    model: Any,
    patient_vector: np.ndarray | pd.Series,
    feature_names: List[str],
    reference_baseline: np.ndarray,
) -> pd.DataFrame:
    """Generate local feature attributions for an individual patient relative to baseline.

    Uses linear attribution approximation: Delta_j = beta_j * (x_j - x_ref_j).
    For tree or non-linear models, uses marginal contribution approximation.

    Args:
        model: Fitted regressor with coef_ or predict method.
        patient_vector: Patient's feature values.
        feature_names: Feature names.
        reference_baseline: Reference population mean or median vector.

    Returns:
        DataFrame containing feature values, deviations from baseline, and contribution to risk.
    """
    x = np.asarray(patient_vector, dtype=np.float64).ravel()
    x_ref = np.asarray(reference_baseline, dtype=np.float64).ravel()
    diff = x - x_ref

    if hasattr(model, "coef_"):
        coefs = np.asarray(model.coef_, dtype=np.float64).ravel()
        contributions = coefs * diff
    else:
        # Perturbation approximation: change one feature at a time to reference value
        base_pred = float(model.predict(x.reshape(1, -1))[0])
        contributions = np.zeros(len(x))
        for j in range(len(x)):
            x_perturbed = x.copy()
            x_perturbed[j] = x_ref[j]
            pred_pert = float(model.predict(x_perturbed.reshape(1, -1))[0])
            contributions[j] = base_pred - pred_pert

    df = pd.DataFrame({
        "feature": feature_names,
        "patient_value": x,
        "reference_value": x_ref,
        "deviation": diff,
        "contribution_to_progression": contributions,
    })
    return df.sort_values(by="contribution_to_progression", key=abs, ascending=False).reset_index(drop=True)
