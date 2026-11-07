"""Unit tests for feature attribution and local patient explanation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from diabetes_ml.data import create_experiment_splits
from diabetes_ml.evaluation.explainability import (
    compute_patient_explanation,
    get_linear_attribution,
)


def test_linear_attribution():
    splits = create_experiment_splits(random_state=42)
    model = Ridge().fit(splits.X_train, splits.y_train)

    attr = get_linear_attribution(model, splits.feature_names)
    assert len(attr) == 10
    assert "feature" in attr.columns
    assert "coefficient" in attr.columns
    assert "relative_importance" in attr.columns
    assert np.isclose(attr["relative_importance"].sum(), 1.0)


def test_compute_patient_explanation():
    splits = create_experiment_splits(random_state=42)
    model = Ridge().fit(splits.X_train, splits.y_train)

    patient = splits.X_test.iloc[0]
    ref_baseline = splits.X_train.mean().values

    expl = compute_patient_explanation(model, patient, splits.feature_names, ref_baseline)
    assert len(expl) == 10
    assert "contribution_to_progression" in expl.columns
    assert not expl["contribution_to_progression"].isna().any()
