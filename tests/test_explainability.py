"""Tests for model-based feature attribution and local feature-profile explanations."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from diabetes_ml.data import create_experiment_splits
from diabetes_ml.evaluation.explainability import (
    compute_patient_explanation,
    get_linear_attribution,
    get_tree_feature_importance,
)


def test_linear_attribution_is_finite_and_normalized():
    splits = create_experiment_splits(random_state=42)
    model = Ridge().fit(splits.X_train, splits.y_train)

    attribution = get_linear_attribution(model, splits.feature_names)

    assert len(attribution) == 10
    assert {"feature", "coefficient", "abs_coefficient", "relative_importance"} <= set(
        attribution.columns
    )
    assert np.isfinite(attribution[["coefficient", "abs_coefficient", "relative_importance"]]).all().all()
    assert np.isclose(attribution["relative_importance"].sum(), 1.0)
    assert attribution["abs_coefficient"].is_monotonic_decreasing


def test_zero_linear_coefficients_have_zero_relative_importance():
    model = SimpleNamespace(coef_=np.zeros(3))

    attribution = get_linear_attribution(model, ["a", "b", "c"])

    assert np.isfinite(attribution["relative_importance"]).all()
    assert np.array_equal(attribution["relative_importance"].to_numpy(), np.zeros(3))


def test_tree_feature_importance_is_validated_and_sorted():
    splits = create_experiment_splits(random_state=42)
    model = RandomForestRegressor(n_estimators=20, random_state=42).fit(
        splits.X_train,
        splits.y_train,
    )

    importance = get_tree_feature_importance(model, splits.feature_names)

    assert len(importance) == 10
    assert np.isfinite(importance["importance"]).all()
    assert (importance["importance"] >= 0.0).all()
    assert importance["importance"].is_monotonic_decreasing
    assert importance["importance"].sum() == pytest.approx(1.0)


def test_linear_local_contributions_sum_to_prediction_difference():
    splits = create_experiment_splits(random_state=42)
    model = Ridge().fit(splits.X_train, splits.y_train)

    feature_profile = splits.X_test.iloc[0].to_numpy(dtype=float)
    reference = splits.X_train.mean().to_numpy(dtype=float)
    explanation = compute_patient_explanation(
        model,
        feature_profile,
        splits.feature_names,
        reference,
    )

    expected_delta = float(model.predict(feature_profile.reshape(1, -1))[0]) - float(
        model.predict(reference.reshape(1, -1))[0]
    )
    observed_delta = float(explanation["contribution_to_progression"].sum())

    assert len(explanation) == 10
    assert np.isfinite(explanation["contribution_to_progression"]).all()
    assert observed_delta == pytest.approx(expected_delta, rel=1e-10, abs=1e-10)


def test_tree_local_explanation_uses_one_feature_at_a_time_perturbations():
    splits = create_experiment_splits(random_state=42)
    model = RandomForestRegressor(n_estimators=20, random_state=42).fit(
        splits.X_train,
        splits.y_train,
    )

    feature_profile = splits.X_test.iloc[0].to_numpy(dtype=float)
    reference = splits.X_train.mean().to_numpy(dtype=float)
    explanation = compute_patient_explanation(
        model,
        feature_profile,
        splits.feature_names,
        reference,
    )

    assert len(explanation) == 10
    assert np.isfinite(explanation["contribution_to_progression"]).all()
    # Perturbation contributions are intentionally not required to sum to the
    # profile-to-reference prediction difference for a nonlinear model.


@pytest.mark.parametrize(
    "patient,reference,names",
    [
        (np.array([1.0, 2.0]), np.array([0.0, 0.0]), ["only_one_name"]),
        (np.array([1.0, 2.0]), np.array([0.0]), ["a", "b"]),
        (np.array([1.0, np.nan]), np.array([0.0, 0.0]), ["a", "b"]),
        (np.array([1.0, np.inf]), np.array([0.0, 0.0]), ["a", "b"]),
        (np.ones((2, 2)), np.zeros(4), ["a", "b", "c", "d"]),
        (np.array([1.0, 2.0]), np.array([0.0, 0.0]), ["a", "a"]),
    ],
)
def test_local_explanation_rejects_invalid_feature_inputs(patient, reference, names):
    model = Ridge().fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        compute_patient_explanation(model, patient, names, reference)


@pytest.mark.parametrize(
    "model,names",
    [
        (SimpleNamespace(coef_=np.array([1.0, 2.0])), ["a"]),
        (SimpleNamespace(coef_=np.array([1.0, np.nan])), ["a", "b"]),
        (SimpleNamespace(coef_=np.array([[1.0, 2.0], [3.0, 4.0]])), ["a", "b"]),
    ],
)
def test_linear_attribution_rejects_invalid_coefficients(model, names):
    with pytest.raises(ValueError):
        get_linear_attribution(model, names)


@pytest.mark.parametrize(
    "model,names",
    [
        (SimpleNamespace(feature_importances_=np.array([0.2, 0.8])), ["a"]),
        (SimpleNamespace(feature_importances_=np.array([-0.2, 1.2])), ["a", "b"]),
        (SimpleNamespace(feature_importances_=np.array([0.5, np.inf])), ["a", "b"]),
    ],
)
def test_tree_importance_rejects_invalid_importances(model, names):
    with pytest.raises(ValueError):
        get_tree_feature_importance(model, names)


def test_linear_local_explanation_rejects_wrong_coefficient_count():
    model = SimpleNamespace(coef_=np.array([1.0, 2.0, 3.0]), predict=lambda X: np.array([1.0]))
    with pytest.raises(ValueError, match="coefficient count"):
        compute_patient_explanation(
            model,
            np.array([1.0, 2.0]),
            ["a", "b"],
            np.zeros(2),
        )


def test_nonlinear_explanation_rejects_non_finite_prediction():
    class InvalidPredictor:
        def predict(self, X):
            return np.array([np.nan])

    with pytest.raises(ValueError, match="finite prediction"):
        compute_patient_explanation(
            InvalidPredictor(),
            np.array([1.0, 2.0]),
            ["a", "b"],
            np.zeros(2),
        )
