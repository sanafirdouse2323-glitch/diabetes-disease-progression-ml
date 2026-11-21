"""Unit tests for classical regression estimators and cross-validation."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from diabetes_ml.data import create_experiment_splits
from diabetes_ml.models import evaluate_cv, get_classical_models


def test_classical_models_instantiation():
    models = get_classical_models(random_state=42)
    expected_names = {"Mean Baseline", "Linear Regression", "Ridge", "Lasso", "Random Forest"}
    assert set(models.keys()) == expected_names


def test_scale_sensitive_models_use_fold_local_pipeline_preprocessing():
    models = get_classical_models(random_state=42)

    for name in ("Linear Regression", "Ridge", "Lasso"):
        model = models[name]
        assert isinstance(model, Pipeline)
        assert isinstance(model.named_steps["scale"], StandardScaler)
        assert model.steps[-1][0] == "model"

    assert not isinstance(models["Mean Baseline"], Pipeline)
    assert not isinstance(models["Random Forest"], Pipeline)


def test_linear_pipeline_scaler_is_fitted_only_on_model_fit_rows():
    splits = create_experiment_splits(random_state=42)
    model = get_classical_models(random_state=42)["Ridge"]
    model.fit(splits.X_train, splits.y_train)

    scaler = model.named_steps["scale"]
    np.testing.assert_allclose(
        scaler.mean_,
        splits.X_train.mean(axis=0).to_numpy(dtype=float),
    )
    all_rows_mean = np.vstack([
        splits.X_train.to_numpy(dtype=float),
        splits.X_test.to_numpy(dtype=float),
    ]).mean(axis=0)
    assert not np.allclose(scaler.mean_, all_rows_mean)


def test_classical_models_fit_and_predict():
    splits = create_experiment_splits(random_state=42)
    models = get_classical_models(random_state=42)

    for name, model in models.items():
        model.fit(splits.X_train, splits.y_train)
        preds = model.predict(splits.X_test)
        assert len(preds) == len(splits.X_test)
        assert np.isfinite(preds).all()


def test_evaluate_cv():
    splits = create_experiment_splits(random_state=42)
    models = get_classical_models(random_state=42)
    ridge = models["Ridge"]

    cv_results = evaluate_cv(ridge, splits.X_train, splits.y_train, cv_folds=3, random_state=42)
    assert "cv_mae_mean" in cv_results
    assert "cv_rmse_mean" in cv_results
    assert "cv_r2_mean" in cv_results
    assert 30.0 < cv_results["cv_mae_mean"] < 70.0
