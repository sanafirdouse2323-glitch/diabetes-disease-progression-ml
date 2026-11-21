"""Classical regression estimators and fold-local preprocessing utilities."""

from __future__ import annotations

from typing import Dict

from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _scaled_model(estimator: RegressorMixin) -> Pipeline:
    """Scale features within each estimator fit, never before data splitting."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def get_classical_models(random_state: int = 42) -> Dict[str, RegressorMixin]:
    """Instantiate the classical model suite.

    Scale-sensitive linear estimators are wrapped in a pipeline so the scaler
    is fitted only on the data passed to that pipeline's fit call. During
    cross-validation, scikit-learn clones and fits the complete pipeline
    separately inside each training fold. Tree and mean-baseline estimators do
    not require feature scaling and are left unwrapped.
    """
    return {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": _scaled_model(LinearRegression()),
        "Ridge": _scaled_model(Ridge(alpha=1.0, random_state=random_state)),
        "Lasso": _scaled_model(Lasso(alpha=0.1, random_state=random_state)),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def evaluate_cv(
    model: RegressorMixin,
    X,
    y,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Perform k-fold cross-validation inside the model-fitting partition.

    For pipeline estimators, each fold's scaler is fitted only on that fold's
    training observations because preprocessing and estimation are evaluated
    as one scikit-learn pipeline.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

    return {
        "cv_mae_mean": float(-scores["test_mae"].mean()),
        "cv_mae_std": float(scores["test_mae"].std()),
        "cv_rmse_mean": float(-scores["test_rmse"].mean()),
        "cv_rmse_std": float(scores["test_rmse"].std()),
        "cv_r2_mean": float(scores["test_r2"].mean()),
        "cv_r2_std": float(scores["test_r2"].std()),
    }
