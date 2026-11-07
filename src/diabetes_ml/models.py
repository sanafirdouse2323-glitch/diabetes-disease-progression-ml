"""Classical regression model definitions and cross-validation utilities."""

from __future__ import annotations

from typing import Dict
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate


def get_classical_models(random_state: int = 42) -> Dict[str, RegressorMixin]:
    """Instantiate the suite of classical regression models.

    Args:
        random_state: Random seed for stochastic models.

    Returns:
        Mapping of model names to instantiated estimators.
    """
    return {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.1, random_state=random_state),
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
    """Perform k-fold cross-validation inside the training partition only.

    Args:
        model: Regressor to evaluate.
        X: Training features.
        y: Training targets.
        cv_folds: Number of cross-validation folds.
        random_state: Seed for cross-validation shuffle.

    Returns:
        Dictionary with mean and std for MAE, RMSE, and R2.
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
