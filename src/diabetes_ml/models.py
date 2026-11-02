"""Leakage-conscious model and pipeline definitions."""
from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(random_state: int = 42) -> dict[str, object]:
    """Return named estimators; all learned feature scaling stays in pipelines."""
    scaled = lambda estimator: Pipeline(
        [("scale", StandardScaler()), ("model", estimator)]
    )
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear_regression": scaled(LinearRegression()),
        "ridge": scaled(Ridge(alpha=1.0)),
        "lasso": scaled(Lasso(alpha=0.01, max_iter=20_000)),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        ),
    }
