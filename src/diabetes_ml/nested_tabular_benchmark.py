"""Nested repeated-CV benchmark for small tabular regression.

Every model family is tuned only inside each outer training fold. The outer
fold is used once for evaluation. This experiment is separate from split-
conformal calibration and does not use the existing repeated hold-out scores
for model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from diabetes_ml.data import load_diabetes_data


def _model_specs(random_state: int) -> dict[str, tuple[Any, list[dict[str, list[Any]]]]]:
    """Return a compact, predeclared search space for each model family."""
    def scaled(estimator: Any) -> Pipeline:
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])

    return {
        "Mean Baseline": (DummyRegressor(strategy="mean"), [{}]),
        "Linear Regression": (
            scaled(LinearRegression()), [{}]
        ),
        "Ridge": (
            scaled(Ridge()),
            [{"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}],
        ),
        "ElasticNet": (
            scaled(ElasticNet(max_iter=20000, random_state=random_state)),
            [{"model__alpha": [0.001, 0.01, 0.1, 1.0],
              "model__l1_ratio": [0.1, 0.5, 0.9]}],
        ),
        "SVR-RBF": (
            scaled(SVR(kernel="rbf")),
            [{"model__C": [0.1, 1.0, 10.0, 100.0],
              "model__epsilon": [0.05, 0.1, 0.2],
              "model__gamma": ["scale", 0.01, 0.1]}],
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=1),
            [{"max_depth": [None, 3, 5],
              "min_samples_leaf": [1, 3, 5],
              "max_features": [0.7, 1.0]}],
        ),
        "Extra Trees": (
            ExtraTreesRegressor(n_estimators=300, random_state=random_state, n_jobs=1),
            [{"max_depth": [None, 3, 5],
              "min_samples_leaf": [1, 3, 5],
              "max_features": [0.7, 1.0]}],
        ),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(random_state=random_state),
            [{"learning_rate": [0.03, 0.1],
              "max_iter": [100, 300],
              "max_leaf_nodes": [7, 15],
              "l2_regularization": [0.0, 1.0]}],
        ),
    }


def _summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Summarize outer-fold scores; SDs are descriptive, not confidence intervals."""
    required = {"model", "outer_fold", "test_mae", "test_rmse", "test_r2"}
    if not required.issubset(per_fold.columns) or per_fold.empty:
        raise ValueError("per_fold must contain model, outer_fold and finite test metrics.")
    if per_fold.duplicated(["model", "outer_fold"]).any():
        raise ValueError("Each model/outer-fold combination must occur once.")
    fold_sets = per_fold.groupby("model", sort=False)["outer_fold"].apply(set)
    expected_folds = set(per_fold["outer_fold"])
    if any(folds != expected_folds for folds in fold_sets):
        raise ValueError("Every model must be evaluated on the same outer folds.")
    values = per_fold[["test_mae", "test_rmse", "test_r2"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Outer-fold metrics must all be finite.")
    summary = (
        per_fold.groupby("model", sort=False)[["test_mae", "test_rmse", "test_r2"]]
        .agg(["count", "mean", "std", "median", "min", "max"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def run_nested_tabular_benchmark(
    output_dir: str | Path = "nested-tabular-benchmark",
    random_state: int = 42,
    outer_folds: int = 5,
    outer_repeats: int = 3,
    inner_folds: int = 3,
    model_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run repeated nested CV with independent inner tuning and outer evaluation.

    The primary selection metric is MAE. Each model family is tuned separately
    within each outer training fold; no outer-fold target is used for tuning.
    """
    for value, name in ((outer_folds, "outer_folds"),
                        (outer_repeats, "outer_repeats"),
                        (inner_folds, "inner_folds")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2.")
    if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
        raise ValueError("random_state must be a non-negative integer.")

    X, y = load_diabetes_data()
    if outer_folds > len(X):
        raise ValueError("outer_folds cannot exceed the number of observations.")
    smallest_outer_train = len(X) - int(np.ceil(len(X) / outer_folds))
    if inner_folds > smallest_outer_train:
        raise ValueError("inner_folds cannot exceed the smallest outer training partition.")
    specs = _model_specs(random_state)
    selected_names = list(specs) if model_names is None else list(model_names)
    if not selected_names or len(set(selected_names)) != len(selected_names):
        raise ValueError("model_names must be a non-empty list of unique known model names.")
    unknown = sorted(set(selected_names) - set(specs))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")

    outer_cv = RepeatedKFold(
        n_splits=outer_folds, n_repeats=outer_repeats, random_state=random_state
    )
    records: list[dict[str, Any]] = []
    family_selection_records: list[dict[str, Any]] = []
    for fold_index, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
        X_outer_train, X_outer_test = X.iloc[train_index], X.iloc[test_index]
        y_outer_train, y_outer_test = y.iloc[train_index], y.iloc[test_index]
        inner_cv = KFold(
            n_splits=inner_folds, shuffle=True,
            random_state=random_state + fold_index + 1,
        )
        outer_searches: dict[str, GridSearchCV] = {}
        for model_name in selected_names:
            estimator, param_grid = specs[model_name]
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="neg_mean_absolute_error",
                cv=inner_cv,
                refit=True,
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(X_outer_train, y_outer_train)
            outer_searches[model_name] = search
            prediction = search.predict(X_outer_test)
            records.append({
                "outer_fold": fold_index,
                "repeat": fold_index // outer_folds,
                "fold_within_repeat": fold_index % outer_folds,
                "model": model_name,
                "inner_best_mae": float(-search.best_score_),
                "test_mae": float(mean_absolute_error(y_outer_test, prediction)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_outer_test, prediction))),
                "test_r2": float(r2_score(y_outer_test, prediction)),
                "best_params": json.dumps(search.best_params_, sort_keys=True, default=str),
                "outer_train_n": int(len(train_index)),
                "outer_test_n": int(len(test_index)),
            })

        # Select the model family using inner-CV MAE only, then evaluate that
        # complete selection procedure once on the untouched outer fold.
        selected_name = max(selected_names, key=lambda name: outer_searches[name].best_score_)
        selected_search = outer_searches[selected_name]
        selected_prediction = selected_search.predict(X_outer_test)
        family_selection_records.append({
            "outer_fold": fold_index,
            "repeat": fold_index // outer_folds,
            "fold_within_repeat": fold_index % outer_folds,
            "selected_model": selected_name,
            "inner_best_mae": float(-selected_search.best_score_),
            "test_mae": float(mean_absolute_error(y_outer_test, selected_prediction)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_outer_test, selected_prediction))),
            "test_r2": float(r2_score(y_outer_test, selected_prediction)),
            "best_params": json.dumps(selected_search.best_params_, sort_keys=True, default=str),
            "outer_train_n": int(len(train_index)),
            "outer_test_n": int(len(test_index)),
        })

    per_fold = pd.DataFrame.from_records(records)
    summary = _summarize(per_fold)
    family_selection = pd.DataFrame.from_records(family_selection_records)
    selector_summary = family_selection[["test_mae", "test_rmse", "test_r2"]].agg(
        ["count", "mean", "std", "median", "min", "max"]
    ).transpose()
    selector_summary.insert(0, "metric", selector_summary.index)
    selector_summary = selector_summary.reset_index(drop=True)
    selector_frequency = (
        family_selection["selected_model"].value_counts()
        .rename_axis("model").reset_index(name="selected_outer_folds")
    )
    selector_frequency["selection_rate"] = (
        selector_frequency["selected_outer_folds"] / len(family_selection)
    )

    # Paired fold-wise comparison to ordinary least squares; descriptive only.
    linear = per_fold.loc[per_fold["model"] == "Linear Regression",
                          ["outer_fold", "test_mae"]].rename(
                              columns={"test_mae": "linear_test_mae"})
    if not linear.empty:
        paired = per_fold.merge(linear, on="outer_fold", how="left")
        paired["mae_difference_vs_linear"] = paired["test_mae"] - paired["linear_test_mae"]
        paired_summary = paired.groupby("model").agg(
            mean_mae_difference_vs_linear=("mae_difference_vs_linear", "mean"),
            outer_folds_won_vs_linear=(
                "mae_difference_vs_linear", lambda values: int((values < 0).sum())
            ),
        ).reset_index()
        summary = summary.merge(paired_summary, on="model", how="left")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(out / "outer_fold_metrics.csv", index=False)
    summary.to_csv(out / "model_summary.csv", index=False)
    family_selection.to_csv(out / "family_selection_metrics.csv", index=False)
    selector_summary.to_csv(out / "family_selection_summary.csv", index=False)
    selector_frequency.to_csv(out / "selected_family_frequency.csv", index=False)
    metadata = {
        "dataset": "sklearn.datasets.load_diabetes",
        "observations": int(len(X)),
        "features": int(X.shape[1]),
        "outer_method": "RepeatedKFold",
        "outer_folds": outer_folds,
        "outer_repeats": outer_repeats,
        "outer_evaluations_per_model": outer_folds * outer_repeats,
        "inner_method": "shuffled KFold within each outer training fold",
        "inner_folds": inner_folds,
        "selection_metric": "negative mean absolute error",
        "selection_boundary": "all hyperparameter and model-family selection occurs using inner folds drawn only from each outer training partition",
        "family_selection_procedure": "For each outer fold, tune each included model family on the same predeclared inner folds; choose the family with the lowest inner-CV MAE; use that family's refitted best estimator for the outer-fold evaluation.",
        "family_comparison_interpretation": "Per-family outer-fold summaries compare the predeclared family-specific tuning procedures. The separate family_selection_metrics.csv evaluates the procedure that chooses a family using inner-CV MAE; use this file when discussing performance after family selection.",
        "outer_test_use": "one-time evaluation only; never used for fitting or tuning",
        "preprocessing": "StandardScaler is fitted inside the pipeline on each inner training fold for scale-sensitive models; tree models are not scaled",
        "uncertainty": "This benchmark does not fit conformal intervals; conformal calibration remains a separate experiment.",
        "summary_interpretation": "Repeated outer folds overlap across repeats. Means and standard deviations are descriptive and are not independent-sample confidence intervals.",
        "limitations": [
            "One small, preprocessed benchmark dataset; no external validation.",
            "The model grids are intentionally compact and do not constitute exhaustive optimization.",
            "The per-family summary ranks families descriptively; the family-selection estimate is reported separately because choosing the smallest outer-fold mean after inspecting results would introduce selection optimism.",
            "Repeated cross-validation folds are dependent because observations recur across repeats.",
            "Predictive performance does not establish clinical validity or utility.",
        ],
        "models": selected_names,
        "random_state": random_state,
    }
    (out / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return per_fold, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nested repeated-CV tabular regression benchmark.")
    parser.add_argument("--output", default="nested-tabular-benchmark")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    args = parser.parse_args()
    per_fold, summary = run_nested_tabular_benchmark(
        output_dir=args.output,
        random_state=args.random_state,
        outer_folds=args.outer_folds,
        outer_repeats=args.outer_repeats,
        inner_folds=args.inner_folds,
    )
    print(f"Completed {per_fold['outer_fold'].nunique()} outer folds.")
    print("\nPer-family outer-fold comparison (descriptive):")
    print(summary.to_string(index=False))
    print("\nInner-CV model-family selection procedure (outer-fold evaluation):")
    print(pd.read_csv(Path(args.output) / "family_selection_summary.csv").to_string(index=False))
    print("\nSelected-family frequency:")
    print(pd.read_csv(Path(args.output) / "selected_family_frequency.csv").to_string(index=False))


if __name__ == "__main__":
    main()
