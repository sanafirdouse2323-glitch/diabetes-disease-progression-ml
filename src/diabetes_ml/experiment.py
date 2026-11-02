"""Run the complete classical-ML benchmark experiment from the command line."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, cross_validate

from diabetes_ml.data import load_diabetes_frame, make_train_test_split
from diabetes_ml.evaluation import regression_metrics, residual_summary
from diabetes_ml.models import build_models


def run_experiment(output_dir: str | Path = "results", random_state: int = 42) -> dict:
    """Fit models on training data, select no test-set hyperparameters, and report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_diabetes_frame()
    split = make_train_test_split(X, y, random_state=random_state)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    for name, estimator in build_models(random_state).items():
        cv_result = cross_validate(
            estimator,
            split.X_train,
            split.y_train,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=None,
            error_score="raise",
        )
        estimator.fit(split.X_train, split.y_train)
        pred = estimator.predict(split.X_test)
        metrics = regression_metrics(split.y_test, pred)
        residuals = residual_summary(split.y_test, pred)
        rows.append({
            "model": name,
            **metrics,
            **residuals,
            "cv_mae_mean": float(-cv_result["test_mae"].mean()),
            "cv_mae_std": float(cv_result["test_mae"].std(ddof=1)),
            "cv_rmse_mean": float(-cv_result["test_rmse"].mean()),
            "cv_rmse_std": float(cv_result["test_rmse"].std(ddof=1)),
            "cv_r2_mean": float(cv_result["test_r2"].mean()),
            "cv_r2_std": float(cv_result["test_r2"].std(ddof=1)),
        })
        prediction_frames.append(pd.DataFrame({
            "row_index": split.X_test.index,
            "model": name,
            "actual": split.y_test.to_numpy(),
            "predicted": pred,
            "residual": split.y_test.to_numpy() - pred,
        }))

    results = pd.DataFrame(rows).sort_values("cv_mae_mean")
    results.to_csv(out / "model_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        out / "test_predictions.csv", index=False
    )
    metadata = {
        "dataset": "sklearn.datasets.load_diabetes",
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "train_samples": int(len(split.X_train)),
        "test_samples": int(len(split.X_test)),
        "random_state": random_state,
        "cv": "5-fold shuffled KFold on training partition",
        "selection_metric": "mean cross-validation MAE",
        "test_set_used_for_model_selection": False,
        "target": "continuous one-year disease-progression measure",
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {"metrics": results, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results", help="Directory for CSV/JSON outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and CV.")
    args = parser.parse_args()
    result = run_experiment(args.output, args.seed)
    print(result["metrics"].to_string(index=False))
    print(f"\\nOutputs written to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
