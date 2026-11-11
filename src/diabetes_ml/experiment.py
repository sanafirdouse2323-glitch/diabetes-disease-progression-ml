"""Classical regression benchmark with split-conformal uncertainty and subgroup error analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from diabetes_ml.data import create_experiment_splits
from diabetes_ml.evaluation import (
    ConformalRegressor,
    calculate_regression_metrics,
    evaluate_conformal_coverage,
    evaluate_subgroup_disparity,
    get_linear_attribution,
    get_tree_feature_importance,
    permutation_test_disparity,
)
from diabetes_ml.models import evaluate_cv, get_classical_models


def run_classical_experiment(
    output_dir: str | Path = "results",
    test_size: float = 0.20,
    cal_fraction: float = 0.20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run CV, held-out regression evaluation, conformal intervals, and exploratory subgroup error analysis."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits = create_experiment_splits(test_size=test_size, random_state=random_state)
    models = get_classical_models(random_state=random_state)

    # Inner calibration split inside the outer training partition (strictly no test data)
    X_tr_fit, X_tr_cal, y_tr_fit, y_tr_cal = train_test_split(
        splits.X_train,
        splits.y_train,
        test_size=cal_fraction,
        random_state=random_state,
        shuffle=True,
    )

    records: List[Dict[str, Any]] = []
    fairness_records: Dict[str, Any] = {}

    for name, model in models.items():
        # 1. Cross-validation stays inside the model-fitting subset; calibration
        # observations are not used for model assessment or selection.
        cv_scores = evaluate_cv(model, X_tr_fit, y_tr_fit, cv_folds=5, random_state=random_state)

        # 2. Conformal Prediction calibration
        conformal_wrapper = ConformalRegressor(model, alpha=0.10)  # 90% target coverage
        conformal_wrapper.fit_and_calibrate(
            X_tr_fit.values,
            y_tr_fit.values,
            X_tr_cal.values,
            y_tr_cal.values,
        )

        # 3. Held-out test evaluation
        intervals = conformal_wrapper.predict_interval(splits.X_test.values)
        y_test_pred = intervals.point_prediction

        reg_metrics = calculate_regression_metrics(splits.y_test.values, y_test_pred)
        conf_metrics = evaluate_conformal_coverage(intervals, splits.y_test.values)

        # 4. Exploratory subgroup error analysis (standardized sex feature only)
        # Note: In standard sklearn diabetes, sex is standardized into two discrete float values
        sex_disparity = evaluate_subgroup_disparity(
            splits.y_test.values,
            y_test_pred,
            splits.X_test["sex"],
            subgroup_name="sex",
        )
        sex_p_val = permutation_test_disparity(
            splits.y_test.values,
            y_test_pred,
            splits.X_test["sex"].values,
            n_permutations=1000,
            random_state=random_state,
        )

        record = {
            "model": name,
            "cv_mae_mean": round(cv_scores["cv_mae_mean"], 3),
            "cv_mae_std": round(cv_scores["cv_mae_std"], 3),
            "test_mae": round(reg_metrics["mae"], 3),
            "test_rmse": round(reg_metrics["rmse"], 3),
            "test_r2": round(reg_metrics["r2"], 3),
            "conformal_coverage_90": round(conf_metrics["empirical_coverage"], 3),
            "conformal_interval_width": round(conf_metrics["mean_interval_width"], 3),
            "sex_mae_disparity": round(sex_disparity["max_mae_disparity"], 3),
            "sex_disparity_p_val": round(sex_p_val, 4),
        }
        records.append(record)
        fairness_records[name] = sex_disparity

    df_results = pd.DataFrame(records)
    df_results.to_csv(out_path / "classical_metrics.csv", index=False)

    with open(out_path / "fairness_audit.json", "w", encoding="utf-8") as f:
        json.dump(fairness_records, f, indent=2)

    # 5. Export illustrative feature attribution from Ridge and Random Forest.
    # These estimators were fitted on X_tr_fit above and are the same
    # estimators used for conformal calibration. Do not refit them on calibration data.
    ridge_model = models["Ridge"]
    ridge_attr = get_linear_attribution(ridge_model, splits.feature_names)
    ridge_attr.to_csv(out_path / "ridge_feature_attribution.csv", index=False)

    rf_model = models["Random Forest"]
    rf_attr = get_tree_feature_importance(rf_model, splits.feature_names)
    rf_attr.to_csv(out_path / "rf_feature_importance.csv", index=False)

    return df_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical ML benchmark on diabetes progression.")
    parser.add_argument("--output", default="results", help="Directory to save experimental results.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed.")
    args = parser.parse_args()

    results = run_classical_experiment(output_dir=args.output, random_state=args.seed)
    print("\n=== Classical Benchmark Results ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
