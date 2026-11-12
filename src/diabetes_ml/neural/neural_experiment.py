"""Full Comparative Experiment Pipeline: Classical vs PyTorch vs TensorFlow with Conformal Uncertainty."""

from __future__ import annotations

import argparsefrom pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from diabetes_ml.data import create_experiment_splits
from diabetes_ml.evaluation import (
    ConformalRegressor,
    calculate_regression_metrics,
    evaluate_conformal_coverage,
    evaluate_subgroup_disparity,
    permutation_test_disparity,
)
from diabetes_ml.experiment import run_classical_experiment
from diabetes_ml.neural.neural_data import prepare_neural_datasets
from diabetes_ml.neural.pytorch_model import TORCH_AVAILABLE, PyTorchTrainer
from diabetes_ml.neural.tensorflow_model import TF_AVAILABLE, TensorFlowTrainer


def run_full_comparison_experiment(
    output_dir: str | Path = "results",
    epochs: int = 200,
    patience: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run full benchmark comparing classical regressors with PyTorch and TensorFlow MLPs."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Run classical models first
    classical_df = run_classical_experiment(output_dir=out_path, random_state=random_state)

    splits = create_experiment_splits(random_state=random_state)
    neural_arrays = prepare_neural_datasets(
        splits.X_train,
        splits.y_train,
        splits.X_test,
        splits.y_test,
        val_fraction=0.20,
        random_state=random_state,
    )

    comparison_records: List[Dict[str, Any]] = classical_df.to_dict(orient="records")
    histories: List[Dict[str, Any]] = []

    # 2. PyTorch Evaluation
    if TORCH_AVAILABLE:
        print("Training PyTorch MLP with Early Stopping...")
        pt_trainer = PyTorchTrainer(
            in_features=10,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            dropout_p=0.1,
            random_state=random_state,
        )
        pt_trainer.fit(
            neural_arrays.X_train,
            neural_arrays.y_train,
            neural_arrays.X_val,
            neural_arrays.y_val,
        )

        for rec in pt_trainer.history:
            histories.append({"framework": "PyTorch", **rec})

        pt_pred = pt_trainer.predict(neural_arrays.X_test)
        pt_metrics = calculate_regression_metrics(splits.y_test.values, pt_pred)

        # Calibration set is untouched by fitting and early-stopping model selection.
        pt_conformal = ConformalRegressor(pt_trainer, alpha=0.10)
        pt_conformal.calibrate(neural_arrays.X_cal, neural_arrays.y_cal)
        pt_intervals = pt_conformal.predict_interval(neural_arrays.X_test)
        pt_conf_eval = evaluate_conformal_coverage(pt_intervals, splits.y_test.values)

        pt_sex_disp = evaluate_subgroup_disparity(
            splits.y_test.values,
            pt_pred,
            splits.X_test["sex"],
            subgroup_name="sex",
        )
        pt_sex_p = permutation_test_disparity(
            splits.y_test.values,
            pt_pred,
            splits.X_test["sex"].values,
            random_state=random_state,
        )

        comparison_records.append({
            "model": "MLP (16, 8) - PyTorch",
            "cv_mae_mean": np.nan,
            "cv_mae_std": np.nan,
            "test_mae": round(pt_metrics["mae"], 3),
            "test_rmse": round(pt_metrics["rmse"], 3),
            "test_r2": round(pt_metrics["r2"], 3),
            "conformal_coverage_90": round(pt_conf_eval["empirical_coverage"], 3),
            "conformal_interval_width": round(pt_conf_eval["mean_interval_width"], 3),
            "sex_mae_disparity": round(pt_sex_disp["max_mae_disparity"], 3),
            "sex_disparity_p_val": round(pt_sex_p, 4),
        })

    # 3. TensorFlow Evaluation
    if TF_AVAILABLE:
        print("Training TensorFlow / Keras MLP with Early Stopping...")
        tf_trainer = TensorFlowTrainer(
            in_features=10,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            dropout_p=0.1,
            random_state=random_state,
        )
        tf_trainer.fit(
            neural_arrays.X_train,
            neural_arrays.y_train,
            neural_arrays.X_val,
            neural_arrays.y_val,
        )

        for rec in tf_trainer.history_records:
            histories.append({"framework": "TensorFlow", **rec})

        tf_pred = tf_trainer.predict(neural_arrays.X_test)
        tf_metrics = calculate_regression_metrics(splits.y_test.values, tf_pred)

        # Use the independent calibration set, never the early-stopping validation set.
        tf_conformal = ConformalRegressor(tf_trainer, alpha=0.10)
        tf_conformal.calibrate(neural_arrays.X_cal, neural_arrays.y_cal)
        tf_intervals = tf_conformal.predict_interval(neural_arrays.X_test)
        tf_conf_eval = evaluate_conformal_coverage(tf_intervals, splits.y_test.values)

        tf_sex_disp = evaluate_subgroup_disparity(
            splits.y_test.values,
            tf_pred,
            splits.X_test["sex"],
            subgroup_name="sex",
        )
        tf_sex_p = permutation_test_disparity(
            splits.y_test.values,
            tf_pred,
            splits.X_test["sex"].values,
            random_state=random_state,
        )

        comparison_records.append({
            "model": "MLP (16, 8) - TensorFlow",
            "cv_mae_mean": np.nan,
            "cv_mae_std": np.nan,
            "test_mae": round(tf_metrics["mae"], 3),
            "test_rmse": round(tf_metrics["rmse"], 3),
            "test_r2": round(tf_metrics["r2"], 3),
            "conformal_coverage_90": round(tf_conf_eval["empirical_coverage"], 3),
            "conformal_interval_width": round(tf_conf_eval["mean_interval_width"], 3),
            "sex_mae_disparity": round(tf_sex_disp["max_mae_disparity"], 3),
            "sex_disparity_p_val": round(tf_sex_p, 4),
        })

    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv(out_path / "model_comparison.csv", index=False)

    if histories:
        pd.DataFrame(histories).to_csv(out_path / "neural_training_history.csv", index=False)

    return df_comp


def main() -> None:
    parser = argparse.ArgumentParser(description="Full Classical vs Neural Benchmark with Uncertainty.")
    parser.add_argument("--output", default="results", help="Directory for output artifacts.")
    parser.add_argument("--epochs", type=int, default=200, help="Max neural training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    df = run_full_comparison_experiment(
        output_dir=args.output,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    print("\n=== Model Comparison & Conformal Uncertainty Metrics ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
