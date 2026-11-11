"""Repeated-seed validation for PyTorch and TensorFlow regressors.

Each seed uses disjoint model-fitting, early-stopping validation, conformal
calibration, and outer test partitions. Across-seed standard deviations are
descriptive split-sensitivity summaries, not confidence intervals.
"""

from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from diabetes_ml.data import create_experiment_splits
from diabetes_ml.evaluation import (
    ConformalRegressor,
    calculate_regression_metrics,
    evaluate_conformal_coverage,
)
from diabetes_ml.neural.neural_data import prepare_neural_datasets
from diabetes_ml.neural.pytorch_model import TORCH_AVAILABLE, PyTorchTrainer
from diabetes_ml.neural.tensorflow_model import TF_AVAILABLE, TensorFlowTrainer
from diabetes_ml.repeated_validation import (
    DEFAULT_SEEDS,
    _validated_fraction,
    _validated_seeds,
    summarize_seed_metrics,
)


def _validated_training_parameters(
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    dropout_p: float,
) -> tuple[int, int, int, float, float]:
    """Validate positive training controls and a valid dropout probability."""
    integer_values = {
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
    }
    for name, value in integer_values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    lr = float(learning_rate)
    dropout = float(dropout_p)
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError("learning_rate must be finite and greater than zero.")
    if not np.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("dropout_p must be finite and in the interval [0, 1).")
    return int(epochs), int(patience), int(batch_size), lr, dropout


def run_repeated_neural_validation(
    output_dir: str | Path = "repeated-neural-validation",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    test_size: float = 0.20,
    cal_fraction: float = 0.20,
    val_fraction: float = 0.20,
    alpha: float = 0.10,
    epochs: int = 100,
    patience: int = 12,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    dropout_p: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate both neural frameworks over repeated, leakage-safe outer splits."""
    seed_values = _validated_seeds(seeds)
    test_fraction = _validated_fraction(test_size, "test_size")
    calibration_fraction = _validated_fraction(cal_fraction, "cal_fraction")
    validation_fraction = _validated_fraction(val_fraction, "val_fraction")
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")
    epochs_value, patience_value, batch_value, lr_value, dropout_value = (
        _validated_training_parameters(
            epochs,
            patience,
            batch_size,
            learning_rate,
            dropout_p,
        )
    )

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install the neural dependencies before running this workflow.")
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required. Install the neural dependencies before running this workflow.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metric_records: list[dict[str, float | int | str]] = []
    history_records: list[dict[str, float | int | str]] = []

    for seed in seed_values:
        splits = create_experiment_splits(
            test_size=test_fraction,
            random_state=seed,
        )
        arrays = prepare_neural_datasets(
            splits.X_train,
            splits.y_train,
            splits.X_test,
            splits.y_test,
            val_fraction=validation_fraction,
            cal_fraction=calibration_fraction,
            random_state=seed,
        )

        trainers = [
            (
                "PyTorch MLP",
                PyTorchTrainer(
                    in_features=arrays.X_train.shape[1],
                    lr=lr_value,
                    batch_size=batch_value,
                    epochs=epochs_value,
                    patience=patience_value,
                    dropout_p=dropout_value,
                    random_state=seed,
                ),
            ),
            (
                "TensorFlow MLP",
                TensorFlowTrainer(
                    in_features=arrays.X_train.shape[1],
                    lr=lr_value,
                    batch_size=batch_value,
                    epochs=epochs_value,
                    patience=patience_value,
                    dropout_p=dropout_value,
                    random_state=seed,
                ),
            ),
        ]

        for model_name, trainer in trainers:
            trainer.fit(
                arrays.X_train,
                arrays.y_train,
                arrays.X_val,
                arrays.y_val,
            )
            history = (
                trainer.history
                if model_name == "PyTorch MLP"
                else trainer.history_records
            )
            if not history:
                raise RuntimeError(f"{model_name} produced no training history for seed {seed}.")
            for row in history:
                history_records.append({
                    "seed": seed,
                    "model": model_name,
                    "epoch": int(row["epoch"]),
                    "train_mse": float(row["train_mse"]),
                    "val_mse": float(row["val_mse"]),
                })

            conformal = ConformalRegressor(trainer, alpha=alpha_value)
            conformal.calibrate(arrays.X_cal, arrays.y_cal)
            intervals = conformal.predict_interval(arrays.X_test)
            point_metrics = calculate_regression_metrics(
                splits.y_test.to_numpy(),
                intervals.point_prediction,
            )
            interval_metrics = evaluate_conformal_coverage(
                intervals,
                splits.y_test.to_numpy(),
            )
            best_epoch_record = min(history, key=lambda row: row["val_mse"])
            metric_records.append({
                "seed": seed,
                "model": model_name,
                "fit_n": int(len(arrays.X_train)),
                "validation_n": int(len(arrays.X_val)),
                "calibration_n": int(len(arrays.X_cal)),
                "test_n": int(len(arrays.X_test)),
                "epochs_trained": int(len(history)),
                "best_validation_epoch": int(best_epoch_record["epoch"]),
                "best_validation_mse": float(best_epoch_record["val_mse"]),
                "test_mae": point_metrics["mae"],
                "test_rmse": point_metrics["rmse"],
                "test_r2": point_metrics["r2"],
                "conformal_nominal_coverage": interval_metrics["nominal_coverage"],
                "conformal_empirical_coverage": interval_metrics["empirical_coverage"],
                "conformal_mean_interval_width": interval_metrics["mean_interval_width"],
                "conformal_winkler_score": interval_metrics["winkler_score"],
            })

    per_seed = pd.DataFrame.from_records(metric_records)
    summary = summarize_seed_metrics(per_seed)
    history_frame = pd.DataFrame.from_records(history_records)
    per_seed.to_csv(out_path / "per_seed_neural_metrics.csv", index=False)
    summary.to_csv(out_path / "summary_neural_metrics.csv", index=False)
    history_frame.to_csv(out_path / "neural_training_history.csv", index=False)

    metadata = {
        "dataset": "sklearn.datasets.load_diabetes",
        "observations": 442,
        "features": 10,
        "seeds": list(seed_values),
        "n_seeds": len(seed_values),
        "outer_test_fraction": test_fraction,
        "calibration_fraction_of_outer_training": calibration_fraction,
        "validation_fraction_of_non_calibration_development_data": validation_fraction,
        "conformal_alpha": alpha_value,
        "nominal_conformal_coverage": 1.0 - alpha_value,
        "training": {
            "epochs": epochs_value,
            "patience": patience_value,
            "batch_size": batch_value,
            "learning_rate": lr_value,
            "dropout_probability": dropout_value,
        },
        "models": ["PyTorch MLP", "TensorFlow MLP"],
        "partition_design": (
            "For each seed, calibration is separated from outer training first. "
            "The remaining development subset is split into model fitting and "
            "early-stopping validation. The feature scaler is fitted only on "
            "the model-fitting subset; the outer test subset is held out."
        ),
        "summary_standard_deviation": "sample standard deviation (ddof=1)",
        "interpretation": (
            "Across-seed summaries describe sensitivity to random hold-out splits. "
            "The repeated test partitions overlap; standard deviations are not "
            "independent-sample confidence intervals and do not constitute external validation."
        ),
        "files": [
            "per_seed_neural_metrics.csv",
            "summary_neural_metrics.csv",
            "neural_training_history.csv",
        ],
    }
    with (out_path / "validation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return per_seed, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repeated-seed validation of PyTorch and TensorFlow diabetes regressors."
    )
    parser.add_argument("--output", default="repeated-neural-validation")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cal-fraction", type=float, default=0.20)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.10)
    args = parser.parse_args()

    per_seed, summary = run_repeated_neural_validation(
        output_dir=args.output,
        seeds=args.seeds,
        test_size=args.test_size,
        cal_fraction=args.cal_fraction,
        val_fraction=args.val_fraction,
        alpha=args.alpha,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_p=args.dropout,
    )
    print(f"Completed {per_seed['seed'].nunique()} seeds for both neural frameworks.")
    print("\n=== Across-seed neural validation summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
