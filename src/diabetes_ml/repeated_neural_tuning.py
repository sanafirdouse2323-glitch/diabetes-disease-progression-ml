"""Validation-selected PyTorch hyperparameter study with untouched test data.

For each outer seed, candidate settings are ranked using validation MSE only.
The selected model is then calibrated on a disjoint calibration subset and
evaluated once on the outer test subset. Across-seed SDs are descriptive:
outer test partitions overlap and are not independent confidence intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from diabetes_ml.repeated_validation import DEFAULT_SEEDS, _validated_seeds, summarize_seed_metrics


# Predeclared settings are compared on validation MSE in original target units.
_BASE_CONFIGS = (
    {"config": "baseline_default", "learning_rate": 0.001, "dropout_p": 0.10, "batch_size": 32},
    {"config": "lower_learning_rate", "learning_rate": 0.0003, "dropout_p": 0.10, "batch_size": 32},
    {"config": "higher_learning_rate", "learning_rate": 0.003, "dropout_p": 0.10, "batch_size": 32},
    {"config": "higher_dropout", "learning_rate": 0.001, "dropout_p": 0.30, "batch_size": 32},
    {"config": "no_dropout", "learning_rate": 0.001, "dropout_p": 0.00, "batch_size": 32},
)
CANDIDATE_CONFIGS = tuple(
    {**cfg, "standardize_target": False} for cfg in _BASE_CONFIGS
) + tuple(
    {**cfg, "config": cfg["config"] + "_target_scaled", "standardize_target": True}
    for cfg in _BASE_CONFIGS
)


def run_repeated_neural_tuning(
    output_dir: str | Path = "repeated-neural-tuning",
    seeds=DEFAULT_SEEDS,
    epochs: int = 300,
    patience: int = 25,
    alpha: float = 0.10,
):
    """Select one predeclared PyTorch configuration per seed on validation MSE."""
    seed_values = _validated_seeds(seeds)
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required; install the neural optional dependencies.")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise ValueError("patience must be a positive integer.")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate_rows, selected_rows, history_rows = [], [], []

    for seed in seed_values:
        split = create_experiment_splits(test_size=0.20, random_state=seed)
        data = prepare_neural_datasets(
            split.X_train, split.y_train, split.X_test, split.y_test,
            val_fraction=0.20, cal_fraction=0.20, random_state=seed,
        )
        best = None
        baseline_trainer = None
        for cfg in CANDIDATE_CONFIGS:
            trainer = PyTorchTrainer(
                in_features=data.X_train.shape[1],
                lr=cfg["learning_rate"],
                batch_size=cfg["batch_size"],
                epochs=epochs,
                patience=patience,
                dropout_p=cfg["dropout_p"],
                random_state=seed,
                standardize_target=cfg["standardize_target"],
            )
            trainer.fit(data.X_train, data.y_train, data.X_val, data.y_val)
            if not trainer.history:
                raise RuntimeError(f"No training history for seed={seed}, config={cfg['config']}.")
            best_epoch = min(trainer.history, key=lambda row: row["val_mse"])
            val_mse = float(best_epoch["val_mse"])
            candidate_rows.append({
                "seed": seed, **cfg, "epochs_trained": len(trainer.history),
                "best_validation_epoch": int(best_epoch["epoch"]),
                "best_validation_mse": val_mse,
            })
            for row in trainer.history:
                history_rows.append({
                    "seed": seed, "config": cfg["config"],
                    "epoch": int(row["epoch"]),
                    "train_mse": float(row["train_mse"]),
                    "validation_mse": float(row["val_mse"]),
                })
            if cfg["config"] == "baseline_default":
                baseline_trainer = trainer
            if best is None or val_mse < best["validation_mse"]:
                best = {"trainer": trainer, "config": cfg, "validation_mse": val_mse,
                        "best_epoch": int(best_epoch["epoch"])}

        if baseline_trainer is None:
            raise RuntimeError(f"Baseline control was not trained for seed={seed}.")
        # The baseline control is evaluated on the same outer test rows for a
        # paired descriptive comparison; these test metrics never affect selection.
        baseline_prediction = baseline_trainer.predict(data.X_test)
        baseline_point = calculate_regression_metrics(split.y_test.to_numpy(), baseline_prediction)

        # Calibration is used only after validation-based configuration choice.
        conformal = ConformalRegressor(best["trainer"], alpha=alpha)
        conformal.calibrate(data.X_cal, data.y_cal)
        intervals = conformal.predict_interval(data.X_test)
        point = calculate_regression_metrics(split.y_test.to_numpy(), intervals.point_prediction)
        interval = evaluate_conformal_coverage(intervals, split.y_test.to_numpy())
        selected_rows.append({
            "seed": seed, "selected_config": best["config"]["config"],
            "learning_rate": best["config"]["learning_rate"],
            "dropout_p": best["config"]["dropout_p"],
            "batch_size": best["config"]["batch_size"],
            "standardize_target": best["config"]["standardize_target"],
            "best_validation_epoch": best["best_epoch"],
            "selected_validation_mse": best["validation_mse"],
            "fit_n": len(data.X_train), "validation_n": len(data.X_val),
            "calibration_n": len(data.X_cal), "test_n": len(data.X_test),
            "test_mae": point["mae"], "test_rmse": point["rmse"], "test_r2": point["r2"],
            "baseline_test_mae": baseline_point["mae"],
            "baseline_test_rmse": baseline_point["rmse"],
            "baseline_test_r2": baseline_point["r2"],
            "mae_change_vs_baseline": point["mae"] - baseline_point["mae"],
            "conformal_nominal_coverage": interval["nominal_coverage"],
            "conformal_empirical_coverage": interval["empirical_coverage"],
            "conformal_mean_interval_width": interval["mean_interval_width"],
            "conformal_winkler_score": interval["winkler_score"],
        })

    candidates = pd.DataFrame(candidate_rows)
    selected = pd.DataFrame(selected_rows)
    summary = summarize_seed_metrics(selected.rename(columns={"selected_config": "model"}).assign(model="Validation-selected PyTorch MLP"))
    candidates.to_csv(out / "candidate_validation_metrics.csv", index=False)
    selected.to_csv(out / "selected_model_per_seed.csv", index=False)
    summary.to_csv(out / "selected_model_summary.csv", index=False)
    pd.DataFrame(history_rows).to_csv(out / "candidate_training_history.csv", index=False)
    metadata = {
        "dataset": "sklearn.datasets.load_diabetes", "observations": 442, "features": 10,
        "seeds": list(seed_values), "n_seeds": len(seed_values),
        "outer_test_fraction": 0.20, "calibration_fraction_of_outer_training": 0.20,
        "validation_fraction_of_non_calibration_development_data": 0.20,
        "candidate_configurations": list(CANDIDATE_CONFIGS),
        "selection_metric": "minimum validation mean squared error",
        "selection_scope": "configuration selected separately within each outer seed using validation data only",
        "calibration_use": "only after configuration selection; not used for selection or fitting",
        "test_use": "final evaluation only; not used for configuration selection",
        "epochs": epochs, "patience": patience, "alpha": alpha,
        "target_scaling": "optional; when enabled, mean and standard deviation are fitted on model-fitting targets only; validation MSE is recorded in original target units",
        "summary_sd": "sample standard deviation across overlapping repeated hold-out splits; descriptive, not a confidence interval",
        "limitations": [
            "The search tests a limited set of optimizer/dropout settings and target-scaling variants for one fixed MLP architecture.",
            "Validation-based selection can be noisy with a small validation subset; the outer test comparison estimates the full selection procedure.",
            "The default baseline is also evaluated on each same outer test subset for a paired descriptive comparison; its test metrics do not influence selection.",
            "Repeated outer test partitions overlap and are not external validation.",
            "The experiment does not establish clinical utility or universal neural superiority."
        ],
    }
    (out / "validation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return candidates, selected, summary


def main():
    parser = argparse.ArgumentParser(description="Run validation-selected repeated-seed PyTorch tuning.")
    parser.add_argument("--output", default="repeated-neural-tuning-results")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()
    candidates, selected, summary = run_repeated_neural_tuning(
        args.output, args.seeds, args.epochs, args.patience, args.alpha
    )
    print(f"Completed {selected['seed'].nunique()} outer seeds and {len(candidates)} candidate fits.")
    print("\nSelected configuration counts:")
    print(selected["selected_config"].value_counts().to_string())
    print("\nSelected-model across-seed summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
