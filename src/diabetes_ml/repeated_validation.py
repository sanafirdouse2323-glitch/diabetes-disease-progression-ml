"""Repeated outer-split validation for classical models and split-conformal intervals.

The reported across-seed standard deviations describe sensitivity to the chosen
random split. Repeated hold-out sets overlap, so these summaries are not
independent-sample confidence intervals and are not external validation.
"""

from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from diabetes_ml.data import create_experiment_splits
from diabetes_ml.evaluation import (
    ConformalRegressor,
    calculate_regression_metrics,
    evaluate_conformal_coverage,
)
from diabetes_ml.models import evaluate_cv, get_classical_models


DEFAULT_SEEDS = tuple(range(42, 52))


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Return at least two unique integer seeds, rejecting booleans."""
    if isinstance(seeds, (str, bytes)):
        raise ValueError("seeds must be a sequence of at least two unique integers.")
    try:
        values = tuple(seeds)
    except TypeError as exc:
        raise ValueError("seeds must be a sequence of at least two unique integers.") from exc
    if len(values) < 2:
        raise ValueError("At least two seeds are required to summarize split variability.")
    if any(isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) for seed in values):
        raise ValueError("Every seed must be an integer; booleans are not valid seeds.")
    normalized = tuple(int(seed) for seed in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("seeds must be unique.")
    return normalized


def _validated_fraction(value: float, name: str) -> float:
    """Validate a strict fractional split parameter."""
    fraction = float(value)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1.")
    return fraction


def summarize_seed_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric metrics by model across complete, unique seed runs.

    Standard deviations use the sample definition (ddof=1). These are
    descriptive split-sensitivity summaries, not confidence intervals.
    """
    required = {"seed", "model"}
    if not required.issubset(per_seed.columns):
        raise ValueError("per_seed must contain 'seed' and 'model' columns.")
    if per_seed.empty:
        raise ValueError("per_seed must not be empty.")
    if per_seed[["seed", "model"]].isna().any().any():
        raise ValueError("seed and model identifiers must not contain missing values.")
    if per_seed.duplicated(["seed", "model"]).any():
        raise ValueError("Each seed/model combination must occur exactly once.")

    seed_sets = per_seed.groupby("model", sort=False)["seed"].apply(lambda values: set(values))
    expected_seeds = set(per_seed["seed"])
    if any(values != expected_seeds for values in seed_sets):
        raise ValueError("Every model must have results for the same set of seeds.")

    metric_columns = [
        column for column in per_seed.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    if not metric_columns:
        raise ValueError("per_seed must contain at least one numeric metric column.")
    metrics = per_seed[metric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(metrics).all():
        raise ValueError("All metric values must be finite before summarization.")

    summary = (
        per_seed.groupby("model", sort=False)[metric_columns]
        .agg(["count", "mean", "std", "median", "min", "max"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def run_repeated_seed_validation(
    output_dir: str | Path = "repeated-validation",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    test_size: float = 0.20,
    cal_fraction: float = 0.20,
    alpha: float = 0.10,
    cv_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run identical classical benchmark procedures over multiple outer seeds.

    Each seed creates a new outer train/test split. Calibration observations
    are then separated from the outer training partition; cross-validation and
    estimator fitting use only the remaining fitting partition. The test set
    is used only for final metrics and is not used for fitting or calibration.
    """
    seed_values = _validated_seeds(seeds)
    test_fraction = _validated_fraction(test_size, "test_size")
    calibration_fraction = _validated_fraction(cal_fraction, "cal_fraction")
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")
    if isinstance(cv_folds, (bool, np.bool_)) or not isinstance(cv_folds, Integral) or cv_folds < 2:
        raise ValueError("cv_folds must be an integer greater than or equal to 2.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | int | str]] = []

    for seed in seed_values:
        splits = create_experiment_splits(
            test_size=test_fraction,
            random_state=seed,
        )
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            splits.X_train,
            splits.y_train,
            test_size=calibration_fraction,
            random_state=seed,
            shuffle=True,
        )
        if len(X_fit) < cv_folds:
            raise ValueError(
                f"Seed {seed} leaves only {len(X_fit)} fitting rows, fewer than cv_folds={cv_folds}."
            )

        models = get_classical_models(random_state=seed)
        for model_name, estimator in models.items():
            cv = evaluate_cv(
                estimator,
                X_fit,
                y_fit,
                cv_folds=int(cv_folds),
                random_state=seed,
            )
            conformal = ConformalRegressor(estimator, alpha=alpha_value)
            conformal.fit_and_calibrate(
                X_fit.to_numpy(),
                y_fit.to_numpy(),
                X_cal.to_numpy(),
                y_cal.to_numpy(),
            )
            intervals = conformal.predict_interval(splits.X_test.to_numpy())
            point_metrics = calculate_regression_metrics(
                splits.y_test.to_numpy(),
                intervals.point_prediction,
            )
            interval_metrics = evaluate_conformal_coverage(
                intervals,
                splits.y_test.to_numpy(),
            )
            records.append({
                "seed": seed,
                "model": model_name,
                "fit_n": int(len(X_fit)),
                "calibration_n": int(len(X_cal)),
                "test_n": int(len(splits.X_test)),
                "cv_mae_mean": cv["cv_mae_mean"],
                "cv_mae_std": cv["cv_mae_std"],
                "test_mae": point_metrics["mae"],
                "test_rmse": point_metrics["rmse"],
                "test_r2": point_metrics["r2"],
                "conformal_nominal_coverage": interval_metrics["nominal_coverage"],
                "conformal_empirical_coverage": interval_metrics["empirical_coverage"],
                "conformal_mean_interval_width": interval_metrics["mean_interval_width"],
                "conformal_winkler_score": interval_metrics["winkler_score"],
            })

    per_seed = pd.DataFrame.from_records(records)
    summary = summarize_seed_metrics(per_seed)
    per_seed.to_csv(out_path / "per_seed_metrics.csv", index=False)
    summary.to_csv(out_path / "summary_metrics.csv", index=False)

    metadata = {
        "dataset": "sklearn.datasets.load_diabetes",
        "observations": 442,
        "features": 10,
        "seeds": list(seed_values),
        "n_seeds": len(seed_values),
        "outer_test_fraction": test_fraction,
        "calibration_fraction_of_outer_training": calibration_fraction,
        "conformal_alpha": alpha_value,
        "nominal_conformal_coverage": 1.0 - alpha_value,
        "cross_validation_folds_within_fit_partition": int(cv_folds),
        "models": list(get_classical_models(random_state=seed_values[0]).keys()),
        "summary_standard_deviation": "sample standard deviation (ddof=1)",
        "interpretation": (
            "Across-seed summaries describe sensitivity to random hold-out splits. "
            "The repeated test partitions overlap; standard deviations are not "
            "independent-sample confidence intervals and do not constitute external validation."
        ),
        "files": ["per_seed_metrics.csv", "summary_metrics.csv"],
    }
    with (out_path / "validation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return per_seed, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repeated-seed validation of classical diabetes regression models."
    )
    parser.add_argument("--output", default="repeated-validation", help="Directory for validation outputs.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="At least two unique integer seeds (default: 42 through 51).",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cal-fraction", type=float, default=0.20)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    per_seed, summary = run_repeated_seed_validation(
        output_dir=args.output,
        seeds=args.seeds,
        test_size=args.test_size,
        cal_fraction=args.cal_fraction,
        alpha=args.alpha,
        cv_folds=args.cv_folds,
    )
    print(f"Completed {per_seed['seed'].nunique()} seeds.")
    print("\n=== Across-seed descriptive summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
