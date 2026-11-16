"""Single fixed-model, participant-held-out ShanghaiT2DM CGM forecast.

This module fits one HistGradientBoostingRegressor. It does not compare model
families. Windows are built within original Excel recording sessions only.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def split_participants(participant_keys: list[str], seed: int = 42) -> dict[str, str]:
    keys = sorted(set(participant_keys))
    if len(keys) < 3:
        raise ValueError("At least three participants are required for train/validation/test splits.")
    shuffled = keys.copy()
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(0.70 * n))
    n_validation = int(round(0.15 * n))
    if n_train < 1 or n_validation < 1 or n - n_train - n_validation < 1:
        raise ValueError("The participant count is too small for a non-empty 70/15/15 split.")
    result: dict[str, str] = {}
    for key in shuffled[:n_train]:
        result[key] = "train"
    for key in shuffled[n_train:n_train + n_validation]:
        result[key] = "validation"
    for key in shuffled[n_train + n_validation:]:
        result[key] = "test"
    return result

def build_session_windows(frame: pd.DataFrame, history_samples: int = 20, horizon_steps: int = 2, interval_minutes: int = 15) -> tuple[np.ndarray, np.ndarray]:
    timestamp_col = next((col for col in ("Date", "timestamp") if col in frame.columns), None)
    glucose_col = next((col for col in ("CGM (mg / dl)", "CGM ", "glucose_value_mg_dl") if col in frame.columns), None)
    if timestamp_col is None or glucose_col is None:
        raise ValueError("A source workbook is missing the expected timestamp or CGM column.")
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce", utc=True)
    glucose = pd.to_numeric(frame[glucose_col], errors="coerce")
    finite = glucose.map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
    valid = timestamps.notna() & finite
    clean = pd.DataFrame({"timestamp": timestamps[valid], "glucose": glucose[valid].astype(float)})
    clean = clean.sort_values("timestamp").reset_index(drop=True)
    if clean.empty:
        return np.empty((0, history_samples), dtype=float), np.empty((0,), dtype=float)
    time_ns = pd.DatetimeIndex(clean["timestamp"]).asi8
    glucose_values = clean["glucose"].to_numpy(dtype=float)
    expected_ns = int(interval_minutes * 60 * 1_000_000_000)
    features: list[np.ndarray] = []
    targets: list[float] = []
    first_target = history_samples - 1 + horizon_steps
    for target_index in range(first_target, len(clean)):
        start_index = target_index - horizon_steps - history_samples + 1
        time_window = time_ns[start_index:target_index + 1]
        if len(time_window) != history_samples + horizon_steps:
            continue
        if not np.all(np.diff(time_window) == expected_ns):
            continue
        input_end = target_index - horizon_steps
        features.append(glucose_values[start_index:input_end + 1])
        targets.append(float(glucose_values[target_index]))
    if not features:
        return np.empty((0, history_samples), dtype=float), np.empty((0,), dtype=float)
    return np.vstack(features), np.asarray(targets, dtype=float)

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_mg_dl": float(mean_absolute_error(y_true, y_pred)),
        "rmse_mg_dl": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }

def _bootstrap_macro_metrics(per_participant: dict[str, tuple[float, float]], seed: int, replicates: int = 1000) -> dict[str, Any]:
    keys = sorted(per_participant)
    if len(keys) < 2:
        raise ValueError("At least two held-out participants are required for cluster bootstrap uncertainty.")
    rng = random.Random(seed + 1)
    mae_samples: list[float] = []
    rmse_samples: list[float] = []
    for _ in range(replicates):
        selected = [rng.choice(keys) for _ in keys]
        mae_samples.append(float(np.mean([per_participant[key][0] for key in selected])))
        rmse_samples.append(float(np.mean([per_participant[key][1] for key in selected])))
    return {
        "method": "Percentile bootstrap resampling held-out participants as clusters; each replicate samples the same number of participants with replacement.",
        "replicates": replicates,
        "participant_clusters": len(keys),
        "macro_mae_95_percent_ci_mg_dl": [float(np.quantile(mae_samples, 0.025)), float(np.quantile(mae_samples, 0.975))],
        "macro_rmse_95_percent_ci_mg_dl": [float(np.quantile(rmse_samples, 0.025)), float(np.quantile(rmse_samples, 0.975))],
    }

def run_forecast(raw_dir: str | Path, output_path: str | Path, seed: int = 42, upstream_revision: str | None = None, bootstrap_replicates: int = 1000) -> dict[str, Any]:
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Shanghai raw workbook directory does not exist: {root}")
    workbooks = sorted(path for path in root.glob("*.xls*") if path.is_file() and not path.name.startswith("~$"))
    if not workbooks:
        raise ValueError(f"No Shanghai Excel workbooks found in: {root}")
    participant_keys = [path.name.split("_", 1)[0] for path in workbooks]
    split_map = split_participants(participant_keys, seed)
    X_parts: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    y_parts: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    participant_windows: dict[str, dict[str, list[np.ndarray]]] = {
        "validation": {}, "test": {}
    }
    session_count_by_split = {"train": 0, "validation": 0, "test": 0}
    session_window_count = {"train": 0, "validation": 0, "test": 0}
    for path in workbooks:
        participant = path.name.split("_", 1)[0]
        split = split_map[participant]
        frame = pd.read_excel(path)
        X_session, y_session = build_session_windows(frame)
        session_count_by_split[split] += 1
        session_window_count[split] += len(y_session)
        if len(y_session) == 0:
            continue
        X_parts[split].append(X_session)
        y_parts[split].append(y_session)
        if split in ("validation", "test"):
            participant_windows[split].setdefault(participant, []).append(y_session)

    combined: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        if not X_parts[split]:
            raise ValueError(f"No valid forecasting windows were generated for the {split} participant split.")
        combined[split] = (np.vstack(X_parts[split]), np.concatenate(y_parts[split]))

    X_train, y_train = combined["train"]
    X_validation, y_validation = combined["validation"]
    X_test, y_test = combined["test"]
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    validation_predictions = model.predict(X_validation)
    test_predictions = model.predict(X_test)

    test_group_metrics: dict[str, tuple[float, float]] = {}
    test_group_errors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cursor = 0
    test_participant_predictions: dict[str, np.ndarray] = {}
    # Reconstruct test predictions by participant using the same session traversal order.
    for participant in sorted(participant_windows["test"]):
        y_values = np.concatenate(participant_windows["test"][participant])
        test_participant_predictions[participant] = y_values
    # Build participant-aligned predictions in a second deterministic pass.
    test_pred_parts: dict[str, list[np.ndarray]] = {}
    for path in workbooks:
        participant = path.name.split("_", 1)[0]
        if split_map[participant] != "test":
            continue
        frame = pd.read_excel(path)
        X_session, y_session = build_session_windows(frame)
        if len(y_session):
            pred_session = model.predict(X_session)
            test_pred_parts.setdefault(participant, []).append(pred_session)
    for participant, true_parts in participant_windows["test"].items():
        y_participant = np.concatenate(true_parts)
        pred_participant = np.concatenate(test_pred_parts[participant])
        test_group_metrics[participant] = (
            float(mean_absolute_error(y_participant, pred_participant)),
            float(math.sqrt(mean_squared_error(y_participant, pred_participant))),
        )
        test_group_errors[participant] = (y_participant, pred_participant)

    def macro(values: list[float]) -> dict[str, float]:
        return {"mean": float(np.mean(values)), "median": float(np.median(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}

    test_macro_mae = [value[0] for value in test_group_metrics.values()]
    test_macro_rmse = [value[1] for value in test_group_metrics.values()]
    validation_macro: dict[str, tuple[float, float]] = {}
    validation_pred_parts: dict[str, list[np.ndarray]] = {}
    for path in workbooks:
        participant = path.name.split("_", 1)[0]
        if split_map[participant] != "validation":
            continue
        frame = pd.read_excel(path)
        X_session, y_session = build_session_windows(frame)
        if len(y_session):
            validation_pred_parts.setdefault(participant, []).append(model.predict(X_session))
    for participant, true_parts in participant_windows["validation"].items():
        y_participant = np.concatenate(true_parts)
        pred_participant = np.concatenate(validation_pred_parts[participant])
        validation_macro[participant] = (
            float(mean_absolute_error(y_participant, pred_participant)),
            float(math.sqrt(mean_squared_error(y_participant, pred_participant))),
        )

    result = {
        "schema_version": 1,
        "dataset": "shanghai_t2dm",
        "task": "30-minute-ahead CGM glucose forecasting",
        "upstream_glucose_ml_revision": upstream_revision,
        "source_workbooks": len(workbooks),
        "participants": len(split_map),
        "window_specification": {
            "history_samples": 20,
            "history_elapsed_span_minutes": 285,
            "sampling_interval_minutes": 15,
            "forecast_horizon_minutes": 30,
            "feature_definition": "The 20 most recent glucose readings within one source workbook/session, ending at the forecast origin.",
            "target_definition": "Glucose reading exactly 30 minutes after the forecast origin.",
            "window_rule": "All timestamps from the first history reading through the target must be exactly 15 minutes apart. Windows never cross source workbook/session boundaries. Missing values and irregular intervals break windows; no interpolation is performed.",
        },
        "participant_split": {
            "seed": seed,
            "train_participants": sum(value == "train" for value in split_map.values()),
            "validation_participants": sum(value == "validation" for value in split_map.values()),
            "test_participants": sum(value == "test" for value in split_map.values()),
            "train_sessions": session_count_by_split["train"],
            "validation_sessions": session_count_by_split["validation"],
            "test_sessions": session_count_by_split["test"],
            "train_windows": len(y_train),
            "validation_windows": len(y_validation),
            "test_windows": len(y_test),
            "all_sessions_for_a_participant_stay_in_one_split": True,
        },
        "model": {
            "name": "HistGradientBoostingRegressor",
            "parameters": {
                "max_iter": 150,
                "learning_rate": 0.05,
                "max_leaf_nodes": 15,
                "l2_regularization": 1.0,
                "early_stopping": False,
                "random_state": seed,
            },
            "selection_policy": "One predeclared model and fixed hyperparameters. No model-family comparison and no hyperparameter search. Validation data are reported but not used for fitting or selection.",
            "features": "CGM lag values only; no demographics, medication variables, or future information.",
        },
        "validation_metrics_pooled_windows": _metrics(y_validation, validation_predictions),
        "validation_participant_macro_mae_mg_dl": macro([value[0] for value in validation_macro.values()]),
        "validation_participant_macro_rmse_mg_dl": macro([value[1] for value in validation_macro.values()]),
        "test_metrics_pooled_windows": _metrics(y_test, test_predictions),
        "test_participant_macro_mae_mg_dl": macro(test_macro_mae),
        "test_participant_macro_rmse_mg_dl": macro(test_macro_rmse),
        "test_participant_cluster_bootstrap": _bootstrap_macro_metrics(test_group_metrics, seed, bootstrap_replicates),
        "interpretation_limits": [
            "This is internal participant-held-out evaluation on one open ShanghaiT2DM cohort, not external validation.",
            "Overlapping windows are correlated; uncertainty is bootstrapped by participant, not by window.",
            "This predicts short-horizon CGM glucose, not long-term diabetes progression, diagnosis, complications, treatment response, or clinical outcomes.",
            "No clinical utility or patient-level decision support is established.",
        ],
        "privacy_policy": "Aggregate-only metrics; no participant identifiers, filenames, predictions, raw glucose values, or participant-level metric rows are emitted.",
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one fixed ShanghaiT2DM CGM forecasting model with participant-held-out evaluation.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upstream-revision", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("Use at least 100 participant-cluster bootstrap replicates.")
    result = run_forecast(args.raw_dir, args.output, args.seed, args.upstream_revision, args.bootstrap_replicates)
    print(f"Trained one fixed model; aggregate metrics written to {args.output}")
    print(f"Train/validation/test windows: {result['participant_split']['train_windows']}/{result['participant_split']['validation_windows']}/{result['participant_split']['test_windows']}")

if __name__ == "__main__":
    main()
