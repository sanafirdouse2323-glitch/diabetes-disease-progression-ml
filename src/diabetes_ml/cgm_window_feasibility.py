"""Session-preserving feasibility counts for a fixed ShanghaiT2DM CGM forecast task.

This module counts windows only; it does not fit or evaluate a predictive model.
Participant identifiers are used internally for grouping and are never emitted.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any
import pandas as pd

def _median(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(clean)) if clean else None

def _session_window_count(path: Path, history_samples: int, horizon_steps: int, interval_minutes: int) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    frame = pd.read_excel(path)
    timestamp_col = next((col for col in ("Date", "timestamp") if col in frame.columns), None)
    glucose_col = next((col for col in ("CGM (mg / dl)", "CGM ", "glucose_value_mg_dl") if col in frame.columns), None)
    if timestamp_col is None or glucose_col is None:
        raise ValueError("A source workbook is missing the expected timestamp or CGM column.")
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce", utc=True)
    glucose = pd.to_numeric(frame[glucose_col], errors="coerce")
    finite = glucose.map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
    valid = timestamps.notna() & finite
    series = pd.DataFrame({"timestamp": timestamps[valid], "glucose": glucose[valid].astype(float)})
    series = series.sort_values("timestamp").reset_index(drop=True)
    if series.empty:
        return 0, None, None
    times = series["timestamp"]
    expected = pd.Timedelta(minutes=interval_minutes)
    windows = 0
    first_target = history_samples - 1 + horizon_steps
    for target_index in range(first_target, len(series)):
        start_index = target_index - horizon_steps - history_samples + 1
        window_times = times.iloc[start_index:target_index + 1]
        if len(window_times) != history_samples + horizon_steps:
            continue
        deltas = window_times.diff().iloc[1:]
        if bool(deltas.eq(expected).all()):
            windows += 1
    return windows, times.iloc[0], times.iloc[-1]

def audit_window_feasibility(raw_dir: str | Path, history_samples: int = 20, horizon_minutes: int = 30, interval_minutes: int = 15, seed: int = 42, upstream_revision: str | None = None) -> dict[str, Any]:
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Shanghai raw workbook directory does not exist: {root}")
    workbooks = sorted(path for path in root.glob("*.xls*") if path.is_file() and not path.name.startswith("~$"))
    if not workbooks:
        raise ValueError(f"No Shanghai Excel workbooks found in: {root}")
    if horizon_minutes % interval_minutes:
        raise ValueError("Forecast horizon must be an integer multiple of the sampling interval.")
    horizon_steps = horizon_minutes // interval_minutes
    participant_windows: dict[str, int] = {}
    participant_sessions: dict[str, int] = {}
    session_windows: list[int] = []
    session_ranges: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for path in workbooks:
        participant_key = path.name.split("_", 1)[0]
        count, start, end = _session_window_count(path, history_samples, horizon_steps, interval_minutes)
        participant_windows[participant_key] = participant_windows.get(participant_key, 0) + count
        participant_sessions[participant_key] = participant_sessions.get(participant_key, 0) + 1
        session_windows.append(count)
        if start is not None and end is not None:
            session_ranges.setdefault(participant_key, []).append((start, end))
    participants = sorted(participant_windows)
    shuffled = participants.copy()
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(0.70 * n))
    n_validation = int(round(0.15 * n))
    train_ids = set(shuffled[:n_train])
    validation_ids = set(shuffled[n_train:n_train + n_validation])
    test_ids = set(shuffled[n_train + n_validation:])
    overlap_pairs = 0
    for ranges in session_ranges.values():
        ordered = sorted(ranges, key=lambda item: item[0])
        for index in range(1, len(ordered)):
            if ordered[index][0] <= ordered[index - 1][1]:
                overlap_pairs += 1
    total_windows = sum(participant_windows.values())
    return {
        "schema_version": 1,
        "dataset": "shanghai_t2dm",
        "upstream_glucose_ml_revision": upstream_revision,
        "forecast_feasibility_only": True,
        "history_samples": history_samples,
        "history_elapsed_span_minutes": (history_samples - 1) * interval_minutes,
        "sampling_interval_minutes": interval_minutes,
        "forecast_horizon_minutes": horizon_minutes,
        "forecast_horizon_steps": horizon_steps,
        "window_rule": "A window is counted only when all input and future timestamps within the source workbook are separated by exactly the configured sampling interval. Windows never cross workbook/session boundaries. Missing glucose rows, duplicate timestamps, and irregular gaps therefore break windows; no interpolation or imputation is performed.",
        "source_workbooks": len(workbooks),
        "participant_filename_prefixes": n,
        "participants_with_multiple_sessions": sum(value > 1 for value in participant_sessions.values()),
        "session_count": len(workbooks),
        "sessions_with_at_least_one_valid_window": sum(value > 0 for value in session_windows),
        "total_valid_windows": total_windows,
        "participants_with_at_least_one_valid_window": sum(value > 0 for value in participant_windows.values()),
        "participants_with_zero_valid_windows": sum(value == 0 for value in participant_windows.values()),
        "median_windows_per_participant_including_zero": _median([float(value) for value in participant_windows.values()]),
        "median_windows_per_session_including_zero": _median([float(value) for value in session_windows]),
        "participant_overlap_pairs_between_recording_sessions": overlap_pairs,
        "split_seed": seed,
        "participant_split": {
            "train_participants": len(train_ids),
            "validation_participants": len(validation_ids),
            "test_participants": len(test_ids),
            "train_windows": sum(participant_windows[key] for key in train_ids),
            "validation_windows": sum(participant_windows[key] for key in validation_ids),
            "test_windows": sum(participant_windows[key] for key in test_ids),
        },
        "split_policy": "70/15/15 participant-level split with a fixed seed. Every recording session belonging to a participant stays in that participant's split; no window-level random split is used.",
        "glucose_policy": "All finite source glucose values are retained for feasibility counting, including readings outside 40–400 mg/dL. No values are removed, clipped, or imputed by this feasibility step.",
        "privacy_policy": "Aggregate-only output; participant identifiers, filenames, per-participant window counts, and raw readings are not emitted.",
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Count session-preserving ShanghaiT2DM forecast windows without training a model.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--history-samples", type=int, default=20)
    parser.add_argument("--horizon-minutes", type=int, default=30)
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upstream-revision", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.history_samples < 1 or args.interval_minutes < 1 or args.horizon_minutes < 1:
        parser.error("History, horizon and interval must be positive integers.")
    report = audit_window_feasibility(args.raw_dir, args.history_samples, args.horizon_minutes, args.interval_minutes, args.seed, args.upstream_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote aggregate Shanghai forecast-window feasibility report: {args.output}")

if __name__ == "__main__":
    main()
