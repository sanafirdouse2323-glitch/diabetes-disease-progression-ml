"""Aggregate-only quality audit for Glucose-ML harmonized CGM CSV files.

This module intentionally emits dataset-level summaries only. It never writes
participant identifiers, participant-level metrics, or raw CGM observations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any
import pandas as pd

TIMESTAMP_COLUMNS = ("timestamp", "_timestamp_")
GLUCOSE_COLUMN = "glucose_value_mg_dl"

def _median(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(clean)) if clean else None

def _participant_file_summary(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    timestamp_column = next((name for name in TIMESTAMP_COLUMNS if name in frame.columns), None)
    if timestamp_column is None or GLUCOSE_COLUMN not in frame.columns:
        return {"schema_valid": False}
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    glucose = pd.to_numeric(frame[GLUCOSE_COLUMN], errors="coerce")
    finite_glucose = glucose.map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
    valid_pair = timestamps.notna() & finite_glucose
    valid_glucose = glucose[valid_pair].astype(float)
    valid_timestamps = timestamps[valid_pair]
    timestamp_missing = int(timestamps.isna().sum())
    glucose_missing = int((~finite_glucose).sum())
    valid_frame = pd.DataFrame({
        "timestamp": valid_timestamps.reset_index(drop=True),
        "glucose": valid_glucose.reset_index(drop=True),
    })
    duplicate_groups = duplicate_rows = conflicting_duplicate_groups = 0
    if not valid_frame.empty:
        duplicate_summary = valid_frame.groupby("timestamp", sort=False)["glucose"].agg(["size", "nunique"])
        duplicate_summary = duplicate_summary[duplicate_summary["size"] > 1]
        duplicate_groups = int(len(duplicate_summary))
        duplicate_rows = int((duplicate_summary["size"] - 1).sum())
        conflicting_duplicate_groups = int((duplicate_summary["nunique"] > 1).sum())
    non_monotonic_steps = int(timestamps.diff().lt(pd.Timedelta(0)).sum())
    sorted_timestamps = valid_timestamps.sort_values()
    positive_deltas = sorted_timestamps.diff().dropna()
    positive_deltas = positive_deltas[positive_deltas > pd.Timedelta(0)]
    intervals_minutes = [float(delta.total_seconds() / 60.0) for delta in positive_deltas]
    median_interval = _median(intervals_minutes)
    gap_count = 0
    if median_interval is not None and median_interval > 0:
        gap_count = sum(interval > (3.0 * median_interval) for interval in intervals_minutes)
    span_days = None
    if len(sorted_timestamps) >= 2:
        span_days = float((sorted_timestamps.iloc[-1] - sorted_timestamps.iloc[0]).total_seconds() / 86400.0)
    if valid_glucose.empty:
        mean_glucose = median_glucose = sd_glucose = None
        below_70 = in_70_180 = above_180 = out_of_range = 0
    else:
        mean_glucose = float(valid_glucose.mean())
        median_glucose = float(valid_glucose.median())
        sd_glucose = float(valid_glucose.std(ddof=1)) if len(valid_glucose) > 1 else None
        below_70 = int((valid_glucose < 70).sum())
        in_70_180 = int(valid_glucose.between(70, 180, inclusive="both").sum())
        above_180 = int((valid_glucose > 180).sum())
        out_of_range = int(((valid_glucose < 40) | (valid_glucose > 400)).sum())
    denominator = len(valid_glucose)
    return {
        "schema_valid": True, "rows": int(len(frame)), "valid_pairs": int(valid_pair.sum()),
        "timestamp_missing": timestamp_missing, "glucose_missing_or_non_numeric": glucose_missing,
        "duplicate_timestamp_rows": duplicate_rows,
        "duplicate_timestamp_groups": duplicate_groups,
        "conflicting_duplicate_timestamp_groups": conflicting_duplicate_groups,
        "non_monotonic_steps": non_monotonic_steps,
        "gap_count": int(gap_count), "median_interval_minutes": median_interval, "span_days": span_days,
        "mean_glucose": mean_glucose, "median_glucose": median_glucose, "sd_glucose": sd_glucose,
        "below_70_fraction": (below_70 / denominator) if denominator else None,
        "in_70_180_fraction": (in_70_180 / denominator) if denominator else None,
        "above_180_fraction": (above_180 / denominator) if denominator else None,
        "out_of_range_readings": out_of_range,
    }

def _dataset_fingerprint(files: list[Path]) -> tuple[int, str]:
    """Hash sorted harmonized CSV contents without exposing filenames."""
    dataset_hash = hashlib.sha256()
    total_bytes = 0
    for path in files:
        file_hash = hashlib.sha256()
        file_bytes = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_hash.update(chunk)
                file_bytes += len(chunk)
        dataset_hash.update(file_bytes.to_bytes(8, "big"))
        dataset_hash.update(file_hash.digest())
        total_bytes += file_bytes
    return total_bytes, dataset_hash.hexdigest()


def audit_dataset(dataset_name: str, input_dir: str | Path, upstream_revision: str | None = None) -> dict[str, Any]:
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    extracted = root / "Extracted-glucose-files"
    data_dir = extracted if extracted.is_dir() else root
    files = sorted(path for path in data_dir.glob("*.csv") if path.is_file())
    if not files:
        raise ValueError(f"No participant CSV files found in dataset directory: {data_dir}")
    summaries: list[dict[str, Any]] = []
    unreadable_files = schema_invalid_files = 0
    for path in files:
        try:
            result = _participant_file_summary(path)
        except Exception:
            unreadable_files += 1
            continue
        if not result["schema_valid"]:
            schema_invalid_files += 1
            continue
        summaries.append(result)
    if not summaries:
        raise ValueError(f"No participant CSV files with the expected schema were found for dataset: {dataset_name}")
    total_bytes, content_sha256 = _dataset_fingerprint(files)
    def total(key: str) -> int:
        return int(sum(int(item.get(key, 0)) for item in summaries))
    def vals(key: str) -> list[float]:
        return [float(item[key]) for item in summaries if item.get(key) is not None]
    return {
        "schema_version": 1, "dataset": dataset_name,
        "upstream_glucose_ml_revision": upstream_revision,
        "harmonized_csv_total_bytes": total_bytes,
        "harmonized_csv_content_sha256": content_sha256,
        "input_layout": "Extracted-glucose-files" if extracted.is_dir() else "participant CSV files directly in dataset directory",
        "participant_files_found": len(files), "participants_with_valid_schema": len(summaries),
        "participants_with_valid_records": sum(1 for item in summaries if item["valid_pairs"] > 0),
        "unreadable_files_count": unreadable_files, "schema_invalid_files_count": schema_invalid_files,
        "record_count": total("rows"), "valid_timestamp_glucose_pairs": total("valid_pairs"),
        "missing_or_unparseable_timestamps": total("timestamp_missing"),
        "missing_or_non_numeric_glucose_values": total("glucose_missing_or_non_numeric"),
        "duplicate_timestamp_rows": total("duplicate_timestamp_rows"),
        "duplicate_timestamp_groups": total("duplicate_timestamp_groups"),
        "conflicting_duplicate_timestamp_groups": total("conflicting_duplicate_timestamp_groups"),
        "participants_with_duplicate_timestamps": sum(1 for item in summaries if item["duplicate_timestamp_rows"] > 0),
        "participants_with_conflicting_duplicate_timestamps": sum(1 for item in summaries if item["conflicting_duplicate_timestamp_groups"] > 0),
        "non_monotonic_timestamp_steps_in_source_order": total("non_monotonic_steps"),
        "participants_with_non_monotonic_timestamps": sum(1 for item in summaries if item["non_monotonic_steps"] > 0),
        "large_gap_count": total("gap_count"),
        "participants_with_large_gaps": sum(1 for item in summaries if item["gap_count"] > 0),
        "participants_with_out_of_range_readings": sum(1 for item in summaries if item["out_of_range_readings"] > 0),
        "large_gap_rule": "Within each participant, a gap is greater than 3 times that participant file's median positive sampling interval.",
        "median_participant_sampling_interval_minutes": _median(vals("median_interval_minutes")),
        "median_participant_observation_span_days": _median(vals("span_days")),
        "median_of_participant_mean_glucose_mg_dl": _median(vals("mean_glucose")),
        "median_of_participant_median_glucose_mg_dl": _median(vals("median_glucose")),
        "median_of_participant_glucose_sd_mg_dl": _median(vals("sd_glucose")),
        "median_reading_fraction_below_70_mg_dl": _median(vals("below_70_fraction")),
        "median_reading_fraction_70_to_180_mg_dl": _median(vals("in_70_180_fraction")),
        "median_reading_fraction_above_180_mg_dl": _median(vals("above_180_fraction")),
        "readings_outside_40_to_400_mg_dl_flagged_not_removed": total("out_of_range_readings"),
        "glucose_units": "mg/dL as declared by the Glucose-ML harmonized schema",
        "range_check_policy": "Values below 40 or above 400 mg/dL are flagged only; this audit does not delete or impute observations.",
        "fraction_policy": "Fractions are proportions of valid readings, not duration-weighted time-in-range estimates.",
        "timestamp_policy": "Timestamps are parsed as UTC for elapsed-time calculations; timezone-naive source timestamps are interpreted as UTC. No local-calendar-day metrics are computed.",
        "privacy_policy": "Aggregate dataset-level output only; no participant identifiers, filenames, or participant-level rows are emitted.",
    }

def parse_dataset_argument(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=PATH for each --dataset argument.")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Dataset name and path must both be non-empty.")
    return name.strip(), path.strip()

def main() -> None:
    parser = argparse.ArgumentParser(description="Create aggregate-only quality summaries for harmonized Glucose-ML CGM CSV files.")
    parser.add_argument("--dataset", action="append", type=parse_dataset_argument, required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True, help="Path for the aggregate JSON report.")
    parser.add_argument("--upstream-revision", default=None, help="Git commit SHA of the upstream Glucose-ML tools.")
    args = parser.parse_args()
    reports = [audit_dataset(name, path, args.upstream_revision) for name, path in args.dataset]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"audit_schema_version": 1, "datasets": reports}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote aggregate-only audit for {len(reports)} dataset(s): {args.output}")

if __name__ == "__main__":
    main()
