"""Source-workbook audit that preserves ShanghaiT2DM recording sessions.

Only aggregate dataset-level fields are written. Source filenames and participant
identifiers are used internally for grouping and are never emitted.
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

SOURCE_PAPER_MEASUREMENTS = 112475
SOURCE_PAPER_URL = "https://doi.org/10.1038/s41597-023-01940-7"

def _median(values: list[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(values)) if values else None

def _percentile(values: list[float], q: float) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(pd.Series(values).quantile(q)) if values else None

def _file_hash(paths: list[Path]) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    for path in sorted(paths):
        item = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                item.update(chunk)
                size += len(chunk)
        digest.update(size.to_bytes(8, "big"))
        digest.update(item.digest())
        total += size
    return total, digest.hexdigest()

def _audit_workbook(path: Path) -> dict[str, Any]:
    frame = pd.read_excel(path)
    timestamp_col = next((col for col in ("Date", "timestamp") if col in frame.columns), None)
    glucose_col = next((col for col in ("CGM (mg / dl)", "CGM ", "glucose_value_mg_dl") if col in frame.columns), None)
    if timestamp_col is None or glucose_col is None:
        return {"schema_valid": False}
    raw_timestamps = frame[timestamp_col]
    raw_glucose = frame[glucose_col]
    timestamps = pd.to_datetime(raw_timestamps, errors="coerce", utc=True)
    glucose = pd.to_numeric(raw_glucose, errors="coerce")
    finite_glucose = glucose.map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
    valid = timestamps.notna() & finite_glucose
    valid_timestamps = timestamps[valid]
    valid_glucose = glucose[valid].astype(float)
    paired = pd.DataFrame({
        "timestamp": valid_timestamps.reset_index(drop=True),
        "glucose": valid_glucose.reset_index(drop=True),
    })
    duplicate_rows = duplicate_groups = conflicting_groups = 0
    if not paired.empty:
        groups = paired.groupby("timestamp", sort=False)["glucose"].agg(["size", "nunique"])
        groups = groups[groups["size"] > 1]
        duplicate_groups = int(len(groups))
        duplicate_rows = int((groups["size"] - 1).sum())
        conflicting_groups = int((groups["nunique"] > 1).sum())
    non_monotonic = int(timestamps.diff().lt(pd.Timedelta(0)).sum())
    sorted_times = valid_timestamps.sort_values()
    delta = sorted_times.diff().dropna()
    delta = delta[delta > pd.Timedelta(0)]
    intervals = [float(item.total_seconds() / 60) for item in delta]
    cadence = _median(intervals)
    gaps = sum(value > 3 * cadence for value in intervals) if cadence and cadence > 0 else 0
    span_days = float((sorted_times.iloc[-1] - sorted_times.iloc[0]).total_seconds() / 86400) if len(sorted_times) > 1 else None
    out_of_range = int(((valid_glucose < 40) | (valid_glucose > 400)).sum())
    return {
        "schema_valid": True,
        "source_rows": int(len(frame)),
        "missing_timestamps": int(raw_timestamps.isna().sum()),
        "unparseable_timestamps": int((raw_timestamps.notna() & timestamps.isna()).sum()),
        "missing_glucose_values": int(raw_glucose.isna().sum()),
        "non_numeric_glucose_values": int((raw_glucose.notna() & glucose.isna()).sum()),
        "non_finite_glucose_values": int((glucose.notna() & ~finite_glucose).sum()),
        "finite_glucose_values": int(finite_glucose.sum()),
        "valid_timestamp_glucose_pairs": int(valid.sum()),
        "duplicate_timestamp_rows": duplicate_rows,
        "duplicate_timestamp_groups": duplicate_groups,
        "conflicting_duplicate_timestamp_groups": conflicting_groups,
        "non_monotonic_steps": non_monotonic,
        "large_gaps": int(gaps),
        "median_sampling_interval_minutes": cadence,
        "session_span_days": span_days,
        "out_of_range_readings": out_of_range,
    }

def audit_shanghai_source(raw_dir: str | Path, harmonized_dir: str | Path | None = None, upstream_revision: str | None = None) -> dict[str, Any]:
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Shanghai raw workbook directory does not exist: {root}")
    workbooks = sorted(path for path in root.glob("*.xls*") if path.is_file() and not path.name.startswith("~$"))
    if not workbooks:
        raise ValueError(f"No Shanghai Excel workbooks found in: {root}")
    subject_sessions: dict[str, int] = {}
    summaries: list[dict[str, Any]] = []
    unreadable = schema_invalid = 0
    for path in workbooks:
        subject_key = path.name.split("_", 1)[0]
        subject_sessions[subject_key] = subject_sessions.get(subject_key, 0) + 1
        try:
            summary = _audit_workbook(path)
        except Exception:
            unreadable += 1
            continue
        if not summary["schema_valid"]:
            schema_invalid += 1
            continue
        summaries.append(summary)
    if not summaries:
        raise ValueError("No Shanghai source workbooks with the expected Date and CGM columns were readable.")
    def total(key: str) -> int:
        return int(sum(int(item.get(key, 0)) for item in summaries))
    def vals(key: str) -> list[float]:
        return [float(item[key]) for item in summaries if item.get(key) is not None]
    raw_bytes, raw_sha = _file_hash(workbooks)
    harmonized_files: list[Path] = []
    harmonized_rows = None
    if harmonized_dir is not None:
        hroot = Path(harmonized_dir)
        if not hroot.is_dir():
            raise FileNotFoundError(f"Shanghai harmonized directory does not exist: {hroot}")
        harmonized_files = sorted(hroot.glob("*.csv"))
        harmonized_rows = 0
        for path in harmonized_files:
            harmonized_rows += int(len(pd.read_csv(path)))
    sessions_per_subject = list(subject_sessions.values())
    result = {
        "schema_version": 1,
        "dataset": "shanghai_t2dm",
        "upstream_glucose_ml_revision": upstream_revision,
        "source_workbooks_found": len(workbooks),
        "source_workbooks_readable": len(summaries),
        "unreadable_workbooks": unreadable,
        "schema_invalid_workbooks": schema_invalid,
        "source_participant_prefixes": len(subject_sessions),
        "participants_with_one_recording_file": sum(count == 1 for count in sessions_per_subject),
        "participants_with_two_recording_files": sum(count == 2 for count in sessions_per_subject),
        "participants_with_three_or_more_recording_files": sum(count >= 3 for count in sessions_per_subject),
        "participants_with_multiple_recording_files": sum(count > 1 for count in sessions_per_subject),
        "source_rows": total("source_rows"),
        "missing_timestamps": total("missing_timestamps"),
        "unparseable_timestamps": total("unparseable_timestamps"),
        "missing_glucose_values": total("missing_glucose_values"),
        "non_numeric_glucose_values": total("non_numeric_glucose_values"),
        "non_finite_glucose_values": total("non_finite_glucose_values"),
        "finite_glucose_values": total("finite_glucose_values"),
        "valid_timestamp_glucose_pairs": total("valid_timestamp_glucose_pairs"),
        "duplicate_timestamp_rows_within_recording_files": total("duplicate_timestamp_rows"),
        "duplicate_timestamp_groups_within_recording_files": total("duplicate_timestamp_groups"),
        "conflicting_duplicate_timestamp_groups_within_recording_files": total("conflicting_duplicate_timestamp_groups"),
        "non_monotonic_timestamp_steps_within_recording_files": total("non_monotonic_steps"),
        "large_gaps_within_recording_files": total("large_gaps"),
        "recording_files_with_large_gaps": sum(1 for item in summaries if item["large_gaps"] > 0),
        "median_recording_span_days": _median(vals("session_span_days")),
        "median_recording_sampling_interval_minutes": _median(vals("median_sampling_interval_minutes")),
        "readings_outside_40_to_400_mg_dl_flagged_not_removed": total("out_of_range_readings"),
        "source_paper_reported_cgm_measurements": SOURCE_PAPER_MEASUREMENTS,
        "difference_valid_pairs_from_source_to_paper_count": total("valid_timestamp_glucose_pairs") - SOURCE_PAPER_MEASUREMENTS,
        "raw_workbook_total_bytes": raw_bytes,
        "raw_workbook_content_sha256": raw_sha,
        "harmonized_participant_csv_files": len(harmonized_files) if harmonized_dir is not None else None,
        "harmonized_participant_csv_rows": harmonized_rows,
        "difference_source_valid_pairs_to_harmonized_rows": (total("valid_timestamp_glucose_pairs") - harmonized_rows) if harmonized_rows is not None else None,
        "source_paper_reference": SOURCE_PAPER_URL,
        "session_boundary_policy": "Each source Excel workbook is treated as a separate recording session. Multiple workbooks sharing the same filename prefix are grouped only to count participant-level recording periods; their time series are not concatenated for session-duration or gap statistics.",
        "privacy_policy": "Aggregate-only report; source filenames and participant identifiers are not emitted.",
    }
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ShanghaiT2DM source workbooks without merging recording sessions.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--harmonized-dir", type=Path, default=None)
    parser.add_argument("--upstream-revision", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_shanghai_source(args.raw_dir, args.harmonized_dir, args.upstream_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote aggregate Shanghai source-workbook audit: {args.output}")

if __name__ == "__main__":
    main()
