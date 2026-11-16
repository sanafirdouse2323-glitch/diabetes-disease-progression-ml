"""Source-level BrisT1D-Open duplicate/device audit.

The upstream Glucose-ML extractor drops the source 'device' column. This audit
checks whether duplicate timestamps in the processed-state source coincide with
multiple device labels, while emitting aggregate counts only.
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

def _median(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(clean)) if clean else None

def _percentile(values: list[float], q: float) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(pd.Series(clean).quantile(q)) if clean else None

def _fingerprint(paths: list[Path]) -> tuple[int, str]:
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

def audit_bris_source(raw_dir: str | Path, harmonized_dir: str | Path | None = None, upstream_revision: str | None = None) -> dict[str, Any]:
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"BrisT1D-Open raw dataset directory does not exist: {root}")
    files = sorted(path for path in root.rglob("**/processed_state/*.csv") if path.is_file())
    if not files:
        raise ValueError(f"No processed_state CSV files found under: {root}")
    summaries: list[dict[str, Any]] = []
    unreadable = schema_invalid = 0
    device_row_counts: dict[str, int] = {}
    for path in files:
        try:
            frame = pd.read_csv(path)
        except Exception:
            unreadable += 1
            continue
        if not {"timestamp", "bg"}.issubset(frame.columns):
            schema_invalid += 1
            continue
        raw_ts = frame["timestamp"]
        raw_bg = frame["bg"]
        timestamps = pd.to_datetime(raw_ts, errors="coerce", utc=True)
        bg = pd.to_numeric(raw_bg, errors="coerce")
        finite = bg.map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
        valid = timestamps.notna() & finite
        valid_bg = bg[valid].astype(float)
        device = frame["device"].astype("string") if "device" in frame.columns else pd.Series(pd.NA, index=frame.index, dtype="string")
        valid_frame = pd.DataFrame({
            "timestamp": timestamps[valid].reset_index(drop=True),
            "bg_mmol_l": valid_bg.reset_index(drop=True),
            "device": device[valid].reset_index(drop=True),
        })
        for label, count in device[valid].fillna("<missing>").value_counts().items():
            device_row_counts[str(label)] = device_row_counts.get(str(label), 0) + int(count)
        duplicate_rows = duplicate_groups = conflicting_groups = identical_groups = 0
        cross_device_conflicts = same_device_conflicts = unknown_device_conflicts = 0
        duplicate_spreads_mg_dl: list[float] = []
        if not valid_frame.empty:
            groups = valid_frame.groupby("timestamp", sort=False)
            for _, group in groups:
                if len(group) < 2:
                    continue
                duplicate_groups += 1
                duplicate_rows += len(group) - 1
                glucose_unique = group["bg_mmol_l"].nunique(dropna=True)
                device_unique = group["device"].nunique(dropna=True)
                has_missing_device = bool(group["device"].isna().any())
                if glucose_unique <= 1:
                    identical_groups += 1
                else:
                    conflicting_groups += 1
                    spread = float((group["bg_mmol_l"].max() - group["bg_mmol_l"].min()) * 18.0)
                    duplicate_spreads_mg_dl.append(spread)
                    if device_unique > 1:
                        cross_device_conflicts += 1
                    elif has_missing_device:
                        unknown_device_conflicts += 1
                    else:
                        same_device_conflicts += 1
        non_monotonic = int(timestamps.diff().lt(pd.Timedelta(0)).sum())
        sorted_ts = timestamps[valid].sort_values()
        deltas = sorted_ts.diff().dropna()
        positive = deltas[deltas > pd.Timedelta(0)]
        interval_minutes = [float(value.total_seconds() / 60) for value in positive]
        cadence = _median(interval_minutes)
        gaps = sum(value > 3 * cadence for value in interval_minutes) if cadence else 0
        summaries.append({
            "rows": int(len(frame)),
            "missing_timestamps": int(raw_ts.isna().sum()),
            "unparseable_timestamps": int((raw_ts.notna() & timestamps.isna()).sum()),
            "missing_bg": int(raw_bg.isna().sum()),
            "non_numeric_bg": int((raw_bg.notna() & bg.isna()).sum()),
            "valid_pairs": int(valid.sum()),
            "duplicate_rows": int(duplicate_rows),
            "duplicate_groups": int(duplicate_groups),
            "conflicting_groups": int(conflicting_groups),
            "identical_groups": int(identical_groups),
            "cross_device_conflicts": int(cross_device_conflicts),
            "same_device_conflicts": int(same_device_conflicts),
            "unknown_device_conflicts": int(unknown_device_conflicts),
            "duplicate_spreads_mg_dl": duplicate_spreads_mg_dl,
            "non_monotonic_steps": non_monotonic,
            "large_gaps": int(gaps),
            "participants_with_device_column": int("device" in frame.columns),
            "unique_device_labels": int(device[valid].nunique(dropna=True)),
            "out_of_range": int((((valid_bg * 18.0) < 40) | ((valid_bg * 18.0) > 400)).sum()),
        })
    if not summaries:
        raise ValueError("No readable BrisT1D processed-state files with timestamp and bg columns were found.")
    def total(key: str) -> int:
        return int(sum(item.get(key, 0) for item in summaries))
    spreads = [value for item in summaries for value in item["duplicate_spreads_mg_dl"]]
    raw_bytes, raw_sha = _fingerprint(files)
    harmonized_files: list[Path] = []
    harmonized_rows = None
    harmonized_duplicate_rows = harmonized_duplicate_groups = harmonized_conflicts = None
    if harmonized_dir is not None:
        hroot = Path(harmonized_dir)
        if not hroot.is_dir():
            raise FileNotFoundError(f"BrisT1D harmonized directory does not exist: {hroot}")
        harmonized_files = sorted(hroot.glob("*.csv"))
        harmonized_rows = harmonized_duplicate_rows = harmonized_duplicate_groups = harmonized_conflicts = 0
        for path in harmonized_files:
            frame = pd.read_csv(path)
            harmonized_rows += len(frame)
            if {"timestamp", "glucose_value_mg_dl"}.issubset(frame.columns):
                frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
                frame["glucose_value_mg_dl"] = pd.to_numeric(frame["glucose_value_mg_dl"], errors="coerce")
                grouped = frame.dropna(subset=["timestamp", "glucose_value_mg_dl"]).groupby("timestamp")["glucose_value_mg_dl"].agg(["size", "nunique"])
                dup = grouped[grouped["size"] > 1]
                harmonized_duplicate_groups += int(len(dup))
                harmonized_duplicate_rows += int((dup["size"] - 1).sum())
                harmonized_conflicts += int((dup["nunique"] > 1).sum())
    labels = sorted(device_row_counts)
    return {
        "schema_version": 1,
        "dataset": "bris_t1d_open",
        "upstream_glucose_ml_revision": upstream_revision,
        "source_processed_state_files_found": len(files),
        "source_files_readable": len(summaries),
        "unreadable_files": unreadable,
        "schema_invalid_files": schema_invalid,
        "participant_files_with_device_column": total("participants_with_device_column"),
        "source_rows": total("rows"),
        "missing_timestamps": total("missing_timestamps"),
        "unparseable_timestamps": total("unparseable_timestamps"),
        "missing_bg_values": total("missing_bg"),
        "non_numeric_bg_values": total("non_numeric_bg"),
        "valid_timestamp_bg_pairs": total("valid_pairs"),
        "source_duplicate_timestamp_rows": total("duplicate_rows"),
        "source_duplicate_timestamp_groups": total("duplicate_groups"),
        "source_conflicting_duplicate_timestamp_groups": total("conflicting_groups"),
        "source_identical_duplicate_timestamp_groups": total("identical_groups"),
        "conflicting_duplicate_groups_across_multiple_device_labels": total("cross_device_conflicts"),
        "conflicting_duplicate_groups_with_one_device_label": total("same_device_conflicts"),
        "conflicting_duplicate_groups_with_missing_or_unknown_device": total("unknown_device_conflicts"),
        "conflicting_duplicate_glucose_spread_mg_dl_median": _median(spreads),
        "conflicting_duplicate_glucose_spread_mg_dl_p95": _percentile(spreads, 0.95),
        "conflicting_duplicate_glucose_spread_mg_dl_max": max(spreads) if spreads else None,
        "participants_with_any_duplicate_timestamp_groups": sum(1 for item in summaries if item["duplicate_groups"] > 0),
        "participants_with_multiple_device_labels": sum(1 for item in summaries if item["unique_device_labels"] > 1),
        "unique_device_labels": labels,
        "device_row_counts": device_row_counts,
        "non_monotonic_timestamp_steps": total("non_monotonic_steps"),
        "large_gap_count": total("large_gaps"),
        "readings_outside_40_to_400_mg_dl_flagged_not_removed": total("out_of_range"),
        "raw_processed_state_total_bytes": raw_bytes,
        "raw_processed_state_content_sha256": raw_sha,
        "harmonized_csv_files": len(harmonized_files) if harmonized_dir is not None else None,
        "harmonized_rows": harmonized_rows,
        "harmonized_duplicate_timestamp_rows": harmonized_duplicate_rows,
        "harmonized_duplicate_timestamp_groups": harmonized_duplicate_groups,
        "harmonized_conflicting_duplicate_timestamp_groups": harmonized_conflicts,
        "difference_source_valid_pairs_to_harmonized_rows": (total("valid_pairs") - harmonized_rows) if harmonized_rows is not None else None,
        "device_column_policy": "The upstream extractor retains only timestamp and bg and drops device. This audit preserves device labels for duplicate-timestamp diagnostics; no rows are deduplicated or selected.",
        "privacy_policy": "Aggregate-only output; filenames, participant identifiers, timestamps and raw glucose readings are not emitted.",
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BrisT1D-Open source device labels and duplicate timestamps.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--harmonized-dir", type=Path, default=None)
    parser.add_argument("--upstream-revision", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_bris_source(args.raw_dir, args.harmonized_dir, args.upstream_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote aggregate BrisT1D source/device audit: {args.output}")

if __name__ == "__main__":
    main()
