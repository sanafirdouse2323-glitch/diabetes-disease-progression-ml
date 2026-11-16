"""Tests for the BrisT1D source device/duplicate audit."""
from pathlib import Path
import pandas as pd
from diabetes_ml.cgm_bris_source_audit import audit_bris_source

def test_duplicate_groups_are_classified_by_device_without_exporting_ids(tmp_path):
    raw = tmp_path / "raw"
    processed = raw / "dataset" / "device_data" / "processed_state"
    processed.mkdir(parents=True)
    frame = pd.DataFrame({
        "timestamp": [
            "2025-01-01T00:00:00Z",
            "2025-01-01T00:00:00Z",
            "2025-01-01T00:05:00Z",
            "2025-01-01T00:05:00Z",
            "2025-01-01T00:10:00Z",
        ],
        "bg": [5.0, 6.0, 5.5, 5.5, 6.0],
        "device": ["DeviceA", "DeviceB", "DeviceA", "DeviceA", "DeviceB"],
    })
    source = processed / "P01.csv"
    frame.to_csv(source, index=False)
    harmonized = tmp_path / "harmonized"
    harmonized.mkdir()
    pd.DataFrame({
        "timestamp": pd.to_datetime(frame["timestamp"], utc=True),
        "glucose_value_mg_dl": frame["bg"] * 18,
    }).to_csv(harmonized / "P01.csv", index=False)

    report = audit_bris_source(raw, harmonized, upstream_revision="abc123")

    assert report["source_processed_state_files_found"] == 1
    assert report["source_duplicate_timestamp_groups"] == 2
    assert report["source_conflicting_duplicate_timestamp_groups"] == 1
    assert report["source_identical_duplicate_timestamp_groups"] == 1
    assert report["conflicting_duplicate_groups_across_multiple_device_labels"] == 1
    assert report["conflicting_duplicate_groups_with_one_device_label"] == 0
    assert report["harmonized_conflicting_duplicate_timestamp_groups"] == 1
    assert report["difference_source_valid_pairs_to_harmonized_rows"] == 0
    assert "P01" not in str(report)
