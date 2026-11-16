"""Tests for the session-preserving Shanghai source workbook audit."""
from pathlib import Path
import pandas as pd
from diabetes_ml.cgm_source_audit import audit_shanghai_source

def test_source_audit_preserves_recording_sessions_and_reconciles_rows(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    names = ["P001_1_20250101.xlsx", "P001_2_20250201.xlsx"]
    for name in names:
        (raw / name).write_bytes(b"test workbook placeholder")
    frames = {
        names[0]: pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=3, freq="15min"),
            "CGM (mg / dl)": [100, 110, 120],
        }),
        names[1]: pd.DataFrame({
            "Date": pd.date_range("2025-02-01", periods=2, freq="15min"),
            "CGM (mg / dl)": [130, 140],
        }),
    }
    monkeypatch.setattr(pd, "read_excel", lambda path: frames[Path(path).name].copy())
    harmonized = tmp_path / "harmonized"
    harmonized.mkdir()
    pd.DataFrame({
        "timestamp": list(pd.date_range("2025-01-01", periods=3, freq="15min")) + list(pd.date_range("2025-02-01", periods=2, freq="15min")),
        "glucose_value_mg_dl": [100, 110, 120, 130, 140],
    }).to_csv(harmonized / "P001.csv", index=False)

    report = audit_shanghai_source(raw, harmonized, upstream_revision="abc123")

    assert report["source_workbooks_found"] == 2
    assert report["source_participant_prefixes"] == 1
    assert report["participants_with_multiple_recording_files"] == 1
    assert report["valid_timestamp_glucose_pairs"] == 5
    assert report["harmonized_participant_csv_files"] == 1
    assert report["harmonized_participant_csv_rows"] == 5
    assert report["difference_source_valid_pairs_to_harmonized_rows"] == 0
    assert report["median_recording_sampling_interval_minutes"] == 15.0
    assert report["upstream_glucose_ml_revision"] == "abc123"
