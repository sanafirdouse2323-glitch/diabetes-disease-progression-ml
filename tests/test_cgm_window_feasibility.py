"""Tests for session-preserving Shanghai forecast-window feasibility."""
from pathlib import Path
import pandas as pd
from diabetes_ml.cgm_window_feasibility import audit_window_feasibility

def test_windows_do_not_cross_workbook_session_boundaries(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    names = ["P001_1.xlsx", "P001_2.xlsx", "P002_1.xlsx"]
    for name in names:
        (raw / name).write_bytes(b"placeholder")
    def series(start, count):
        return pd.DataFrame({
            "Date": pd.date_range(start, periods=count, freq="15min"),
            "CGM (mg / dl)": [120.0] * count,
        })
    frames = {
        names[0]: series("2025-01-01 00:00", 21),
        names[1]: series("2025-01-01 05:15", 21),
        names[2]: series("2025-01-02 00:00", 22),
    }
    monkeypatch.setattr(pd, "read_excel", lambda path: frames[Path(path).name].copy())

    report = audit_window_feasibility(raw, history_samples=20, horizon_minutes=30, interval_minutes=15, seed=42)

    assert report["source_workbooks"] == 3
    assert report["participant_filename_prefixes"] == 2
    assert report["participants_with_multiple_sessions"] == 1
    assert report["session_count"] == 3
    assert report["total_valid_windows"] == 1
    assert report["sessions_with_at_least_one_valid_window"] == 1
    assert sum(report["participant_split"][key] for key in ("train_windows", "validation_windows", "test_windows")) == 1
    assert report["history_samples"] == 20
    assert report["forecast_horizon_minutes"] == 30
    assert report["forecast_feasibility_only"] is True
