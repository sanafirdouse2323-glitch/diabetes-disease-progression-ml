"""Tests for the aggregate-only CGM data quality audit."""
import json
import pandas as pd
from diabetes_ml.cgm_audit import audit_dataset

def test_audit_reports_dataset_aggregates_without_participant_ids(tmp_path):
    dataset = tmp_path / "ShanghaiT2DM"
    dataset.mkdir()
    pd.DataFrame({"timestamp": ["2025-01-01T00:00:00", "2025-01-01T00:05:00", "2025-01-01T00:10:00", "2025-01-01T00:15:00"], "glucose_value_mg_dl": [100, 150, 190, 60]}).to_csv(dataset / "participant_PRIVATE_001.csv", index=False)
    pd.DataFrame({"timestamp": ["2025-01-02T00:00:00", "2025-01-02T00:05:00"], "glucose_value_mg_dl": [110, 120]}).to_csv(dataset / "participant_PRIVATE_002.csv", index=False)
    report = audit_dataset("shanghai_t2dm", dataset)
    assert report["participant_files_found"] == 2
    assert report["participants_with_valid_schema"] == 2
    assert report["record_count"] == 6
    assert report["valid_timestamp_glucose_pairs"] == 6
    assert report["median_participant_sampling_interval_minutes"] == 5.0
    assert report["median_reading_fraction_70_to_180_mg_dl"] == 0.75
    assert "participant_PRIVATE_001" not in json.dumps(report)
    assert report["fraction_policy"].startswith("Fractions are proportions")

def test_audit_flags_bad_schema_without_exposing_filename(tmp_path):
    dataset = tmp_path / "bad"
    dataset.mkdir()
    pd.DataFrame({"time": ["2025-01-01"], "value": [100]}).to_csv(dataset / "sensitive_subject_name.csv", index=False)
    report = audit_dataset("bad_schema", dataset)
    assert report["participant_files_found"] == 1
    assert report["schema_invalid_files_count"] == 1
    assert report["participants_with_valid_schema"] == 0
    assert "sensitive_subject_name" not in json.dumps(report)

def test_audit_flags_missing_and_out_of_range_values(tmp_path):
    dataset = tmp_path / "quality"
    dataset.mkdir()
    pd.DataFrame({"timestamp": ["2025-01-01T00:00:00", "bad-time", "2025-01-01T00:10:00"], "glucose_value_mg_dl": [100, 120, 450]}).to_csv(dataset / "p001.csv", index=False)
    report = audit_dataset("quality", dataset)
    assert report["missing_or_unparseable_timestamps"] == 1
    assert report["readings_outside_40_to_400_mg_dl_flagged_not_removed"] == 1
    assert report["valid_timestamp_glucose_pairs"] == 2
