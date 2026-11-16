"""Tests for the single-model Shanghai CGM forecasting pipeline."""
import numpy as np
import pandas as pd
from diabetes_ml.cgm_forecast import build_session_windows, split_participants

def test_build_windows_uses_exact_history_and_30_minute_target():
    frame = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=22, freq="15min"),
        "CGM (mg / dl)": np.arange(100, 122, dtype=float),
    })
    X, y = build_session_windows(frame, history_samples=20, horizon_steps=2, interval_minutes=15)
    assert X.shape == (1, 20)
    assert y.shape == (1,)
    assert X[0, 0] == 100
    assert X[0, -1] == 119
    assert y[0] == 121

def test_irregular_gap_breaks_forecast_window_without_interpolation():
    timestamps = list(pd.date_range("2025-01-01", periods=11, freq="15min"))
    timestamps += list(pd.date_range("2025-01-01 03:00", periods=11, freq="15min"))
    frame = pd.DataFrame({"Date": timestamps, "CGM (mg / dl)": np.arange(22, dtype=float)})
    X, y = build_session_windows(frame, history_samples=20, horizon_steps=2, interval_minutes=15)
    assert X.shape == (0, 20)
    assert y.shape == (0,)

def test_participant_split_is_disjoint_and_reproducible():
    keys = [f"subject_{index:03d}" for index in range(100)]
    first = split_participants(keys, seed=42)
    second = split_participants(list(reversed(keys)), seed=42)
    assert first == second
    assert sum(value == "train" for value in first.values()) == 70
    assert sum(value == "validation" for value in first.values()) == 15
    assert sum(value == "test" for value in first.values()) == 15
    assert set(key for key, value in first.items() if value == "train").isdisjoint(
        key for key, value in first.items() if value == "test"
    )
