"""Neural dataset preparation with separate fit, validation, calibration, and test sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class NeuralArrays:
    """Scaled arrays for model fitting, early stopping, calibration, and testing."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def prepare_neural_datasets(
    X_train_df,
    y_train_series,
    X_test_df,
    y_test_series,
    val_fraction: float = 0.20,
    cal_fraction: float = 0.20,
    random_state: int = 42,
) -> NeuralArrays:
    """Create non-overlapping neural train, validation, calibration, and test sets.

    The outer test set is passed through untouched. A calibration subset is
    first removed from the outer training partition. The remaining development
    subset is divided into model-fitting and early-stopping validation subsets.
    The scaler is fitted only on model-fitting features, then applied unchanged
    to validation, calibration, and outer test features.

    Args:
        X_train_df: Features from the outer training partition.
        y_train_series: Targets from the outer training partition.
        X_test_df: Features from the untouched outer test partition.
        y_test_series: Targets from the untouched outer test partition.
        val_fraction: Fraction of the non-calibration development subset used
            for early-stopping validation.
        cal_fraction: Fraction of the outer training partition reserved only
            for final conformal calibration.
        random_state: Seed controlling both inner partitions.

    Returns:
        NeuralArrays with four disjoint partitions and training-only scaler stats.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1.")
    if not 0.0 < cal_fraction < 1.0:
        raise ValueError("cal_fraction must be strictly between 0 and 1.")

    X_outer_train, X_cal_raw, y_outer_train, y_cal_raw = train_test_split(
        X_train_df,
        y_train_series,
        test_size=cal_fraction,
        random_state=random_state,
        shuffle=True,
    )
    X_tr_raw, X_val_raw, y_tr_raw, y_val_raw = train_test_split(
        X_outer_train,
        y_outer_train,
        test_size=val_fraction,
        random_state=random_state,
        shuffle=True,
    )

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)
    X_cal_scaled = scaler.transform(X_cal_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_df).astype(np.float32)

    y_tr_arr = np.asarray(y_tr_raw, dtype=np.float32).reshape(-1, 1)
    y_val_arr = np.asarray(y_val_raw, dtype=np.float32).reshape(-1, 1)
    y_cal_arr = np.asarray(y_cal_raw, dtype=np.float32).reshape(-1, 1)
    y_test_arr = np.asarray(y_test_series, dtype=np.float32).reshape(-1, 1)

    return NeuralArrays(
        X_train=X_tr_scaled,
        y_train=y_tr_arr,
        X_val=X_val_scaled,
        y_val=y_val_arr,
        X_cal=X_cal_scaled,
        y_cal=y_cal_arr,
        X_test=X_test_scaled,
        y_test=y_test_arr,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
