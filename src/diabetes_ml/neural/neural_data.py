"""Neural dataset preparation, inner split, and isolated feature scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class NeuralArrays:
    """Scaled numpy arrays for neural network training and evaluation."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
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
    random_state: int = 42,
) -> NeuralArrays:
    """Split training partition into neural train/val subsets and fit scaler strictly on train.

    Ensures zero data leakage:
    1. Validation subset is carved out of the training partition only.
    2. StandardScaler is fitted ONLY on the neural training subset.
    3. Neural validation and outer test sets are transformed using the fitted training statistics.

    Args:
        X_train_df: Outer training feature DataFrame.
        y_train_series: Outer training targets.
        X_test_df: Outer test feature DataFrame.
        y_test_series: Outer test targets.
        val_fraction: Fraction of outer training partition reserved for validation early stopping.
        random_state: Seed for inner split.

    Returns:
        NeuralArrays container with scaled arrays and scaler statistics.
    """
    X_tr_raw, X_val_raw, y_tr_raw, y_val_raw = train_test_split(
        X_train_df,
        y_train_series,
        test_size=val_fraction,
        random_state=random_state,
        shuffle=True,
    )

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_df).astype(np.float32)

    y_tr_arr = np.asarray(y_tr_raw, dtype=np.float32).reshape(-1, 1)
    y_val_arr = np.asarray(y_val_raw, dtype=np.float32).reshape(-1, 1)
    y_test_arr = np.asarray(y_test_series, dtype=np.float32).reshape(-1, 1)

    return NeuralArrays(
        X_train=X_tr_scaled,
        y_train=y_tr_arr,
        X_val=X_val_scaled,
        y_val=y_val_arr,
        X_test=X_test_scaled,
        y_test=y_test_arr,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
