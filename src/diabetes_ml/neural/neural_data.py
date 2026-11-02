"""Create one leakage-conscious data split shared by both neural frameworks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from diabetes_ml.data import DiabetesSplit


@dataclass(frozen=True)
class NeuralData:
    """Scaled arrays and row identifiers for neural-network training."""

    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    feature_names: tuple[str, ...]
    scaler: StandardScaler


def prepare_neural_data(
    split: DiabetesSplit,
    *,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> NeuralData:
    """Split outer training data and fit the scaler on the neural train subset only."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be strictly between 0 and 1.")
    if len(split.X_train) != len(split.y_train):
        raise ValueError("Training features and targets are misaligned.")
    if len(split.X_test) != len(split.y_test):
        raise ValueError("Test features and targets are misaligned.")
    if split.X_train.empty or split.X_test.empty:
        raise ValueError("Training and test partitions must not be empty.")

    X_fit, X_validation, y_fit, y_validation = train_test_split(
        split.X_train,
        split.y_train,
        test_size=validation_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit).astype(np.float32)
    X_validation_scaled = scaler.transform(X_validation).astype(np.float32)
    X_test_scaled = scaler.transform(split.X_test).astype(np.float32)

    # A column target matches the one-output tensor produced by each framework.
    y_fit_array = y_fit.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_validation_array = y_validation.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_test_array = split.y_test.to_numpy(dtype=np.float32).reshape(-1, 1)

    arrays = (
        X_fit_scaled,
        X_validation_scaled,
        X_test_scaled,
        y_fit_array,
        y_validation_array,
        y_test_array,
    )
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Prepared neural data contains non-finite values.")

    return NeuralData(
        X_train=X_fit_scaled,
        X_validation=X_validation_scaled,
        X_test=X_test_scaled,
        y_train=y_fit_array,
        y_validation=y_validation_array,
        y_test=y_test_array,
        train_indices=X_fit.index.to_numpy(),
        validation_indices=X_validation.index.to_numpy(),
        test_indices=split.X_test.index.to_numpy(),
        feature_names=tuple(str(name) for name in split.X_train.columns),
        scaler=scaler,
    )
