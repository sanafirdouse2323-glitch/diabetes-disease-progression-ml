"""Dataset loading and deterministic train/test splitting."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DiabetesSplit:
    """Named train/test partitions for the benchmark data."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_diabetes_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Load the public scikit-learn diabetes regression dataset.

    Returns feature columns and the continuous one-year progression target.
    """
    bunch = load_diabetes(as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.copy()
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("Expected pandas objects from load_diabetes(as_frame=True).")
    if X.empty or len(X) != len(y):
        raise ValueError("Dataset is empty or features and target are misaligned.")
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Unexpected missing values in the bundled dataset.")
    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DiabetesSplit:
    """Create a reproducible held-out split without fitting transformations."""
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of observations.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return DiabetesSplit(X_train, X_test, y_train, y_test)
