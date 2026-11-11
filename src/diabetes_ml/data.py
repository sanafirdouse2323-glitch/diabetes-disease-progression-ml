"""Data loading, validation, and splitting for diabetes progression benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplits:
    """Immutable container for train and held-out test splits."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def test_size(self) -> int:
        return len(self.X_test)


def load_diabetes_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the scikit-learn diabetes benchmark as pandas objects.

    Returns:
        X: Feature DataFrame (442 samples, 10 preprocessed baseline features).
        y: Continuous quantitative disease progression target measured one year post-baseline.
    """
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target
    return X, y


def create_experiment_splits(
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplits:
    """Create reproducible outer train/test partition.

    Args:
        test_size: Fraction of samples reserved for final held-out testing (default: 0.20).
        random_state: Seed for reproducibility.

    Returns:
        DatasetSplits container with train and test partitions.
    """
    X, y = load_diabetes_data()
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        feature_names=feature_names,
    )
