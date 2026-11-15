"""Unit tests for dataset loading, validation, and split integrity."""

import pandas as pd
from sklearn.model_selection import train_test_split

from diabetes_ml.data import create_experiment_splits, load_diabetes_data


def test_load_diabetes_data_dimensions():
    X, y = load_diabetes_data()
    assert X.shape == (442, 10)
    assert len(y) == 442
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_create_experiment_splits_shapes():
    splits = create_experiment_splits(test_size=0.20, random_state=42)
    assert splits.train_size == 353
    assert splits.test_size == 89
    assert len(splits.feature_names) == 10
    assert set(splits.feature_names) == set(splits.X_train.columns)


def test_split_partitions_are_disjoint_and_match_source_rows():
    X, y = load_diabetes_data()
    expected_train_indices, expected_test_indices = train_test_split(
        X.index.to_numpy(),
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    # Prove that the source-observation identities assigned to the partitions
    # are disjoint, rather than inferring isolation from reset output indices.
    assert set(expected_train_indices).isdisjoint(set(expected_test_indices))

    splits = create_experiment_splits(test_size=0.20, random_state=42)

    # Compare the returned feature and target rows with their original source
    # observations. This also checks that each target remains aligned with X.
    pd.testing.assert_frame_equal(
        splits.X_train,
        X.loc[expected_train_indices].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        splits.X_test,
        X.loc[expected_test_indices].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        splits.y_train,
        y.loc[expected_train_indices].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        splits.y_test,
        y.loc[expected_test_indices].reset_index(drop=True),
    )
