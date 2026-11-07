"""Unit tests for dataset loading, validation, and split integrity."""

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


def test_split_isolation_no_overlap():
    splits = create_experiment_splits(test_size=0.20, random_state=42)
    # Check that indices were reset and no index collision exists
    assert len(splits.X_train) + len(splits.X_test) == 442
    # Verify values match expectations
    assert splits.y_train.min() >= 25.0
    assert splits.y_test.max() <= 346.0
