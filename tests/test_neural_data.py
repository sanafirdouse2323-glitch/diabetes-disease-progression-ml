import numpy as np

from diabetes_ml.data import load_diabetes_frame, make_train_test_split
from diabetes_ml.neural.neural_data import prepare_neural_data


def test_neural_partitions_are_aligned_and_disjoint():
    X, y = load_diabetes_frame()
    split = make_train_test_split(X, y, random_state=42)
    data = prepare_neural_data(split, random_state=42)

    assert data.X_train.shape[1] == 10
    assert data.X_validation.shape[1] == 10
    assert data.X_test.shape[1] == 10
    assert data.y_train.shape == (len(data.X_train), 1)
    assert data.y_validation.shape == (len(data.X_validation), 1)
    assert data.y_test.shape == (len(data.X_test), 1)
    assert set(data.train_indices).isdisjoint(data.validation_indices)
    assert set(data.train_indices).isdisjoint(data.test_indices)
    assert set(data.validation_indices).isdisjoint(data.test_indices)
    assert np.array_equal(data.test_indices, split.X_test.index.to_numpy())
    assert np.isfinite(data.X_train).all()
    assert np.isfinite(data.X_validation).all()
    assert np.isfinite(data.X_test).all()


def test_scaler_is_fitted_on_neural_training_subset_only():
    X, y = load_diabetes_frame()
    split = make_train_test_split(X, y, random_state=42)
    data = prepare_neural_data(split, random_state=42)

    assert np.allclose(data.X_train.mean(axis=0), 0.0, atol=1e-6)
    assert np.all(data.scaler.scale_ > 0)
