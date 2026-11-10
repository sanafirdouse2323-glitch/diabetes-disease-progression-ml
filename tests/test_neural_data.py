import numpy as np

from diabetes_ml.data import create_experiment_splits
from diabetes_ml.neural.neural_data import prepare_neural_datasets


def test_neural_partitions_are_aligned_and_finite():
    splits = create_experiment_splits(random_state=42)
    data = prepare_neural_datasets(
        splits.X_train,
        splits.y_train,
        splits.X_test,
        splits.y_test,
        val_fraction=0.20,
        random_state=42,
    )

    assert data.X_train.shape == (282, 10)
    assert data.X_val.shape == (71, 10)
    assert data.X_test.shape == (89, 10)
    assert data.y_train.shape == (282, 1)
    assert data.y_val.shape == (71, 1)
    assert data.y_test.shape == (89, 1)
    for array in (
        data.X_train, data.X_val, data.X_test,
        data.y_train, data.y_val, data.y_test,
    ):
        assert np.isfinite(array).all()


def test_scaler_is_fitted_on_neural_training_subset_only():
    splits = create_experiment_splits(random_state=42)
    data = prepare_neural_datasets(
        splits.X_train,
        splits.y_train,
        splits.X_test,
        splits.y_test,
        val_fraction=0.20,
        random_state=42,
    )

    # StandardScaler training data should have zero mean and unit variance
    # (within floating-point tolerance); validation/test data are only transformed.
    assert np.allclose(data.X_train.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(data.X_train.std(axis=0), 1.0, atol=1e-5)
    assert np.all(data.scaler_scale > 0)
