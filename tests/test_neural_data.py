import numpy as np
import pytest

from diabetes_ml.data import create_experiment_splits
from diabetes_ml.neural.neural_data import prepare_neural_datasets


def test_neural_partitions_are_aligned_finite_and_separate():
    splits = create_experiment_splits(random_state=42)
    data = prepare_neural_datasets(
        splits.X_train,
        splits.y_train,
        splits.X_test,
        splits.y_test,
        val_fraction=0.20,
        cal_fraction=0.20,
        random_state=42,
    )

    # Outer training = 353; calibration = 71; remaining development = 282;
    # early-stopping validation = 57; model-fitting subset = 225.
    assert data.X_train.shape == (225, 10)
    assert data.X_val.shape == (57, 10)
    assert data.X_cal.shape == (71, 10)
    assert data.X_test.shape == (89, 10)
    assert data.y_train.shape == (225, 1)
    assert data.y_val.shape == (57, 1)
    assert data.y_cal.shape == (71, 1)
    assert data.y_test.shape == (89, 1)

    for array in (
        data.X_train, data.X_val, data.X_cal, data.X_test,
        data.y_train, data.y_val, data.y_cal, data.y_test,
    ):
        assert np.isfinite(array).all()


def test_scaler_is_fitted_only_on_neural_model_fitting_subset():
    splits = create_experiment_splits(random_state=42)
    data = prepare_neural_datasets(
        splits.X_train,
        splits.y_train,
        splits.X_test,
        splits.y_test,
        val_fraction=0.20,
        cal_fraction=0.20,
        random_state=42,
    )

    # The model-fitting subset is standardized; the other partitions are only transformed.
    assert np.allclose(data.X_train.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(data.X_train.std(axis=0), 1.0, atol=1e-5)
    assert np.all(data.scaler_scale > 0)


@pytest.mark.parametrize("val_fraction,cal_fraction", [(0.0, 0.2), (1.0, 0.2), (0.2, 0.0), (0.2, 1.0)])
def test_invalid_neural_partition_fractions_are_rejected(val_fraction, cal_fraction):
    splits = create_experiment_splits(random_state=42)
    with pytest.raises(ValueError):
        prepare_neural_datasets(
            splits.X_train,
            splits.y_train,
            splits.X_test,
            splits.y_test,
            val_fraction=val_fraction,
            cal_fraction=cal_fraction,
            random_state=42,
        )
