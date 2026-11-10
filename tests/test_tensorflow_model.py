import numpy as np
import pytest

pytest.importorskip("tensorflow")

from diabetes_ml.data import create_experiment_splits
from diabetes_ml.neural.neural_data import prepare_neural_datasets
from diabetes_ml.neural.tensorflow_model import TensorFlowTrainer


def test_tensorflow_training_produces_finite_predictions():
    splits = create_experiment_splits(random_state=42)
    data = prepare_neural_datasets(
        splits.X_train, splits.y_train, splits.X_test, splits.y_test,
        random_state=42,
    )
    trainer = TensorFlowTrainer(
        in_features=10, epochs=5, patience=3, random_state=11,
    )
    trainer.fit(data.X_train, data.y_train, data.X_val, data.y_val)
    predictions = trainer.predict(data.X_test)

    assert predictions.shape == (len(data.X_test),)
    assert np.isfinite(predictions).all()
    assert 1 <= len(trainer.history_records) <= 5
