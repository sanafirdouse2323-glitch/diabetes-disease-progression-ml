import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from diabetes_ml.data import load_diabetes_frame, make_train_test_split
from diabetes_ml.neural.neural_data import prepare_neural_data
from diabetes_ml.neural.tensorflow_model import build_tensorflow_model, train_tensorflow_model


def test_tensorflow_mlp_forward_shape():
    model = build_tensorflow_model(input_dim=10)
    output = model(np.zeros((4, 10), dtype=np.float32), training=False)
    assert tuple(output.shape) == (4, 1)


def test_tensorflow_training_produces_finite_test_metrics():
    X, y = load_diabetes_frame()
    split = make_train_test_split(X, y, random_state=42)
    data = prepare_neural_data(split, random_state=42)
    result = train_tensorflow_model(data, epochs=5, patience=3, seed=11)

    assert result.predictions.shape == (len(data.X_test),)
    assert np.isfinite(result.predictions).all()
    assert set(result.metrics) == {"mae", "rmse", "r2"}
    assert all(np.isfinite(value) for value in result.metrics.values())
    assert 1 <= result.best_epoch <= len(result.history) <= 5
