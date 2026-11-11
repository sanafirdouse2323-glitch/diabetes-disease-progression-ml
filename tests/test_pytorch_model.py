import numpy as np
import pytest

torch = pytest.importorskip("torch")

from diabetes_ml.data import create_experiment_splits
from diabetes_ml.neural.neural_data import prepare_neural_datasets
from diabetes_ml.neural.pytorch_model import DiabetesMLP, PyTorchTrainer


def test_pytorch_mlp_forward_shape():
    model = DiabetesMLP(in_features=10)
    output = model(torch.zeros((4, 10), dtype=torch.float32))
    assert tuple(output.shape) == (4, 1)


def test_pytorch_training_produces_finite_predictions_and_mc_uncertainty():
    splits = create_experiment_splits(random_state=42)
    data = prepare_neural_datasets(
        splits.X_train, splits.y_train, splits.X_test, splits.y_test,
        random_state=42,
    )
    trainer = PyTorchTrainer(
        in_features=10, epochs=5, patience=3, random_state=11,
    )
    trainer.fit(data.X_train, data.y_train, data.X_val, data.y_val)

    predictions = trainer.predict(data.X_test)
    previous_mode = trainer.model.training
    mc_mean, mc_std = trainer.predict_mc_dropout(data.X_test, n_samples=20)

    assert trainer.model.training is previous_mode
    assert predictions.shape == (len(data.X_test),)
    assert mc_mean.shape == predictions.shape
    assert mc_std.shape == predictions.shape
    assert np.isfinite(predictions).all()
    assert np.isfinite(mc_mean).all()
    assert np.isfinite(mc_std).all()
    assert np.all(mc_std >= 0)
    assert 1 <= len(trainer.history) <= 5


@pytest.mark.parametrize("n_samples", [0, -1, 1.5, True])
def test_mc_dropout_rejects_invalid_sample_counts(n_samples):
    trainer = PyTorchTrainer(in_features=10, epochs=2, patience=1, random_state=11)
    with pytest.raises(ValueError, match="positive integer"):
        trainer.predict_mc_dropout(np.zeros((2, 10), dtype=np.float32), n_samples=n_samples)
