"""An equivalent TensorFlow/Keras multilayer perceptron."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from diabetes_ml.evaluation import regression_metrics
from diabetes_ml.neural.neural_data import NeuralData


@dataclass
class TensorFlowTrainingResult:
    model: tf.keras.Model
    history: list[dict[str, float]]
    predictions: np.ndarray
    metrics: dict[str, float]
    best_epoch: int


def build_tensorflow_model(
    input_dim: int = 10,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build and compile the same 16/8 ReLU MLP used in the PyTorch version."""
    if input_dim < 1:
        raise ValueError("input_dim must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ],
        name="diabetes_mlp",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    return model


def train_tensorflow_model(
    data: NeuralData,
    *,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 20,
    seed: int = 42,
) -> TensorFlowTrainingResult:
    """Train with Adam/MSE, validation early stopping and best-weight restore."""
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_tensorflow_model(
        input_dim=data.X_train.shape[1], learning_rate=learning_rate
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=1e-4,
        restore_best_weights=True,
    )
    fit_history = model.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_validation, data.y_validation),
        epochs=epochs,
        batch_size=min(batch_size, len(data.X_train)),
        shuffle=True,
        callbacks=[early_stopping],
        verbose=0,
    )

    training_losses = fit_history.history["loss"]
    validation_losses = fit_history.history["val_loss"]
    history = [
        {
            "epoch": float(index + 1),
            "train_loss": float(training_losses[index]),
            "validation_loss": float(validation_losses[index]),
        }
        for index in range(len(training_losses))
    ]
    best_epoch = int(np.argmin(validation_losses)) + 1
    predictions = model.predict(data.X_test, verbose=0).reshape(-1)
    metrics = regression_metrics(data.y_test.reshape(-1), predictions)

    return TensorFlowTrainingResult(
        model=model,
        history=history,
        predictions=predictions,
        metrics=metrics,
        best_epoch=best_epoch,
    )
