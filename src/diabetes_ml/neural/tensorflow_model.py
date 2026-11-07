"""TensorFlow / Keras MLP implementation with Early Stopping."""

from __future__ import annotations

from typing import Dict, List
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TensorFlowTrainer:
    """Manages TensorFlow/Keras model training and evaluation."""

    def __init__(
        self,
        in_features: int = 10,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 200,
        patience: int = 15,
        dropout_p: float = 0.1,
        random_state: int = 42,
    ):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Install tensorflow to use TensorFlowTrainer.")

        self.in_features = in_features
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.dropout_p = dropout_p
        self.random_state = random_state

        tf.keras.utils.set_random_seed(random_state)
        self.model = self._build_model()
        self.history_records: List[Dict[str, float]] = []

    def _build_model(self) -> keras.Model:
        model = keras.Sequential([
            layers.Input(shape=(self.in_features,)),
            layers.Dense(16, activation="relu"),
            layers.Dropout(self.dropout_p),
            layers.Dense(8, activation="relu"),
            layers.Dropout(self.dropout_p),
            layers.Dense(1, activation="linear"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss="mean_squared_error",
        )
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TensorFlowTrainer:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
            mode="min",
        )

        hist = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
            verbose=0,
        )

        for ep, (tr, val) in enumerate(zip(hist.history["loss"], hist.history["val_loss"]), start=1):
            self.history_records.append({
                "epoch": ep,
                "train_mse": float(tr),
                "val_mse": float(val),
            })

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X, verbose=0)
        return np.asarray(preds, dtype=np.float64).ravel()
