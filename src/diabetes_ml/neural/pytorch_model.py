"""PyTorch MLP implementation with Early Stopping and Monte Carlo Dropout uncertainty."""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class DiabetesMLP(nn.Module):
        """Multilayer Perceptron with Dropout for tabular regression."""

        def __init__(self, in_features: int = 10, dropout_p: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 16),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(8, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


class PyTorchTrainer:
    """Manages PyTorch model training, early stopping, and MC-Dropout inference."""

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
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install torch to use PyTorchTrainer.")

        self.in_features = in_features
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.dropout_p = dropout_p
        self.random_state = random_state

        torch.manual_seed(random_state)
        self.model = DiabetesMLP(in_features=in_features, dropout_p=dropout_p)
        self.best_state_dict = None
        self.history: List[Dict[str, float]] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> PyTorchTrainer:
        """Train model with validation early stopping."""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        X_val_t = torch.from_numpy(X_val).float()
        y_val_t = torch.from_numpy(y_val).float()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            n_train = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                train_loss_sum += float(loss.item()) * len(batch_x)
                n_train += len(batch_x)

            epoch_train_loss = train_loss_sum / n_train

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t)
                val_loss = float(criterion(val_preds, y_val_t).item())

            self.history.append({
                "epoch": epoch,
                "train_mse": epoch_train_loss,
                "val_mse": val_loss,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Deterministic inference (evaluation mode)."""
        self.model.eval()
        X_t = torch.from_numpy(X).float()
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy().ravel()
        return preds

    def predict_mc_dropout(
        self,
        X: np.ndarray,
        n_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate predictive variability with Monte Carlo dropout.

        Dropout stays active during repeated inference passes. The returned
        standard deviation is an approximate uncertainty measure, not a
        calibrated prediction interval.
        """
        if (
            isinstance(n_samples, (bool, np.bool_))
            or not isinstance(n_samples, (int, np.integer))
            or n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer.")

        X_t = torch.from_numpy(X).float()
        was_training = self.model.training
        samples = []
        try:
            self.model.train()  # Keep dropout active for stochastic predictions.
            with torch.no_grad():
                for _ in range(int(n_samples)):
                    pred = self.model(X_t).cpu().numpy().ravel()
                    samples.append(pred)
        finally:
            self.model.train(was_training)

        samples_arr = np.asarray(samples)  # (n_samples, N)
        mean_pred = np.mean(samples_arr, axis=0)
        std_pred = np.std(samples_arr, axis=0)
        return mean_pred, std_pred
