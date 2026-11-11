"""PyTorch MLP implementation with early stopping and MC-dropout uncertainty."""

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
        """Small multilayer perceptron for tabular regression."""
        def __init__(self, in_features: int = 10, dropout_p: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 16), nn.ReLU(), nn.Dropout(p=dropout_p),
                nn.Linear(16, 8), nn.ReLU(), nn.Dropout(p=dropout_p),
                nn.Linear(8, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


class PyTorchTrainer:
    """Train the MLP; optional target scaling is learned from fitting targets only."""

    def __init__(
        self, in_features: int = 10, lr: float = 1e-3, batch_size: int = 32,
        epochs: int = 200, patience: int = 15, dropout_p: float = 0.1,
        random_state: int = 42, standardize_target: bool = False,
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
        self.standardize_target = bool(standardize_target)
        self.target_mean_: float | None = None
        self.target_scale_: float = 1.0
        torch.manual_seed(random_state)
        self.model = DiabetesMLP(in_features=in_features, dropout_p=dropout_p)
        self.best_state_dict = None
        self.history: List[Dict[str, float]] = []

    def _transform_target(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=np.float32)
        if self.standardize_target:
            return (values - self.target_mean_) / self.target_scale_
        return values

    def _inverse_target(self, y: np.ndarray) -> np.ndarray:
        if self.standardize_target:
            return y * self.target_scale_ + self.target_mean_
        return y

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> "PyTorchTrainer":
        """Fit using training-only target scaling and validation early stopping."""
        y_train_values = np.asarray(y_train, dtype=np.float32)
        y_val_values = np.asarray(y_val, dtype=np.float32)
        if self.standardize_target:
            self.target_mean_ = float(np.mean(y_train_values))
            scale = float(np.std(y_train_values))
            self.target_scale_ = scale if scale > 0.0 else 1.0
        y_train_fit = self._transform_target(y_train_values)
        y_val_fit = self._transform_target(y_val_values)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        train_ds = TensorDataset(
            torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
            torch.from_numpy(y_train_fit.astype(np.float32)),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        X_val_t = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
        y_val_t = torch.from_numpy(y_val_fit.astype(np.float32))
        best_val_loss = float("inf")
        patience_counter = 0
        scale_squared = self.target_scale_ ** 2 if self.standardize_target else 1.0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            n_train = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item()) * len(batch_x)
                n_train += len(batch_x)

            self.model.eval()
            with torch.no_grad():
                val_loss_scaled = float(criterion(self.model(X_val_t), y_val_t).item())
            self.history.append({
                "epoch": epoch,
                "train_mse": (train_loss_sum / n_train) * scale_squared,
                "val_mse": val_loss_scaled * scale_squared,
            })
            if val_loss_scaled < best_val_loss:
                best_val_loss = val_loss_scaled
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
        """Return predictions in original target units."""
        self.model.eval()
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            pred = self.model(X_t).cpu().numpy().ravel()
        return self._inverse_target(pred)

    def predict_mc_dropout(self, X: np.ndarray, n_samples: int = 100
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Return MC-dropout mean and SD in original target units."""
        if (isinstance(n_samples, (bool, np.bool_))
                or not isinstance(n_samples, (int, np.integer)) or n_samples < 1):
            raise ValueError("n_samples must be a positive integer.")
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        was_training = self.model.training
        samples = []
        try:
            self.model.train()
            with torch.no_grad():
                for _ in range(int(n_samples)):
                    samples.append(self.model(X_t).cpu().numpy().ravel())
        finally:
            self.model.train(was_training)
        samples_arr = np.asarray(samples)
        mean_pred = self._inverse_target(np.mean(samples_arr, axis=0))
        std_pred = np.std(samples_arr, axis=0) * self.target_scale_
        return mean_pred, std_pred
