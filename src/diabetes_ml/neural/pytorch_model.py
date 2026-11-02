"""A small PyTorch multilayer perceptron for continuous-value prediction."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from diabetes_ml.evaluation import regression_metrics
from diabetes_ml.neural.neural_data import NeuralData


class DiabetesMLP(nn.Module):
    """Feed-forward network: input -> 16 ReLU -> 8 ReLU -> one output."""

    def __init__(self, input_dim: int = 10) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass
class PyTorchTrainingResult:
    model: DiabetesMLP
    history: list[dict[str, float]]
    predictions: np.ndarray
    metrics: dict[str, float]
    best_epoch: int


def train_pytorch_model(
    data: NeuralData,
    *,
    epochs: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 20,
    seed: int = 42,
) -> PyTorchTrainingResult:
    """Train with Adam/MSE, monitor validation loss and restore best weights."""
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DiabetesMLP(input_dim=data.X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train = torch.from_numpy(data.X_train)
    y_train = torch.from_numpy(data.y_train)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=min(batch_size, len(X_train)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    X_validation = torch.from_numpy(data.X_validation)
    y_validation = torch.from_numpy(data.y_validation)
    X_test = torch.from_numpy(data.X_test)

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        seen = 0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.detach()) * len(batch_features)
            seen += len(batch_features)

        model.eval()
        with torch.no_grad():
            validation_predictions = model(X_validation)
            validation_loss = float(criterion(validation_predictions, y_validation))

        history.append({
            "epoch": float(epoch),
            "train_loss": total_train_loss / max(seen, 1),
            "validation_loss": validation_loss,
        })

        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test).cpu().numpy().reshape(-1)

    metrics = regression_metrics(data.y_test.reshape(-1), test_predictions)
    return PyTorchTrainingResult(
        model=model,
        history=history,
        predictions=test_predictions,
        metrics=metrics,
        best_epoch=best_epoch,
    )
