"""Run the classical, PyTorch and TensorFlow comparisons on shared test rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from diabetes_ml.data import load_diabetes_frame, make_train_test_split
from diabetes_ml.experiment import run_experiment
from diabetes_ml.neural.neural_data import prepare_neural_data
from diabetes_ml.neural.pytorch_model import train_pytorch_model
from diabetes_ml.neural.tensorflow_model import train_tensorflow_model


def run_neural_experiment(
    output_dir: str | Path = "results",
    *,
    random_state: int = 42,
    epochs: int = 200,
    patience: int = 20,
    batch_size: int = 32,
) -> dict[str, pd.DataFrame]:
    """Run both neural frameworks and add them to the classical comparison."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    classical_result = run_experiment(output_dir=out, random_state=random_state)
    X, y = load_diabetes_frame()
    shared_split = make_train_test_split(X, y, random_state=random_state)
    neural_data = prepare_neural_data(shared_split, random_state=random_state)

    pytorch_result = train_pytorch_model(
        neural_data, epochs=epochs, patience=patience,
        batch_size=batch_size, seed=random_state,
    )
    tensorflow_result = train_tensorflow_model(
        neural_data, epochs=epochs, patience=patience,
        batch_size=batch_size, seed=random_state,
    )

    neural_metrics = pd.DataFrame([
        {
            "model": "pytorch_mlp_16_8",
            "framework": "PyTorch",
            **pytorch_result.metrics,
            "best_epoch": pytorch_result.best_epoch,
        },
        {
            "model": "tensorflow_mlp_16_8",
            "framework": "TensorFlow/Keras",
            **tensorflow_result.metrics,
            "best_epoch": tensorflow_result.best_epoch,
        },
    ])
    neural_metrics.to_csv(out / "neural_metrics.csv", index=False)

    history_frames = []
    for framework, result in (
        ("PyTorch", pytorch_result),
        ("TensorFlow/Keras", tensorflow_result),
    ):
        frame = pd.DataFrame(result.history)
        frame.insert(0, "framework", framework)
        history_frames.append(frame)
    pd.concat(history_frames, ignore_index=True).to_csv(
        out / "neural_training_history.csv", index=False
    )

    prediction_frames = []
    for framework, result in (
        ("PyTorch", pytorch_result),
        ("TensorFlow/Keras", tensorflow_result),
    ):
        actual = neural_data.y_test.reshape(-1)
        prediction_frames.append(pd.DataFrame({
            "row_index": neural_data.test_indices,
            "framework": framework,
            "actual": actual,
            "predicted": result.predictions,
            "residual": actual - result.predictions,
        }))
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        out / "neural_test_predictions.csv", index=False
    )

    classical_metrics = classical_result["metrics"].copy()
    classical_metrics.insert(1, "framework", "scikit-learn")
    classical_comparison = classical_metrics[["model", "framework", "mae", "rmse", "r2"]]
    neural_comparison = neural_metrics[["model", "framework", "mae", "rmse", "r2"]]
    combined = pd.concat([classical_comparison, neural_comparison], ignore_index=True)
    combined.to_csv(out / "model_comparison.csv", index=False)

    scaler_metadata = {
        "feature_names": list(neural_data.feature_names),
        "mean": neural_data.scaler.mean_.tolist(),
        "scale": neural_data.scaler.scale_.tolist(),
        "fit_partition": "neural training subset only",
    }
    (out / "neural_feature_scaler.json").write_text(
        json.dumps(scaler_metadata, indent=2), encoding="utf-8"
    )

    import torch
    import tensorflow as tf

    torch.save(pytorch_result.model.state_dict(), out / "pytorch_mlp_state_dict.pt")
    tensorflow_result.model.save(out / "tensorflow_mlp.keras")

    metadata = {
        "dataset": "sklearn.datasets.load_diabetes",
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "outer_split": "80% training / 20% held-out test",
        "neural_validation": "20% of the outer training partition",
        "random_state": random_state,
        "neural_training_samples": int(len(neural_data.X_train)),
        "neural_validation_samples": int(len(neural_data.X_validation)),
        "test_samples": int(len(neural_data.X_test)),
        "architecture": [int(X.shape[1]), 16, 8, 1],
        "hidden_activation": "ReLU",
        "optimizer": "Adam",
        "loss": "Mean Squared Error",
        "early_stopping": {"monitor": "validation loss", "patience": patience},
        "maximum_epochs": epochs,
        "batch_size": batch_size,
        "pytorch_version": str(torch.__version__),
        "tensorflow_version": str(tf.__version__),
        "test_set_used_for_training_or_early_stopping": False,
    }
    (out / "neural_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "classical_metrics": classical_metrics,
        "neural_metrics": neural_metrics,
        "model_comparison": combined,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results", help="Directory for experiment outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data partitions.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum neural training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training mini-batch size.")
    args = parser.parse_args()
    result = run_neural_experiment(
        output_dir=args.output,
        random_state=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    print(result["model_comparison"].to_string(index=False))
    print(f"\nExperiment outputs written to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
