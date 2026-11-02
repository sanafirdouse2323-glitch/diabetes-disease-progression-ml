# Experimental results

## Completed reference run — 2 November 2026

The first end-to-end neural run completed successfully in GitHub Actions on commit c54c5c5f94f220e4097d3b9d2be0b51da4675e85.

- Workflow: [Neural framework checks](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/41379771316)
- Python: 3.11
- PyTorch: 2.15.0+cpu
- TensorFlow: 2.21.0
- Dataset: sklearn.datasets.load_diabetes (442 observations; 10 processed numerical features)
- Seed: 42
- Outer split: 353 training rows and 89 held-out test rows
- Neural split: 282 training rows, 71 validation rows and the same 89 held-out test rows
- Neural architecture: 10 → 16 ReLU → 8 ReLU → 1 linear output
- Optimiser/loss: Adam / mean-squared error
- Neural run limit: 100 epochs, batch size 32, early-stopping patience 12

The neural scaler was fitted only on the 282 neural training rows. Both frameworks received the same training, validation and test arrays. The test rows were not used for training or early stopping.

## Held-out test comparison

These are the measured results from this run, rounded to three decimal places. The metrics are calculated on the same 89 held-out test observations.

| Model | Framework | Test MAE | Test RMSE | Test R² |
|---|---|---:|---:|---:|
| Ridge | scikit-learn | 42.812 | 53.777 | 0.454 |
| Lasso | scikit-learn | 42.795 | 53.836 | 0.453 |
| Linear Regression | scikit-learn | 42.794 | 53.853 | 0.453 |
| Random Forest | scikit-learn | 43.830 | 54.091 | 0.448 |
| Mean baseline | scikit-learn | 64.006 | 73.222 | -0.012 |
| MLP (16, 8) | PyTorch | 51.070 | 64.622 | 0.212 |
| MLP (16, 8) | TensorFlow/Keras | 48.751 | 62.337 | 0.267 |

## Conventional-model cross-validation

Five-fold shuffled cross-validation was performed on the outer training partition only. The mean and standard deviation of MAE were:

| Model | CV MAE (mean ± SD) |
|---|---:|
| Ridge | 44.999 ± 2.021 |
| Lasso | 45.040 ± 1.989 |
| Linear Regression | 45.048 ± 1.982 |
| Random Forest | 47.665 ± 2.693 |
| Mean baseline | 66.867 ± 6.297 |

Ridge had the lowest mean cross-validation MAE. On the held-out test set, Linear Regression had the lowest MAE by a very small margin, while Ridge had the lowest RMSE and highest R² among the compared models. These small differences should not be over-interpreted from one split.

## Neural training behaviour

Both neural models completed the configured maximum of 100 epochs. Their lowest validation losses occurred at epoch 100, so the early-stopping patience condition did not stop either run before the epoch limit.

| Framework | First-epoch validation MSE | Final training MSE | Final / best validation MSE | Best epoch |
|---|---:|---:|---:|---:|
| PyTorch | 34,112.293 | 3,782.428 | 4,450.363 | 100 |
| TensorFlow/Keras | 34,118.262 | 3,604.887 | 4,669.475 | 100 |

Validation loss decreased over the 100 epochs. Training loss was lower than validation loss at the end, leaving a train/validation gap that should be monitored in further experiments. Early stopping is implemented, but it was not triggered before the maximum epoch count in this particular run.

## Interpretation

- Both neural implementations trained successfully and produced finite held-out predictions and metrics.
- TensorFlow/Keras performed better than the PyTorch MLP on this run, but both neural models were worse than the conventional scikit-learn regressors on the held-out test metrics.
- Both neural models improved on the mean-prediction baseline.
- This is consistent with the small dataset size: a compact neural network is not guaranteed to outperform regularised linear models on 442 observations.
- The result demonstrates working framework implementations and model evaluation; it does not prove advanced mastery or clinical usefulness.

## Reproduction

Install the base and neural dependencies, then run:

~~~bash
python -m diabetes_ml.neural.neural_experiment --output results --seed 42 --epochs 100 --patience 12 --batch-size 32
~~~

The run's CSVs, training histories, scaler information, metadata and trained model files were uploaded as a GitHub Actions artifact: [download the run artifact](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/41379771316).

## Important limitations

The scikit-learn Diabetes data are preprocessed tabular benchmark features, not raw medical images, wearable data or repeated patient visits. This experiment is regression on one public dataset, not clinical validation. It does not demonstrate multimodal learning or actual longitudinal model development.
