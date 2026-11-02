# Experimental results

This file is a reporting template. Populate it only from a completed run; do not insert estimated performance values.

## Data and protocol

- Dataset: sklearn.datasets.load_diabetes (442 observations; 10 processed numerical features)
- Outer split: 80% training / 20% held-out test, random state 42
- Classical models: five-fold shuffled cross-validation on outer training data; final metrics on the held-out test set
- Neural split: 20% of the outer training partition is reserved for validation
- Neural preprocessing: StandardScaler is fitted on the neural training subset only
- Both neural frameworks use identical arrays and the same held-out test rows
- Neural architecture: input → 16 ReLU → 8 ReLU → 1 linear output
- Neural optimiser/loss: Adam / mean-squared error
- Neural stopping: validation-loss early stopping with best weights restored

## Run commands

~~~bash
python -m diabetes_ml.experiment --output results
python -m diabetes_ml.neural.neural_experiment --output results
~~~

## Combined held-out test results

Copy values from results/model_comparison.csv after a completed run.

| Model | Framework | Test MAE | Test RMSE | Test R² |
|---|---|---:|---:|---:|
| Mean baseline | scikit-learn | — | — | — |
| Linear Regression | scikit-learn | — | — | — |
| Ridge | scikit-learn | — | — | — |
| Lasso | scikit-learn | — | — | — |
| Random Forest | scikit-learn | — | — | — |
| MLP (16, 8) | PyTorch | — | — | — |
| MLP (16, 8) | TensorFlow/Keras | — | — | — |

## Interpretation

After a completed run, inspect training and validation loss, selected epochs, held-out predictions and errors. Record whether neural models outperform conventional models; do not assume that they will. Keep the run metadata with any reported results.

The experiment writes model metrics, neural metrics, predictions, training histories, framework versions, scaler statistics and trained model files to the output directory.
