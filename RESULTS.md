# Experimental results

This file is intentionally a results template until the experiment has actually been run in a Python environment.

## Protocol

- Dataset: `sklearn.datasets.load_diabetes`
- Split: 80% training / 20% held-out test, random state 42
- Cross-validation: shuffled 5-fold KFold on the training partition only
- Model selection comparison: mean cross-validation MAE
- Final held-out metrics: MAE, RMSE and R-squared
- Models: mean DummyRegressor, Linear Regression, Ridge, Lasso and Random Forest

## Results

Run:

```bash
python -m diabetes_ml.experiment --output results
```

Then transfer the observed model metrics from `results/model_metrics.csv` into this table. Do not enter estimated or invented performance values.

| Model | CV MAE (mean ± SD) | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Mean baseline | — | — | — | — |
| Linear Regression | — | — | — | — |
| Ridge | — | — | — | — |
| Lasso | — | — | — | — |
| Random Forest | — | — | — | — |

## Interpretation

Complete after reviewing the measured output:

- Which model had the lowest cross-validation MAE?
- Did the same model perform best on the held-out test set?
- How much did models improve over the mean baseline?
- What do residuals suggest about prediction errors?
- Are differences large enough to be practically meaningful?

## Reproducibility

The command writes `model_metrics.csv`, `test_predictions.csv` and `run_metadata.json`. The held-out test set is not used to select hyperparameters.
