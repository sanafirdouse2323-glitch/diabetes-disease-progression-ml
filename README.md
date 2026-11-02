# Diabetes Disease Progression: Classical and Neural Machine Learning

A reproducible Python study comparing conventional regression models with small multilayer perceptrons implemented independently in PyTorch and TensorFlow/Keras.

> **Scope:** This is an educational experiment using scikit-learn's public, preprocessed Diabetes benchmark. It is not a clinical study, diagnostic tool, treatment recommendation system, or validated model for individual patient care.

## Dataset and research question

The project loads sklearn.datasets.load_diabetes: 442 observations, ten baseline variables and a continuous target measured one year after baseline. The features are already preprocessed. There are no raw images, wearable signals or repeated patient visits in this dataset.

Dataset documentation: https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset

The research question is how a mean baseline, conventional regression models and small neural networks compare when predicting the benchmark's one-year disease-progression measure.

## Models

**Conventional scikit-learn:** DummyRegressor, LinearRegression, Ridge, Lasso and RandomForestRegressor.

**Neural networks:** two independently implemented MLPs with the same architecture:
- 10 input features
- Dense layer with 16 ReLU units
- Dense layer with 8 ReLU units
- One linear output

The PyTorch and TensorFlow/Keras versions use Adam, mean-squared-error loss, validation-loss early stopping and the same train/validation/test rows.

## Evaluation design

- Create an 80% training / 20% held-out test split.
- Run conventional model cross-validation only inside the training partition.
- Split the training partition again into a neural training subset and validation subset.
- Fit the neural feature scaler on the neural training subset only.
- Give both frameworks the exact same scaled arrays.
- Do not use the held-out test partition for neural training, early stopping or model selection.
- Report held-out MAE, RMSE and R-squared.

The same test-row indices are used for conventional and neural comparisons when the default seed is retained.

## Setup

Python 3.11 is recommended for the combined framework workflow. The classical project supports Python 3.10 or newer.

~~~bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-neural.txt
pip install -e .
~~~

The neural requirements install CPU-capable packages. GPU acceleration is not required for this small experiment.

## Run experiments

Conventional models:

~~~bash
python -m diabetes_ml.experiment --output results
~~~

Both neural implementations and the full comparison:

~~~bash
python -m diabetes_ml.neural.neural_experiment --output results
~~~

Optional controls:

~~~bash
python -m diabetes_ml.neural.neural_experiment --output results --seed 42 --epochs 200 --patience 20 --batch-size 32
~~~

The combined command writes model_metrics.csv, neural_metrics.csv, model_comparison.csv, neural_training_history.csv, neural_test_predictions.csv, neural_run_metadata.json, neural_feature_scaler.json, pytorch_mlp_state_dict.pt and tensorflow_mlp.keras.

## Tests

~~~bash
pytest
~~~

Tests for optional frameworks are skipped when a framework is not installed. Install requirements-neural.txt to run both framework tests.

## What this project demonstrates

The code demonstrates Python data handling, scikit-learn pipelines and conventional regression, neural-network architecture definition, PyTorch autograd and optimisation, Keras training, validation-based early stopping and quantitative evaluation. A successful neural CI run verifies that both implementations train and evaluate in the CI environment.

The repository alone does not establish expert-level mastery. Be prepared to explain the layers, loss, optimiser, backpropagation, validation split, early stopping and observed results.

## Limitations

- The dataset is small and is a classic benchmark.
- The target is a continuous one-year progression measure; this is regression, not classification.
- Features are processed numerical variables, not raw multimodal clinical data.
- There are no longitudinal visit sequences, image files or wearable streams.
- Neural models may perform worse than simpler models; that result must be reported honestly.
- This experiment does not establish clinical utility, causal effects, fairness, external validity or patient-level suitability.
- This tabular dataset does **not** demonstrate multimodal learning. Any multimodal claim needs separate evidence from a project that processes and combines distinct modalities.

## Dataset attribution

The dataset is distributed through scikit-learn. Consult the dataset documentation and scikit-learn licence for attribution and reuse details.
