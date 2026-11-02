# Diabetes Disease Progression: Classical Machine Learning

A small, reproducible Python project that compares baseline and conventional regression models on scikit-learn's Diabetes dataset.

> **Scope:** This is an educational machine-learning experiment using a public, preprocessed benchmark dataset. It is not a clinical study, diagnostic tool, treatment recommendation system, or validated model for individual patient care.

## Research question

How do simple baseline and conventional regression models compare when predicting the dataset's one-year disease-progression measure from ten baseline variables?

## Dataset

The project loads `sklearn.datasets.load_diabetes` directly. The dataset contains 442 observations, ten baseline variables and a quantitative target measured one year after baseline. The features are already preprocessed by scikit-learn; this project does not claim access to raw clinical records, medical images, wearable signals or repeated patient visits.

Dataset documentation: https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset

## Models

- DummyRegressor (mean baseline)
- LinearRegression
- Ridge regression
- Lasso regression
- RandomForestRegressor

All learned preprocessing is placed inside scikit-learn pipelines. Model selection uses cross-validation on the training partition. The held-out test partition is used only for final evaluation.

## Evaluation

The experiment reports MAE, RMSE and R-squared. It records cross-validation results, test-set predictions and residual diagnostics. Metrics are calculated from the actual run; no performance values are asserted in advance.

## Repository layout

```text
diabetes-disease-progression-ml/
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_model_comparison.ipynb
├── src/
│   └── diabetes_ml/
│       ├── __init__.py
│       ├── data.py
│       ├── models.py
│       ├── evaluation.py
│       └── experiment.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_evaluation.py
├── .github/workflows/tests.yml
├── RESULTS.md
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Run the experiment

```bash
python -m diabetes_ml.experiment --output results
```

The command writes machine-readable metrics and test predictions to the chosen output directory. Update `RESULTS.md` with the results from your own run; do not copy unverified numbers into a CV or application.

## Run tests

```bash
pytest
```

## Limitations

- The dataset is small and is a classic benchmark.
- The target is a continuous one-year progression measure; this is regression, not classification.
- The supplied features are processed numerical variables, not raw multimodal clinical data.
- There are no longitudinal visit sequences, image files or wearable streams in this dataset.
- The experiment does not establish clinical utility, causal effects, fairness, external validity or suitability for patient-level decisions.
- Results depend on the split and model configuration; compare models using the recorded protocol rather than selecting a favourable random run.

## What this project demonstrates

It provides inspectable evidence of Python implementation, data inspection, train/test separation, scikit-learn pipelines, conventional regression models, cross-validation and quantitative evaluation. It does **not** by itself demonstrate PyTorch, TensorFlow, neural networks or multimodal learning.

## License and attribution

The dataset is distributed through scikit-learn. Consult the dataset documentation and the scikit-learn project licence for attribution and reuse details.