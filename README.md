# Diabetes Disease Progression: Machine-Learning Experiments

[![CI](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

I built this project to compare different machine-learning methods on a small diabetes disease-progression dataset. I wanted to see how ordinary regression models, tree-based models and neural networks perform on the same task, and whether the more complicated methods actually give better results.

I also used the project to work through several parts of a complete machine-learning experiment: keeping training and test data separate, choosing model settings, estimating prediction uncertainty, looking at errors across groups, and recording results so that the experiments can be repeated.

This is a regression project. It predicts a continuous disease-progression measure from the dataset; it does not diagnose diabetes.

**Project documents:** [Research brief](docs/RESEARCH_BRIEF.md) · [Detailed results](RESULTS.md) · [PhD demonstration guide](docs/PHD_DEMO_GUIDE.md)

## Dataset and purpose

The experiments use `sklearn.datasets.load_diabetes`, a public dataset with 442 observations and 10 preprocessed baseline features. The target is a continuous measure of disease progression one year after baseline.

The dataset is useful for testing and explaining regression methods, but it is small and already preprocessed. It is not a collection of raw hospital records, medical images, genomic data or longitudinal electronic health records. The interactive demo uses synthetic standardised feature values rather than real patient measurements.

The project is for learning and research demonstration. It is not a diagnostic tool or a clinical decision-support system, and the results have not been validated on an independent clinical cohort.

## What I wanted to examine

The main questions were:

1. How do linear regression models compare with tree-based models and neural networks on this small tabular dataset?
2. Does choosing a model through validation improve on using a simpler, fixed model?
3. Can split-conformal prediction give useful prediction intervals, and what assumptions are needed for its coverage statement?
4. Do prediction errors differ between the two encoded values of the dataset's standardised `sex` feature?
5. What can model coefficients, feature importance and simple feature perturbations tell us about how a fitted model makes predictions?

These questions are about model performance and evaluation on this dataset. They do not establish medical or biological conclusions.

## How the experiments are organised

I kept model fitting, validation, conformal calibration and final testing separate where the experiment required those roles.

### Single-run comparison

The initial comparison uses one seeded 80/20 split: 353 observations for the training side and 89 for the held-out test set.

For the classical models, 282 observations are used for fitting and 71 for conformal calibration. For the neural models, the training side is divided into 225 fitting observations, 57 validation observations for early stopping, and 71 separate calibration observations. The neural feature scaler is fitted only on the 225 fitting observations.

The 89 test observations are kept out of fitting, early stopping and calibration. The single-run comparison is useful for showing the workflow, but its scores describe only that particular split.

### Repeated-seed comparisons

I also ran experiments over ten seeds (42–51). Each run creates a new outer training/test split and keeps a separate calibration partition inside the training data.

For a more direct comparison between classical and neural models, I ran a matched-sample version. Both use 225 fitting observations, 57 reserved validation observations, 71 calibration observations and 89 test observations per seed. The classical models do not use the reserved validation observations.

The test partitions overlap across seeds because the same 442 observations are split repeatedly. For that reason, the reported means and standard deviations describe how results vary across these splits. They are not confidence intervals, and they do not come from ten independent datasets.

### Nested cross-validation and model-family selection

A separate experiment compares seven predeclared regression model families using five outer folds repeated three times, with three inner folds.

Tuning and model-family selection take place within the outer training data. Preprocessing for scale-sensitive models is placed inside the relevant scikit-learn pipeline, so scaling is learned from training folds rather than from the full dataset.

The experiment reports two different things:

- **Per-family results:** each candidate family is tuned using inner folds and evaluated on the corresponding outer fold.
- **Selection-procedure results:** the family with the lowest inner-fold MAE is selected within each outer training partition, refitted, and evaluated on that partition's untouched outer fold.

The second result measures the model-selection procedure. It avoids choosing a supposed winner by looking at outer-test scores first. Since the outer cross-validation is repeated, observations can appear in test folds in more than one repeat. The standard deviations are therefore descriptive, not independent-sample confidence intervals.

### Split-conformal prediction intervals

For the experiments that include conformal prediction, I use absolute calibration residuals:

`s_i = |y_i - \\hat{y}(x_i)|`

The finite-sample rank is:

`k = ceil((n_cal + 1) * (1 - alpha))`

The corresponding ordered calibration residual is used as the interval threshold. If the required rank is larger than the number of calibration observations, the threshold is positive infinity.

The resulting interval is centred on the model prediction. Under exchangeability, split conformal prediction gives a marginal coverage guarantee over future observations. This is not a probability statement about one particular patient, and it does not guarantee coverage for every subgroup or under distribution shift.

### Subgroup error and feature analysis

I compare absolute prediction errors across the two encoded values of the standardised `sex` feature. The analysis reports the difference in group MAE and uses a permutation test with 1,000 resamples.

This is a limited exploratory check. It is not a full fairness assessment, and a non-significant result does not prove that a model has no bias.

For model interpretation, I report fitted linear-model coefficients and Random Forest feature importance. The demo also shows reference-relative linear contributions and one-feature-at-a-time perturbations for the tree model. These describe model behaviour on this dataset; they are not causal explanations or clinically validated explanations. The tree perturbation values are not SHAP values and are not guaranteed to add up to the full prediction difference.

## Results

The figures below come from different experiments and split designs. They should not be combined as if they were measured under one common evaluation protocol.

### One seeded held-out test split

The following results use the same 89-observation test set:

| Model | Test MAE | Test RMSE | Test R² | Observed 90%-nominal coverage | Mean interval width |
|---|---:|---:|---:|---:|---:|
| Lasso | 42.703 | 52.941 | 0.471 | 88.8% | 170.445 |
| Linear Regression | 42.908 | 53.952 | 0.451 | 84.3% | 164.600 |
| Random Forest | 43.729 | 53.875 | 0.452 | 94.4% | 190.048 |
| Ridge | 46.564 | 56.151 | 0.405 | 89.9% | 181.196 |
| PyTorch MLP (16, 8) | 51.538 | 65.382 | 0.193 | 89.9% | 219.819 |
| TensorFlow MLP (16, 8) | 50.458 | 66.376 | 0.168 | 94.4% | 269.726 |
| Mean Baseline | 63.529 | 72.899 | -0.003 | 93.3% | 240.433 |

Lasso had the lowest test MAE and RMSE on this split. That does not mean it will be the best model on every new sample. The observed coverage values are the results on this particular test set; they are not required to equal the 90% nominal target on every realised split.

### Ten-seed matched-sample comparison

When the classical and neural models use the same number of fitting observations, the mean test MAE was:

| Model | Mean test MAE |
|---|---:|
| Linear Regression | 42.899 |
| Lasso | 42.927 |
| Random Forest | 44.858 |
| Ridge | 49.015 |
| PyTorch MLP (default) | 53.964 |
| TensorFlow MLP (default) | 57.204 |

Linear Regression and Lasso performed similarly and had lower mean test MAE than the default neural models in this experiment. This is a descriptive comparison on repeated, overlapping hold-out splits; it is not a formal significance test.

The classical repeated-seed results are available in the [GitHub Actions run and its downloadable artifacts](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/42936695778). The corresponding [neural repeated-seed run](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/42936695423) contains the neural results and metadata.

### Model tuning and nested selection

A separate validation-selected PyTorch tuning experiment reported mean test MAE of 44.934, compared with 53.963 for its default control. This brought the tuned network closer to the classical models, but did not show that it was better than Linear Regression or Lasso. Details are in [Section 8 of RESULTS.md](RESULTS.md#8-controlled-validation-selected-pytorch-hyperparameter-study).

In the nested repeated-CV experiment, the mean outer-fold MAE for the fixed Linear Regression family strategy was 44.483. The inner-fold family-selection procedure had mean outer-fold MAE of 44.854. The selector therefore did not improve on the fixed Linear Regression strategy in this experiment. The difference is small, and the results are descriptive rather than a formal significance test. See [Section 11 of RESULTS.md](RESULTS.md#11-nested-repeated-cv-comparison-and-inner-fold-family-selection).

A further PyTorch experiment tested target standardisation using statistics from the fitting data only. The validation-selected model had mean test MAE of 43.408, compared with 44.086 for its paired default control trained under the same 300-epoch and 25-epoch-patience budget. The mean paired MAE change was -0.677 target units; the selected model had lower MAE on eight of the ten seeds and higher MAE on two. This is a modest descriptive difference, not evidence that neural networks are generally better. See [Section 12 of RESULTS.md](RESULTS.md#12-target-standardised-neural-training-study).

The detailed tables, fold-level results, metadata and experiment provenance are in [RESULTS.md](RESULTS.md).

## Run the project

### 1. Create an environment and install the package

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# Linux or macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

### 2. Install optional packages when needed

For the PyTorch and TensorFlow experiments:

```bash
pip install -e ".[neural]"
```

For the Streamlit demo:

```bash
pip install -e ".[demo]"
```

For tests:

```bash
pip install -e ".[dev]"
```

### 3. Run the experiments

Classical benchmark, including conformal intervals and subgroup error analysis:

```bash
python -m diabetes_ml.experiment --output results
```

Single-run comparison with PyTorch and TensorFlow:

```bash
python -m diabetes_ml.neural.neural_experiment --output results --epochs 200 --patience 20
```

This single-run command is not a matched-sample comparison: the classical models use 282 fitting observations, while the neural models use 225 fitting observations and 57 validation observations. Use the repeated-seed matched-sample workflows when comparing those model families.

Repeated-seed classical validation:

```bash
python -m diabetes_ml.repeated_validation --output repeated-validation
```

To reserve the same 57 validation observations as the neural models:

```bash
python -m diabetes_ml.repeated_validation --output matched-validation --val-fraction 0.20
```

Repeated-seed PyTorch and TensorFlow validation:

```bash
python -m diabetes_ml.repeated_neural_validation --output repeated-neural-validation
```

Controlled PyTorch hyperparameter study:

```bash
python -m diabetes_ml.repeated_neural_tuning --output repeated-neural-tuning
```

Nested tabular benchmark with inner-fold model-family selection:

```bash
python -m diabetes_ml.nested_tabular_benchmark --output nested-tabular-benchmark
```

The commands write their metrics and experiment metadata to the output directories. The repeated-validation summaries include means, sample standard deviations, medians and ranges. Remember that the repeated test partitions overlap, so these standard deviations describe split-to-split variation; they are not confidence intervals.

### 4. Open the demo

```bash
streamlit run demo/app.py
```

The demo lets you inspect a prediction, its split-conformal interval, feature-attribution views and subgroup-error results. Its input controls use synthetic standardised values. Do not enter real patient measurements or use the demo to guide diagnosis, treatment or care.

### 5. Run the tests

```bash
pytest -v --cov=diabetes_ml tests/
```

The test suite checks the package and experiment code. It also checks notebook structure, syntax and imports against the current package API. The separate notebook workflow executes the notebooks in GitHub Actions.

## Notebooks, CI and Docker

The notebooks cover dataset exploration, the classical benchmark, and the PyTorch/TensorFlow baselines. Install the neural extras before running the neural notebook.

GitHub Actions runs the automated checks and the relevant experiment workflows. The notebook workflow executes the notebooks on changes to notebooks or their supporting source and configuration files, and on pushes to `main`.

The Docker image runs the Streamlit research demo:

```bash
docker build -t diabetes-progression-demo .
docker run --rm -p 8501:8501 diabetes-progression-demo
```

The image installs the core project dependencies and the demo extra. It does not include PyTorch, TensorFlow, SHAP or test-only packages. The container runs as an unprivileged user. The Docker workflow checks the runtime user, starts the application and tests its health and HTTP endpoints.

## Limitations and what I would do next

There are important limits to keep in mind:

- The dataset has only 442 observations and 10 preprocessed features.
- The experiments do not include raw images, longitudinal health records, genomic data or a new patient cohort.
- Repeated hold-out and repeated outer-fold test sets overlap. Their standard deviations are descriptive, not independent-sample confidence intervals.
- The conformal coverage statement depends on exchangeability. It does not guarantee subgroup coverage or performance after distribution shift.
- The subgroup analysis is narrow and does not establish fairness.
- Feature coefficients and importance scores describe fitted models; they do not show cause and effect.
- The results do not establish clinical validity, patient benefit or readiness for use in a hospital.

If I extend this work, I would first define a clinically meaningful prediction task and the point in time at which the prediction should be made. I would then need a larger, appropriately governed dataset, patient-level and temporal splits where suitable, and evaluation on an independent cohort. Depending on the research question and available data, I would also consider longitudinal or multimodal inputs and work with clinical experts to decide which outcomes and error measures matter.

## References and licence

Dataset: Efron, Hastie, Johnstone and Tibshirani (2004), *Least Angle Regression*, Annals of Statistics. The dataset is provided through scikit-learn's `load_diabetes` function.

This project is licensed under the MIT License.
