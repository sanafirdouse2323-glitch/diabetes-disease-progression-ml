# Diabetes Disease Progression: Tabular ML Benchmark with Conformal Uncertainty & Subgroup Error Analysis

[![CI](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational tabular regression benchmark comparing classical regression models with PyTorch and TensorFlow neural networks on the scikit-learn diabetes dataset. It demonstrates split-conformal prediction intervals, exploratory subgroup error analysis, and model-based feature attribution.

> **Scope disclaimer:** This is an educational research benchmark using scikit-learn's public, preprocessed diabetes dataset (Efron et al., 2004). It demonstrates tabular regression, split-conformal uncertainty estimation, exploratory subgroup error analysis, and model-based feature attribution. It is not a clinical diagnostic tool or a validated medical decision system.

---

## 🎯 Research Questions & Benchmark Objectives

1. **Model Complexity vs. Tabular Generalisation:** How do regularised linear models (Ridge, Lasso) compare with non-linear tree ensembles (Random Forest) and deep neural architectures (PyTorch & TensorFlow MLPs) on a small tabular benchmark with 10 preprocessed features ($N=442$)?
2. **Prediction Uncertainty:** Can split-conformal prediction provide finite-sample marginal coverage under exchangeability without assuming Gaussian residuals?
3. **Exploratory Subgroup Error Analysis:** Does prediction error differ across the two encoded values of the benchmark's standardized sex feature? This is a limited audit, not a comprehensive fairness evaluation.
4. **Model Interpretation:** Which input features contribute most to the predictions of the fitted linear and tree-based models? These attributions are associational, not causal.

---

## 🔬 Methodological Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │ sklearn.datasets.load_diabetes (N = 442)     │
                    │ 10 preprocessed baseline features     │
                    └──────────────────────┬───────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        │   Outer Split (80% / 20% Held-out)  │
                        └─────────┬───────────────────┬───────┘
                                  │                   │
                 ┌────────────────┴──────────────┐    │
                 │ Outer Training Split (N = 353)│    │ Held-Out Test Set (N = 89)
                 └────────┬──────────────────────┘    │
                          │                           │
              ┌───────────┴─────────────┐             │
              │ Calibration only (N=71)│             │
              └───────────┬─────────────┘             │
                          │                           │
                  Development (N=282)                 │
                          │                           │
             ┌────────────┴────────────┐               │
             │                        │               │
      Classical model path     Neural model path       │
      Fit: N=282               Fit: N=225              │
      Cal: N=71                Early stop: N=57        │
                              Cal: N=71               │
             │                        │               │
             └────────────────────────┴───────────────┘
                                      │
                                      ▼
                    Final evaluation on untouched N=89
```

### 1. Data Isolation & Leakage Prevention
* In the single-run convenience comparison, classical estimators are fitted on 282 observations and calibrated on a disjoint 71-observation subset. The matched repeated-seed comparison reserves the same 57 validation observations used by the neural models, leaving 225 classical fitting observations.
* Neural models use 225 observations for fitting, 57 for early-stopping validation, and a separate 71 for conformal calibration.
* Neural feature scalers (`StandardScaler`) are fitted **only** on the 225-observation model-fitting subset.
* Classical cross-validation and neural early stopping do not use conformal calibration observations.
* The outer test partition ($N=89$) remains untouched until final evaluation.

### 2. Distribution-Free Conformal Prediction
We apply **Split Conformal Prediction** to regression without assuming Gaussian residuals:
* Non-conformity score: $s_i = |y_i - \hat{y}(x_i)|$.
* Finite-sample rank: $k = \lceil (n_{cal}+1)(1-\alpha) \rceil$; set $\hat{q}$ to the $k$-th sorted calibration score. If $k > n_{cal}$, set $\hat{q}=+\infty$.
* Calibrated prediction interval: $C(x) = [\hat{y}(x) - \hat{q}, \; \hat{y}(x) + \hat{q}]$.
* **Coverage interpretation:** at least $1-\alpha$ marginal coverage over exchangeable future observations; this is not a patient-specific probability statement.

### 3. Exploratory Subgroup Error Analysis
* Compares absolute prediction errors across the two encoded values of the standardized `sex` feature; this is exploratory and does not establish fairness.
* Computes Max Disparity: $|MAE_{group1} - MAE_{group2}|$.
* Uses an exploratory two-group permutation test (1,000 resamples) for mean absolute-error disparity, preserving group sizes and applying a plus-one Monte Carlo p-value correction. A p-value does not establish fairness or prove the absence of bias.


### 4. Model-based Feature Attribution
* Linear-model coefficients summarize the direction and magnitude of fitted associations.
* For a linear model, baseline-relative feature contributions are coefficient × feature deviation from the model-fitting reference mean; these contributions sum to the prediction difference from that reference.
* For non-linear models, the demo replaces one feature at a time with its reference value and records the prediction change. These perturbation values may overlap and are not guaranteed to sum to the total prediction change.
* These methods are not SHAP, neural input-gradient attribution, causal inference, or a validated clinical explanation method.
---

## 📊 Regenerated Benchmark Results

The following values were measured by the corrected end-to-end workflow on one seeded 80/20 outer split (89 held-out observations). They are descriptive for this split, not estimates of performance across new populations.

| Model | Test MAE | Test RMSE | Test R² | Observed 90%-nominal coverage | Mean interval width |
|---|---:|---:|---:|---:|---:|
| **Lasso** | **42.703** | **52.941** | **0.471** | 88.8% | 170.445 |
| Linear Regression | 42.908 | 53.952 | 0.451 | 84.3% | 164.600 |
| Random Forest | 43.729 | 53.875 | 0.452 | 94.4% | 190.048 |
| Ridge | 46.564 | 56.151 | 0.405 | 89.9% | 181.196 |
| PyTorch MLP (16, 8) | 51.538 | 65.382 | 0.193 | 89.9% | 219.819 |
| TensorFlow MLP (16, 8) | 50.458 | 66.376 | 0.168 | 94.4% | 269.726 |
| Mean Baseline | 63.529 | 72.899 | -0.003 | 93.3% | 240.433 |

Lasso had the strongest point-prediction metrics on this particular split. The empirical test coverage is not identical to the 90% nominal target for every model; split-conformal guarantees marginal coverage under exchangeability over future observations, not at least 90% empirical coverage on every realized test set or a patient-specific probability.

A follow-up ten-seed analysis of the classical models (seeds 42–51) found very similar average performance for Linear Regression and Lasso: mean test MAE 42.579 and 42.972, respectively. Linear Regression had slightly lower mean MAE/RMSE and slightly higher mean R²; the difference is descriptive, not evidence of a statistically significant winner. See the full [ten-seed results and methodology in RESULTS.md](RESULTS.md#6-repeated-seed-classical-robustness-check) and the [GitHub Actions run and downloadable CSV/JSON artifact](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/42934789858).

The end-to-end run and its generated CSV/JSON artifact are documented in [RESULTS.md](RESULTS.md). The workflow also completed the test suite and conventional/neural experiment successfully.
 
For in-depth analysis, feature rankings, and training curves, see [RESULTS.md](RESULTS.md).

---

## 💻 Installation & Quickstart

### 1. Setup Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -e .

# Optional: install PyTorch and TensorFlow for neural experiments
pip install -e ".[neural]"

# Optional: install Streamlit for the demo
pip install -e ".[demo]"
```

### 2. Run Experiments

**Classical benchmark with conformal uncertainty and subgroup error analysis:**
```bash
python -m diabetes_ml.experiment --output results
```

**Full Comparative Run (including PyTorch & TensorFlow MLPs):**
```bash
python -m diabetes_ml.neural.neural_experiment --output results --epochs 200 --patience 20
```

This convenience run is not a matched-sample comparison: the classical models fit 282 rows, while each neural model fits 225 rows and uses 57 separate early-stopping validation rows. Use the ten-seed matched-sample workflows below when comparing classical and neural performance directly.

**Repeated-seed robustness analysis (classical models):**
```bash
python -m diabetes_ml.repeated_validation --output repeated-validation
```

The default runs 10 outer splits (seeds 42–51), with a separate calibration partition inside each training split. It writes `per_seed_metrics.csv`, `summary_metrics.csv`, and `validation_metadata.json`. The summary reports means, sample standard deviations, medians, and ranges for point-prediction and conformal metrics. Because repeated hold-out test sets overlap, these standard deviations describe split sensitivity; they are not independent-sample confidence intervals or external validation. To match the neural models' 225-row fitting sample, also run `python -m diabetes_ml.repeated_validation --output matched-validation --val-fraction 0.20`; this reserves 57 rows from the development subset as unused data while keeping the same calibration and test partitions. The neural frameworks are evaluated by the separate repeated-seed neural command below.

**Repeated-seed neural validation (PyTorch and TensorFlow):**
```bash
python -m diabetes_ml.repeated_neural_validation --output repeated-neural-validation
```

This command also defaults to seeds 42–51. It uses separate model-fitting, early-stopping validation, conformal calibration, and outer test partitions for each seed; the scaler is fitted only on the model-fitting subset. It writes per-seed metrics, across-seed summaries, training histories, and metadata. Install the optional neural dependencies first with `pip install -e ".[neural]"`. The across-seed standard deviations remain descriptive because the outer test partitions overlap.

The measured ten-seed comparison is now published in [RESULTS.md](RESULTS.md). With classical and neural models matched to 225 fitting observations, Linear Regression and Lasso had mean test MAE 42.899 and 42.927, versus 53.964 for PyTorch and 57.204 for TensorFlow. This is a benchmark-specific descriptive comparison, not proof of general superiority; the repeated test partitions overlap and no formal significance test was performed. See the [classical ten-seed workflow](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/42936695778) and [neural ten-seed workflow](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/42936695423) for the downloadable artifacts.


**Controlled PyTorch hyperparameter study (validation-selected):**
```bash
python -m diabetes_ml.repeated_neural_tuning --output repeated-neural-tuning
```

This separate study evaluates five predeclared learning-rate/dropout settings over seeds 42–51. It selects a setting using validation MSE within each seed, then calibrates on the separate calibration subset and evaluates both the selected model and the original default configuration on the same untouched test rows. The measured selected model had mean test MAE 44.934 ± 4.931 versus 53.963 ± 3.984 for the default control; mean paired MAE change was −9.030. The higher learning rate (0.003) was selected in all ten seeds. These are descriptive results on overlapping repeated hold-out splits, not confidence intervals or evidence of clinical utility. Full details and workflow artifact are in [RESULTS.md](RESULTS.md#8-controlled-validation-selected-pytorch-hyperparameter-study).

### 3. Launch the Educational Research Demo
```bash
streamlit run demo/app.py
```

### 4. Run Automated Test Suite
Install the development tools, then run the tests:
```bash
pip install -e ".[dev]"
pytest -v --cov=diabetes_ml tests/

# The test suite also checks that every notebook's Python imports match the current package API.
```

### 5. Notebooks

The three notebooks cover dataset exploration, the classical benchmark, and the PyTorch/TensorFlow baselines. Run them from the repository root after installing the core dependencies; install `.[neural]` before running the neural notebook. The automated test suite checks notebook JSON, Python syntax, and imports against the current package API. Separately, the **End-to-end notebook execution** GitHub Actions workflow executes all three notebooks on pull requests that change notebooks or their supporting source/configuration files, and on pushes to `main`; it uploads the executed notebooks and generated experiment outputs as a workflow artifact.

### 6. Docker Deployment

Build and run the lightweight Streamlit research demo image:

```bash
docker build -t diabetes-progression-demo .
docker run --rm -p 8501:8501 diabetes-progression-demo
```

The Docker image installs the project's core dependencies plus the Streamlit demo extra. It intentionally excludes PyTorch, TensorFlow, SHAP, and test-only packages; install the neural extra with `pip install -e ".[neural]"` when running the neural experiments outside this container. The runtime runs as an unprivileged UID (10001), not root. The Docker CI workflow drops Linux capabilities, enables `no-new-privileges`, verifies the runtime UID, exercises the Streamlit script, and checks the health and HTTP endpoints.

---

## Limitations and Next Steps

1. **Tabular benchmark scope:** This project uses a small, public, preprocessed 10-feature dataset. It does not contain multimodal imaging, longitudinal EHR, raw physiological signals, or genomic measurements.
2. **Published results and repeated splits:** The headline table in this document is from one seeded split. Separate ten-seed workflows report classical and neural results; their overlapping hold-out sets provide descriptive split-sensitivity summaries, not independent-sample confidence intervals. Repeated nested validation and, ideally, external validation are still needed.
3. **No clinical validation:** This is a research/educational benchmark and not a diagnostic or clinical decision-support system.

---

## 📜 Citation & License

This project is licensed under the MIT License.
Dataset attribution: Efron, Hastie, Johnstone, and Tibshirani (2004), *Least Angle Regression*, Annals of Statistics.

## Nested tabular model benchmark

A separate nested repeated-cross-validation experiment compares tuned model
families on the full 442-row scikit-learn diabetes progression benchmark.
Each model family is tuned using inner folds drawn only from the corresponding
outer training fold; the outer fold is reserved for evaluation. The compact,
predeclared search covers linear and regularised models, RBF-SVR, Random Forest,
Extra Trees, and histogram gradient boosting. Scale-sensitive estimators use
a pipeline so scaling is fitted within each inner training fold.

Run it with:

```bash
python -m diabetes_ml.nested_tabular_benchmark --output nested-tabular-benchmark
```

The output includes per-outer-fold metrics, model-level descriptive summaries,
selected hyperparameters, and metadata describing the evaluation boundary.
Repeated outer folds overlap across repeats, so the reported standard
deviations are descriptive and are not independent-sample confidence
intervals. This benchmark is separate from split-conformal calibration and
does not establish external validity, clinical utility, or clinical readiness.
