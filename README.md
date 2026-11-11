# Diabetes Disease Progression: Tabular ML Benchmark with Conformal Uncertainty & Subgroup Error Analysis

[![CI](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational tabular regression benchmark comparing regularised classical models with PyTorch and TensorFlow neural networks on the scikit-learn diabetes dataset. It demonstrates split-conformal prediction intervals, exploratory subgroup error analysis, and model-based feature attribution.

> **Clinical Scope Disclaimer:** This is an academic study using scikit-learn's public, preprocessed Diabetes benchmark (Efron et al., 2004). It is designed to demonstrate rigorous clinical ML evaluation methodology and uncertainty estimation. It is not an approved clinical diagnostic tool or medical decision system for direct patient management.

---

## 🎯 Research Questions & Medical Data Science Objectives

1. **Model Complexity vs. Tabular Generalisation:** How do regularised linear models (Ridge, Lasso) compare with non-linear tree ensembles (Random Forest) and deep neural architectures (PyTorch & TensorFlow MLPs) on a small tabular benchmark with 10 preprocessed features ($N=442$)?
2. **Prediction Uncertainty:** Can split-conformal prediction provide finite-sample marginal coverage under exchangeability without assuming Gaussian residuals?
3. **Exploratory Subgroup Error Analysis:** Does prediction error differ across the two encoded values of the benchmark's standardized sex feature? This is a limited audit, not a comprehensive fairness evaluation.
4. **Model Interpretation:** Which input features contribute most to the predictions of the fitted linear and tree-based models? These attributions are associational, not causal.

---

## 🔬 Methodological Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │ sklearn.datasets.load_diabetes (N = 442)     │
                    │ 10 preprocessed physiological biomarkers     │
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
* Classical estimators are fitted on 282 observations and calibrated on a disjoint 71-observation subset.
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

### 4. Model-based Feature Attribution
* Linear-model coefficients summarize the direction and magnitude of fitted associations.
* For a linear model, baseline-relative feature contributions are coefficient × feature deviation from the model-fitting reference mean; these contributions sum to the prediction difference from that reference.
* For non-linear models, the demo replaces one feature at a time with its reference value and records the prediction change. These perturbation values may overlap and are not guaranteed to sum to the total prediction change.
* These methods are not SHAP, neural input-gradient attribution, causal inference, or a validated clinical explanation method.

### 3. Exploratory Subgroup Error Analysis
* Compares absolute prediction errors across the two encoded values of the standardized `sex` feature; this is exploratory and does not establish fairness.
* Computes Max Disparity: $|MAE_{group1} - MAE_{group2}|$.
* Uses an exploratory two-group permutation test (1,000 resamples) for mean absolute-error disparity, preserving group sizes and applying a plus-one Monte Carlo p-value correction. A p-value does not establish fairness or prove the absence of bias.

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

The end-to-end run and its generated CSV/JSON artifact are documented in [RESULTS.md](RESULTS.md). The workflow also completed the test suite and conventional/neural experiment successfully.
 
For in-depth analysis, biomarker rankings, and training curves, see [RESULTS.md](RESULTS.md).

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
pip install -r requirements.txt
pip install -e .
```

To include PyTorch, TensorFlow, and Streamlit:
```bash
pip install -r requirements-neural.txt
```

### 2. Run Experiments

**Classical Benchmark with Conformal Uncertainty & Fairness:**
```bash
python -m diabetes_ml.experiment --output results
```

**Full Comparative Run (including PyTorch & TensorFlow MLPs):**
```bash
python -m diabetes_ml.neural.neural_experiment --output results --epochs 200 --patience 20
```

### 3. Launch the Educational Research Demo
```bash
streamlit run demo/app.py
```

### 4. Run Automated Test Suite
```bash
pytest -v --cov=diabetes_ml tests/
```

### 5. Docker Deployment

Build and run the lightweight Streamlit research demo image:

```bash
docker build -t diabetes-progression-demo .
docker run --rm -p 8501:8501 diabetes-progression-demo
```

The Docker image installs the project's core dependencies plus the Streamlit demo extra. It intentionally excludes PyTorch, TensorFlow, SHAP, and test-only packages; install `requirements-neural.txt` separately when running the neural experiments outside this container. A GitHub Actions workflow builds the image and checks the Streamlit health and HTTP endpoints.

---

## 🔬 Limitations & Future Horizons in Medical AI

1. **Tabular benchmark scope:** This project uses a small, public, preprocessed 10-feature dataset. It does not contain multimodal imaging, longitudinal EHR, raw physiological signals, or genomic measurements.
2. **Single-split evaluation:** The published metrics are from one seeded split and should be complemented with repeated nested validation and, ideally, external validation.
3. **No clinical validation:** This is a research/educational benchmark and not a diagnostic or clinical decision-support system.

---

## 📜 Citation & License

This project is licensed under the MIT License.
Dataset attribution: Efron, Hastie, Johnstone, and Tibshirani (2004), *Least Angle Regression*, Annals of Statistics.
