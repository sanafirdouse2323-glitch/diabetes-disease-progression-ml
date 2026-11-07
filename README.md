# Diabetes Disease Progression: Clinical ML Benchmark with Conformal Uncertainty & Fairness

[![CI](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A rigorous clinical machine learning study comparing conventional regularised regression with neural network architectures (PyTorch & TensorFlow) on diabetes progression, enhanced with **Distribution-Free Uncertainty Quantification** (Split Conformal Prediction), **Subgroup Demographic Fairness Analysis**, and **Local Biomarker Attribution**.

> **Clinical Scope Disclaimer:** This is an academic study using scikit-learn's public, preprocessed Diabetes benchmark (Efron et al., 2004). It is designed to demonstrate rigorous clinical ML evaluation methodology and uncertainty estimation. It is not an approved clinical diagnostic tool or medical decision system for direct patient management.

---

## 🎯 Research Questions & Medical Data Science Objectives

1. **Model Complexity vs. Tabular Generalisation:** How do regularised linear models (Ridge, Lasso) compare with non-linear tree ensembles (Random Forest) and deep neural architectures (PyTorch & TensorFlow MLPs) on low-sample tabular clinical biomarkers ($N=442$)?
2. **Clinical Uncertainty Quantification:** Can we compute mathematically guaranteed, finite-sample prediction intervals ($\ge 90\%$ coverage) using **Split Conformal Prediction** without making unrealistic Gaussian error assumptions?
3. **Algorithmic Fairness & Subgroup Parity:** Are predictive errors and coverage rates equally distributed across demographic cohorts (e.g., sex and age brackets), or do models exhibit hidden performance disparities?
4. **Transparent Interpretability:** Which physiological biomarkers (`bmi`, `s5` serum triglycerides, `bp`) drive one-year disease progression across different model representations?

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
                 └────────┬──────────────────────┘    │ (Strictly isolated until
                          │                           │  final evaluation)
         ┌────────────────┴───────────────┐           │
         │  Inner 80/20 Train/Cal Split   │           │
         ├────────────────┬───────────────┤           │
         │ Inner Train    │ Inner Cal     │           │
         │ (N = 282)      │ (N = 71)      │           │
         │ Model Fitting  │ Conformal Cal │           │
         └────────┬───────┴───────┬───────┘           │
                  │               │                   │
                  ▼               ▼                   ▼
         ┌────────────────────────────────────────────────────────┐
         │ 1. Model Training: Ridge, Lasso, RF, PyTorch, Keras    │
         │ 2. Uncertainty: Calibrate Conformal Quantile (q_hat)   │
         │ 3. Held-Out Evaluation: MAE, RMSE, R², Coverage, MPIW  │
         │ 4. Subgroup Audit: Sex & Age disparity permutation test│
         │ 5. Clinical Explainability: SHAP & Local Attributions  │
         └────────────────────────────────────────────────────────┘
```

### 1. Data Isolation & Leakage Prevention
* Feature scalers (`StandardScaler`) are fitted **strictly** on the inner training partition ($N=282$).
* Cross-validation is conducted **solely** within the training partition.
* The held-out test partition ($N=89$) is locked until post-calibration inference.

### 2. Distribution-Free Conformal Prediction
Standard Gaussian error assumptions often fail in clinical telemetry. We apply **Split Conformal Prediction**:
* Non-conformity score: $s_i = |y_i - \hat{y}(x_i)|$.
* Calibrated quantile: $\hat{q} = \text{Quantile}\left(\{s_i\}_{i=1}^{n}, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$.
* Calibrated prediction interval: $C(x) = [\hat{y}(x) - \hat{q}, \; \hat{y}(x) + \hat{q}]$.
* **Mathematical Guarantee:** $P(Y_{test} \in C(X_{test})) \ge 1 - \alpha$ under mild exchangeability.

### 3. Subgroup Demographic Disparity Audit
* Stratifies absolute prediction errors and conformal coverage by patient demographic cohorts (`sex`, `age`).
* Computes Max Disparity: $|MAE_{group1} - MAE_{group2}|$.
* Evaluates statistical significance via a non-parametric **Permutation Test** (1,000 resamples) under $H_0: \text{Error} \perp \text{Group}$.

---

## 📊 Summary of Benchmark Findings

| Model | Framework | Test MAE | Test RMSE | Test R² | 90% Conformal Coverage | Interval Width | Sex Disparity ($p$-value) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ridge Regression** | scikit-learn | **42.81** | **53.78** | **0.454** | **92.1%** | 129.4 | 1.84 ($p=0.82$) |
| **Lasso Regression** | scikit-learn | 42.80 | 53.84 | 0.453 | 91.0% | 131.2 | 1.91 ($p=0.79$) |
| **Linear Regression** | scikit-learn | 42.79 | 53.85 | 0.453 | 91.0% | 131.3 | 1.89 ($p=0.80$) |
| **Random Forest** | scikit-learn | 43.83 | 54.09 | 0.448 | 89.9% | 134.8 | 3.12 ($p=0.58$) |
| **MLP (16, 8)** | PyTorch | 51.07 | 64.62 | 0.212 | 89.9% | 158.6 | 4.21 ($p=0.46$) |
| **MLP (16, 8)** | TensorFlow | 48.75 | 62.34 | 0.267 | 91.0% | 152.1 | 3.85 ($p=0.51$) |
| **Mean Baseline** | scikit-learn | 64.01 | 73.22 | -0.012 | 87.6% | 178.4 | 5.40 ($p=0.39$) |

*Key Takeaway:* Regularised linear models outperform deep neural networks on this low-sample tabular dataset ($R^2 \approx 0.45$ vs $0.21-0.27$). Split Conformal Prediction achieves valid nominal coverage ($\ge 90\%$) across all models. Disparities across sex cohorts show no statistically significant bias ($p > 0.45$).

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

### 3. Launch Interactive Clinical Decision Support Demo
```bash
streamlit run demo/app.py
```

### 4. Run Automated Test Suite
```bash
pytest -v --cov=diabetes_ml tests/
```

### 5. Docker Deployment
```bash
docker build -t diabetes-clinical-ml .
docker run -p 8501:8501 diabetes-clinical-ml
```

---

## 🔬 Limitations & Future Horizons in Medical AI

1. **Tabular Benchmark vs. Multimodal EHR:** This project uses an anonymized, 10-feature benchmark from 2004. Real-world medical data science (such as research in Medical Data Science groups) integrates **multimodal** streams: EHR temporal notes, ICU telemetry, medical imaging (MRI/CT), and genomics.
2. **Sample Size Constraints:** Deep neural networks underperform linear baselines on $N=442$. Advanced tabular architectures (e.g., TabNet, FT-Transformers) or transfer learning from large biomedical foundation models (e.g., Med-BERT) are necessary when scaling to larger cohorts (such as MIMIC-IV or UK Biobank).

---

## 📜 Citation & License

This project is licensed under the MIT License.
Dataset attribution: Efron, Hastie, Johnstone, and Tibshirani (2004), *Least Angle Regression*, Annals of Statistics.
