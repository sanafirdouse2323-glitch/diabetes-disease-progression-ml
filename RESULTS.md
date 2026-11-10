# Previously Reported Experimental Results & ML Audit

> **Reproducibility warning:** All numerical tables and interpretations in this file were produced before the calibration-isolation corrections now in the code. They have not yet been regenerated with the corrected pipeline. Treat them as historical results only; do not cite them as validated outputs of the current implementation until the experiments have been rerun and the generated files reviewed.

This document records the earlier empirical results, uncertainty calibration, subgroup audits, and feature attribution findings across conventional and neural machine learning models.

---

## 1. Experimental Protocol & Partitioning

* **Benchmark Dataset:** `sklearn.datasets.load_diabetes`
  * Sample size: $N = 442$ observations.
  * Input features: $d = 10$ preprocessed physiological baseline variables (`age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`).
  * Continuous target: Quantitative disease progression measured one year post-baseline.
* **Outer Split:** 80% training ($N = 353$) and 20% strictly isolated test set ($N = 89$).
* **Classical path:** the outer training partition is split into 282 model-fitting observations and 71 disjoint calibration observations.
* **Neural path:** 71 observations are reserved for conformal calibration first; the remaining 282 are split into 225 model-fitting and 57 early-stopping validation observations.
* **Neural preprocessing:** scaler statistics ($\mu, \sigma$) are fitted only on the 225 neural model-fitting observations and then applied unchanged to validation, calibration, and test data.
* **Cross-validation:** classical cross-validation is restricted to the 282 model-fitting observations and excludes calibration data.
* **Random Seed:** $42$ (enforcing deterministic row partitioning across all models).

---

## 2. Held-Out Generalisation Performance

Evaluated on the exact same 89 held-out test observations.

| Model | Framework | Test MAE | Test RMSE | Test R² | Pearson $r$ |
|---|---|:---:|:---:|:---:|:---:|
| **Ridge ($\alpha=1.0$)** | scikit-learn | **42.812** | **53.777** | **0.454** | **0.675** |
| **Lasso ($\alpha=0.1$)** | scikit-learn | 42.795 | 53.836 | 0.453 | 0.674 |
| **Linear Regression** | scikit-learn | 42.794 | 53.853 | 0.453 | 0.674 |
| **Random Forest ($B=100$)** | scikit-learn | 43.830 | 54.091 | 0.448 | 0.671 |
| **MLP (16, 8)** | TensorFlow / Keras | 48.751 | 62.337 | 0.267 | 0.548 |
| **MLP (16, 8)** | PyTorch | 51.070 | 64.622 | 0.212 | 0.521 |
| **Mean Baseline** | scikit-learn | 64.006 | 73.222 | -0.012 | 0.000 |

### Previously Reported Cross-Validation (Legacy 353-Observation Protocol)

The following CV values were generated under the earlier protocol, which used all 353 outer-training observations. They are historical and do not represent the current code's cross-validation on the 282-observation model-fitting subset:

| Model | CV MAE (Mean ± SD) | CV RMSE (Mean ± SD) | CV R² (Mean ± SD) |
|---|:---:|:---:|:---:|
| **Ridge** | **44.999 ± 2.021** | **55.031 ± 2.450** | **0.501 ± 0.043** |
| **Lasso** | 45.040 ± 1.989 | 55.082 ± 2.421 | 0.500 ± 0.044 |
| **Linear Regression** | 45.048 ± 1.982 | 55.094 ± 2.418 | 0.500 ± 0.044 |
| **Random Forest** | 47.665 ± 2.693 | 58.214 ± 3.120 | 0.442 ± 0.061 |
| **Mean Baseline** | 66.867 ± 6.297 | 77.420 ± 6.812 | -0.015 ± 0.012 |

---

## 3. Split-Conformal Prediction (Historical Results; Not Yet Revalidated)

The earlier experiment targeted nominal **90% marginal coverage** ($\alpha = 0.10$) using split-conformal prediction. The results below are historical and must be recomputed after the calibration-isolation correction.

| Model | Nominal Coverage | Empirical Test Coverage | Mean Interval Width (MPIW) | Conformal Margin ($\hat{q}$) | Winkler Score |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ridge** | 90.0% | **92.1%** | **129.42** | 64.71 | **162.34** |
| **Linear Regression** | 90.0% | 91.0% | 131.28 | 65.64 | 165.12 |
| **Lasso** | 90.0% | 91.0% | 131.20 | 65.60 | 165.05 |
| **Random Forest** | 90.0% | 89.9% | 134.80 | 67.40 | 171.20 |
| **TensorFlow MLP** | 90.0% | 91.0% | 152.14 | 76.07 | 194.50 |
| **PyTorch MLP** | 90.0% | 89.9% | 158.62 | 79.31 | 206.18 |

### Key Uncertainty Takeaways:
1. **Historical empirical coverage:** These values are single held-out test-set estimates. They do not prove the theoretical coverage result, and they do not represent a patient-specific probability. The finite-sample guarantee concerns marginal coverage under exchangeability of calibration and future test observations.
2. **Historical interval efficiency:** The reported Ridge interval width and Winkler score were computed under the earlier pipeline and must be regenerated before comparing models.

---

## 4. Exploratory Sex-Feature Subgroup Error Analysis

Subgroup error analysis can identify potential performance differences. This earlier audit used the two encoded values of the standardized `sex` feature in the scikit-learn benchmark; it is not a comprehensive demographic fairness evaluation.

| Model | Group 0 MAE ($N=42$) | Group 1 MAE ($N=47$) | Absolute Disparity | Parity Ratio ($\min / \max$) | Permutation $p$-value |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ridge** | 41.83 | 43.67 | 1.84 | 0.958 | 0.824 (Not significant) |
| **Lasso** | 41.79 | 43.70 | 1.91 | 0.956 | 0.792 (Not significant) |
| **Random Forest** | 42.21 | 45.33 | 3.12 | 0.931 | 0.581 (Not significant) |
| **PyTorch MLP** | 48.91 | 53.12 | 4.21 | 0.921 | 0.463 (Not significant) |
| **TensorFlow MLP** | 46.78 | 50.63 | 3.85 | 0.924 | 0.512 (Not significant) |

*Historical hypothesis-test result:* The earlier permutation tests did not reject the null hypothesis at conventional significance levels. This is not evidence that the model is unbiased or fair; the small held-out sample limits the strength of subgroup conclusions.

---

## 5. Feature Attribution & Interpretability (Associational, Not Causal)

Standardized feature attributions from the top-performing model (**Ridge**) compared against **Random Forest** feature importance:

| Biomarker | Description | Ridge Standardized $\beta$ | Relative Impact (%) | Random Forest Importance |
|---|---|:---:|:---:|:---:|
| **`s5`** | Serum Triglycerides (log) | **+24.81** | **23.4%** | **0.294** |
| **`bmi`** | Body Mass Index | **+23.94** | **22.6%** | **0.248** |
| **`bp`** | Mean Blood Pressure | **+15.22** | **14.3%** | **0.098** |
| **`s1`** | Total Serum Cholesterol | -11.08 | 10.4% | 0.052 |
| **`s4`** | Total Cholesterol / HDL | +8.74 | 8.2% | 0.061 |
| **`s2`** | Low-Density Lipoproteins (LDL) | -8.12 | 7.7% | 0.058 |
| **`s3`** | High-Density Lipoproteins (HDL) | -7.53 | 7.1% | 0.071 |
| **`s6`** | Blood Glucose Level | +4.10 | 3.9% | 0.068 |
| **`age`** | Patient Age | +1.64 | 1.5% | 0.042 |
| **`sex`** | Patient Sex | -0.92 | 0.9% | 0.008 |

### Interpretation Limits:
* These values describe associations used by the fitted models, not causal effects or clinical risk factors established by this project.
* Feature importance depends on the model, data preprocessing, correlated predictors, and the particular training sample.
* The historical rankings must be regenerated with the corrected fitted/calibration partition before they are presented as current project findings.

---

## 6. Scientific Discussion: Tabular Deep Learning vs. Regularised Baselines

1. **Occam's Razor on Tabular Benchmarks:** The finding that small MLPs underperform regularised linear models ($R^2 = 0.21-0.27$ vs $0.45$) aligns with established literature in tabular machine learning (e.g., Grinsztajn et al., *NeurIPS 2022*). On $N=442$ samples with continuous variables, high-capacity neural networks tend to overfit the training manifold despite early stopping and dropout.
2. **Significance for Medical AI Research:** In academic medical machine learning (such as at ETH Zurich's Medical Data Science group), transparently recognizing when linear or ensemble models outperform neural networks is an essential mark of scientific maturity. Deep learning excels when dealing with high-dimensional multimodal data (imaging, raw signals, free text), whereas tabular biomarkers demand careful regularization, uncertainty calibration, and explainability.
