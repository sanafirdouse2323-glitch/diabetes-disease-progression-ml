# Comprehensive Experimental Results & Clinical ML Audit

This document records the empirical results, uncertainty calibration, demographic subgroup audits, and biomarker attribution findings across conventional and neural machine learning models.

---

## 1. Experimental Protocol & Partitioning

* **Benchmark Dataset:** `sklearn.datasets.load_diabetes`
  * Sample size: $N = 442$ observations.
  * Input features: $d = 10$ preprocessed physiological baseline variables (`age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`).
  * Continuous target: Quantitative disease progression measured one year post-baseline.
* **Outer Split:** 80% training ($N = 353$) and 20% strictly isolated test set ($N = 89$).
* **Inner Split (Uncertainty & Early Stopping):** 
  * Neural & Conformal calibration split: 80% inner training ($N = 282$) and 20% inner calibration/validation ($N = 71$).
  * Scaler statistics ($\mu, \sigma$) computed strictly on the $282$ inner training samples.
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

### Training Cross-Validation Performance (Training Partition Only)

5-fold cross-validation conducted strictly inside the 353 training observations:

| Model | CV MAE (Mean ± SD) | CV RMSE (Mean ± SD) | CV R² (Mean ± SD) |
|---|:---:|:---:|:---:|
| **Ridge** | **44.999 ± 2.021** | **55.031 ± 2.450** | **0.501 ± 0.043** |
| **Lasso** | 45.040 ± 1.989 | 55.082 ± 2.421 | 0.500 ± 0.044 |
| **Linear Regression** | 45.048 ± 1.982 | 55.094 ± 2.418 | 0.500 ± 0.044 |
| **Random Forest** | 47.665 ± 2.693 | 58.214 ± 3.120 | 0.442 ± 0.061 |
| **Mean Baseline** | 66.867 ± 6.297 | 77.420 ± 6.812 | -0.015 ± 0.012 |

---

## 3. Clinical Uncertainty Quantification (Split Conformal Prediction)

In medical decision systems, single-point estimates are inadequate. We evaluate distribution-free **Split Conformal Prediction** targeting a nominal **90% confidence level** ($\alpha = 0.10$).

| Model | Nominal Coverage | Empirical Test Coverage | Mean Interval Width (MPIW) | Conformal Margin ($\hat{q}$) | Winkler Score |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ridge** | 90.0% | **92.1%** | **129.42** | 64.71 | **162.34** |
| **Linear Regression** | 90.0% | 91.0% | 131.28 | 65.64 | 165.12 |
| **Lasso** | 90.0% | 91.0% | 131.20 | 65.60 | 165.05 |
| **Random Forest** | 90.0% | 89.9% | 134.80 | 67.40 | 171.20 |
| **TensorFlow MLP** | 90.0% | 91.0% | 152.14 | 76.07 | 194.50 |
| **PyTorch MLP** | 90.0% | 89.9% | 158.62 | 79.31 | 206.18 |

### Key Uncertainty Takeaways:
1. **Finite-Sample Validity:** All evaluated models achieve empirical test coverage $\ge 89.9\%$, validating the theoretical coverage theorem $P(Y \in C(X)) \ge 1 - \alpha$.
2. **Interval Efficiency:** Ridge regression achieves the narrowest prediction interval ($129.42$ points) and lowest Winkler score, demonstrating superior uncertainty efficiency alongside top point-prediction accuracy.

---

## 4. Subgroup Demographic Fairness Audit

Clinical AI must be evaluated for disparate error rates across patient demographics. We audit performance across patient `sex` (binary standardized biomarker) on the held-out test cohort:

| Model | Group 0 MAE ($N=42$) | Group 1 MAE ($N=47$) | Absolute Disparity | Parity Ratio ($\min / \max$) | Permutation $p$-value |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ridge** | 41.83 | 43.67 | 1.84 | 0.958 | 0.824 (Not significant) |
| **Lasso** | 41.79 | 43.70 | 1.91 | 0.956 | 0.792 (Not significant) |
| **Random Forest** | 42.21 | 45.33 | 3.12 | 0.931 | 0.581 (Not significant) |
| **PyTorch MLP** | 48.91 | 53.12 | 4.21 | 0.921 | 0.463 (Not significant) |
| **TensorFlow MLP** | 46.78 | 50.63 | 3.85 | 0.924 | 0.512 (Not significant) |

*Hypothesis Testing Verdict:* Non-parametric permutation tests (1,000 resamples) yielded $p > 0.45$ across all models, failing to reject the null hypothesis of error independence. The slight numerical disparities are consistent with expected sampling variation given $N=89$.

---

## 5. Biomarker Attribution & Interpretability

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

### Clinical Insights:
* **Primary Risk Drivers:** `s5` (serum triglycerides) and `bmi` (body mass index) together account for over **46%** of total predictive weight in linear models and over **54%** of tree-based split decisions.
* **Secondary Biomarkers:** Mean blood pressure (`bp`) is a consistent positive predictor of elevated one-year progression.
* **Demographics:** `age` and `sex` had minimal standalone influence once metabolic and lipid profiles were accounted for.

---

## 6. Scientific Discussion: Tabular Deep Learning vs. Regularised Baselines

1. **Occam's Razor on Tabular Benchmarks:** The finding that small MLPs underperform regularised linear models ($R^2 = 0.21-0.27$ vs $0.45$) aligns with established literature in tabular machine learning (e.g., Grinsztajn et al., *NeurIPS 2022*). On $N=442$ samples with continuous variables, high-capacity neural networks tend to overfit the training manifold despite early stopping and dropout.
2. **Significance for Medical AI Research:** In academic medical machine learning (such as at ETH Zurich's Medical Data Science group), transparently recognizing when linear or ensemble models outperform neural networks is an essential mark of scientific maturity. Deep learning excels when dealing with high-dimensional multimodal data (imaging, raw signals, free text), whereas tabular biomarkers demand careful regularization, uncertainty calibration, and explainability.
