# Research brief: diabetes disease progression

## Why I built this project

I wanted to compare a range of machine-learning methods on a public diabetes disease-progression dataset. My main question was whether more complex models would improve on simpler regression methods, and how much the results changed when I evaluated the models across different data splits.

This is a regression study. It estimates a continuous benchmark target; it does not diagnose diabetes or predict whether someone has the condition.

## Dataset

I use scikit-learn's `load_diabetes` dataset. It contains 442 observations, 10 baseline features and a continuous measure of disease progression one year after baseline. In the current implementation, I load the unstandardized source feature representation with `scaled=False`. The dataset is small and prepared for research use; it is not a set of raw hospital records.

The demonstrations use synthetic feature profiles drawn from the benchmark's source representation. They are examples for exploring the model interface, not measurements from real people.

## Models and experiments

I compared Linear Regression, Ridge, Lasso, ElasticNet, Support Vector Regression, Random Forest, Extra Trees and Histogram Gradient Boosting. I also trained small multilayer perceptrons with PyTorch and TensorFlow/Keras.

I evaluated the models with held-out test data, repeated train/test splits and nested cross-validation. For scale-sensitive classical models, the `StandardScaler` is part of the estimator pipeline. It is fitted only on the model-fitting data and is re-fitted separately inside each cross-validation training fold. The neural workflow follows the same principle: it learns its scaler from the neural model-fitting subset and applies it unchanged to the validation, calibration and test subsets.

For prediction uncertainty, I implemented split-conformal intervals and kept the calibration data separate from model fitting. I also compared errors across the dataset's two coded values of the `sex` feature and examined model coefficients and feature importance. The subgroup comparison is exploratory, and the feature summaries do not show cause and effect.

## What the corrected results show

In the corrected ten-seed matched-sample comparison, Ridge, Lasso and Linear Regression had mean test MAE values of 42.822, 42.844 and 42.899, respectively. The fixed PyTorch and TensorFlow baselines had higher mean errors under the training settings used in that experiment.

A separate validation-selected PyTorch study improved on its paired default control and reached a mean test MAE of 43.408. It remained behind the three linear estimators in the matched-sample comparison.

I also ran nested cross-validation with five outer folds repeated three times and three inner folds for model tuning. SVR-RBF had the lowest descriptive per-family mean MAE at 44.414, closely followed by Linear Regression at 44.483. The procedure that selected a model family from inner-fold results had a mean outer-fold MAE of 44.854, so this experiment did not show a clear benefit from adding a model-selection step.

The exact metrics, split designs and workflow links are in [RESULTS.md](../RESULTS.md). The older tables in that file are kept as historical records from the earlier full-dataset-standardized implementation. They should not be quoted as results from the corrected code.

## What I learned

This project gave me practical experience with data preparation, classical regression, neural-network training, cross-validation, model selection, conformal prediction and model interpretation. It also reinforced the importance of keeping preprocessing, fitting, calibration and testing separate.

## A separate readmission experiment

The repository includes a second, independent project: a 30-day hospital readmission classification study using the UCI *Diabetes 130-US Hospitals for Years 1999–2008* dataset. It has a different cohort, unit of observation and target. It is not external validation of the one-year disease-progression regression experiment. The design and limitations are described in [READMISSION_STUDY_BRIEF.md](READMISSION_STUDY_BRIEF.md).

## Limitations and next steps

The progression dataset contains only 442 observations, and no independent clinical validation has been performed. The subgroup audit is limited to two dataset codes and is not a full fairness assessment. The demos are educational and should not be used for patient care.

A sensible next step would be to define a clinically meaningful research question and evaluate it on a larger, appropriately governed dataset, with patient-level splitting and independent validation. Any use of hospital data would also require the relevant permissions, privacy safeguards and research approvals.
