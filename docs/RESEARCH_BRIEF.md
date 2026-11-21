# Research Brief: Diabetes Disease Progression

## Project aim

I built this project to compare machine-learning methods for estimating a measure of diabetes disease progression. I wanted to see whether more complex models improve on simpler regression methods and how carefully the results can be evaluated.

This is a regression study. It does not diagnose diabetes or predict whether a person has the condition.

## Dataset

The project uses scikit-learn's public `load_diabetes` dataset:

- 442 observations.
- 10 unstandardized source features (loaded with `scaled=False`).
- A continuous disease-progression measure recorded one year after baseline.

The dataset is small and is not raw hospital data. The demos use synthetic examples in the unstandardized source feature representation.

## Models

I compare:

- Linear Regression, Ridge, Lasso and ElasticNet.
- Support Vector Regression.
- Random Forest, Extra Trees and Histogram Gradient Boosting.
- Small neural networks built with PyTorch and TensorFlow/Keras.

The purpose is to compare these approaches, not to assume that a neural network or a more complex model will perform best.

## How I evaluated them

I use held-out test data, repeated train/test splits and nested cross-validation. Linear-estimator scaling is fitted only inside the relevant model-fitting pipeline; during cross-validation, each fold fits its own scaler on that fold's training observations. Neural scaling is fitted only on the neural model-fitting partition. Conformal calibration and final testing remain separate.

I also use split-conformal prediction to form prediction intervals, compare errors across the two coded values of the unstandardized `sex` source feature, and examine model coefficients and feature importance.

These extra analyses have limits: conformal coverage is marginal and relies on exchangeability; the subgroup check is not a full fairness assessment; and feature importance does not show cause and effect.

## Earlier findings and current status

The numerical comparisons currently retained in [RESULTS.md](../RESULTS.md) were produced before the fold-isolated preprocessing correction, using scikit-learn's full-dataset standardized feature representation. They are historical findings, not current performance estimates for the corrected implementation. The affected single-split, repeated-seed, neural and nested experiments must be rerun before quoting model rankings or MAE values from this project.

The experiments use different split designs. Their scores should not be combined into one direct ranking. New results will be reported separately with exact workflow and commit provenance.

## What the project demonstrates

This work gave me practice in preparing data, training and comparing models, setting up evaluation, checking for data leakage, estimating prediction uncertainty, examining model behaviour and recording experiments so they can be repeated.

## Separate repository experiment: 30-day readmission

The repository also contains a distinct UCI hospital readmission classification experiment. It predicts readmission within 30 days after discharge from a different historical dataset and uses encounters rather than the 442 baseline observations in this progression study. It is **not external validation** of the one-year disease-progression regression model. Its methods and limitations are documented in [READMISSION_STUDY_BRIEF.md](READMISSION_STUDY_BRIEF.md).

## Limitations

The dataset is small and uses a public benchmark feature representation. There is no independent clinical validation, no raw medical imaging, no longitudinal health-record data and no evidence of clinical benefit. The demo is for demonstration only and must not be used for patient care.

A next step would be to define a clinically meaningful task and evaluate it on a larger, properly governed dataset with patient-level splitting and independent validation.
