# Research Brief: Diabetes Disease Progression

## Project aim

I built this project to compare machine-learning methods for estimating a measure of diabetes disease progression. I wanted to see whether more complex models improve on simpler regression methods and how carefully the results can be evaluated.

This is a regression study. It does not diagnose diabetes or predict whether a person has the condition.

## Dataset

The project uses scikit-learn's public `load_diabetes` dataset:

- 442 observations.
- 10 preprocessed baseline features.
- A continuous disease-progression measure recorded one year after baseline.

The dataset is small and is not raw hospital data. The interactive demo uses synthetic standardised values.

## Models

I compare:

- Linear Regression, Ridge, Lasso and ElasticNet.
- Support Vector Regression.
- Random Forest, Extra Trees and Histogram Gradient Boosting.
- Small neural networks built with PyTorch and TensorFlow/Keras.

The purpose is to compare these approaches, not to assume that a neural network or a more complex model will perform best.

## How I evaluated them

I used held-out test data, repeated train/test splits and nested cross-validation. Where needed, fitting, validation, conformal calibration and final testing were kept separate. Scaling was fitted within the training folds in the nested benchmark to reduce data leakage.

I also used split-conformal prediction to form prediction intervals, compared errors across the two encoded values of the standardised `sex` feature, and examined model coefficients and feature importance.

These extra analyses have limits: conformal coverage is marginal and relies on exchangeability; the subgroup check is not a full fairness assessment; and feature importance does not show cause and effect.

## Main findings

- Lasso had the lowest MAE (42.703) in the single held-out test run.
- In the ten-seed matched-sample experiment, Linear Regression and Lasso had mean MAE values of 42.899 and 42.927. The default PyTorch and TensorFlow networks had higher mean MAE values.
- Validation-based tuning improved the PyTorch network compared with its default control, but did not establish that it was better than the simpler models.
- The nested model-family selection procedure did not improve on the fixed Linear Regression strategy in that experiment.

The experiments use different split designs. Their scores should not be combined into one direct ranking. The full metrics and experiment records are in [RESULTS.md](../RESULTS.md).

## What the project demonstrates

This work gave me practice in preparing data, training and comparing models, setting up evaluation, checking for data leakage, estimating prediction uncertainty, examining model behaviour and recording experiments so they can be repeated.

## Limitations

The dataset is small and preprocessed. There is no independent clinical validation, no raw medical imaging, no longitudinal health-record data and no evidence of clinical benefit. The demo is for demonstration only and must not be used for patient care.

A next step would be to define a clinically meaningful task and evaluate it on a larger, properly governed dataset with patient-level splitting and independent validation.
