# Fold-isolated preprocessing correction

## Purpose

The progression benchmark now loads scikit-learn's public diabetes dataset with
`load_diabetes(as_frame=True, scaled=False)`. This prevents feature means and
standard deviations from being computed using observations that may later be
assigned to validation, calibration, or test partitions.

The dataset still contains 442 observations, 10 source features, and the same
continuous one-year progression target. The change affects feature
representation and the estimation procedure; it does not change the target or
turn the project into a clinical prediction system.

## Preprocessing boundaries

- **Linear Regression, Ridge, and Lasso:** each estimator is a scikit-learn
  `Pipeline` with `StandardScaler` followed by the estimator. A normal model
  fit learns scaling only from the model-fitting partition. During
  cross-validation, scikit-learn fits a fresh pipeline and scaler separately
  inside each training fold.
- **Mean baseline and Random Forest:** no feature scaling is applied.
- **Neural models:** `prepare_neural_datasets` creates disjoint fitting,
  early-stopping validation, conformal calibration, and outer-test partitions.
  Its `StandardScaler` is fitted only on the neural fitting partition and is
  applied unchanged to the other partitions.
- **Nested benchmark:** scale-sensitive estimators place `StandardScaler`
  inside the estimator pipeline used by inner-fold `GridSearchCV`; the scaler
  is learned from each inner training fold.
- **Demos:** synthetic slider values use the unstandardized source feature
  representation. Slider limits come from the model-fitting partition rather
  than from the held-out test data.

## Attribution

For fitted linear pipelines, coefficients from the standardized model space
are converted back to the input feature units before displaying coefficient
summaries or local baseline-relative contributions. Contributions for a
linear model should sum (within floating-point tolerance) to the difference
between the profile prediction and the reference-profile prediction.
Non-linear perturbation explanations retain their existing limitations and are
not causal explanations or SHAP values.

## Historical results

The numerical tables in the current `RESULTS.md` were generated before this
correction, from the full-dataset standardized feature representation. They
remain in the file for historical traceability but are not results of the
current implementation. The single-split, repeated classical, repeated
neural, neural tuning, nested-CV, and demo outputs that depend on this loader
must be regenerated from the corrected branch and published with the exact
commit and workflow run. No updated performance figure should be inferred
from the old tables.

## Scope

This correction addresses an unsupervised feature-preprocessing boundary. It
does not establish independent external validation, clinical validity,
fairness, transportability, or clinical utility. The separate UCI readmission
study and the separate open CGM extension remain distinct tasks and must not be
combined with the one-year progression regression results.
