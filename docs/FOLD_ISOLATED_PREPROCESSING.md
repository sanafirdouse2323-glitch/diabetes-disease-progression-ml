# How preprocessing is handled

## Why I changed it

The original version of this project used scikit-learn's default `load_diabetes` behaviour, which standardised the feature matrix using statistics calculated across all 442 observations before the repository created its train/test and cross-validation splits. This did not use the target labels, but it meant that information from observations later assigned to validation, calibration or test partitions had already influenced the feature scaling.

I changed the loader to `load_diabetes(as_frame=True, scaled=False)`. The dataset still has 442 observations, 10 features and the same one-year disease-progression target. What changes is the feature representation supplied to the experiments and where scaling is learned.

## Where scaling is fitted now

For Linear Regression, Ridge and Lasso, `StandardScaler` is inside the scikit-learn estimator pipeline. When the model is fitted, the scaler learns its statistics from that model-fitting partition. When cross-validation is run, a fresh copy of the pipeline is fitted inside each training fold.

The mean baseline and Random Forest do not need feature scaling, so they are used without a scaler.

The neural workflow first separates model fitting, early-stopping validation, conformal calibration and outer testing. It then fits its scaler only on the neural model-fitting subset and applies that same transformation to the other subsets. It does not refit the scaler on validation, calibration or test data.

The nested benchmark uses a pipeline for every scale-sensitive model. As a result, each scaler is learned from the relevant inner training fold, not from the complete outer training partition or the full dataset.

The Streamlit and Gradio demos use synthetic values in the unstandardized source feature representation. Their slider ranges are based on the model-fitting subset rather than the held-out test data.

## How feature attribution is reported

A linear model fitted after standardisation stores its coefficients in scaled feature units. Before reporting those coefficients, or using them to calculate local contributions, the code converts them back to the units of the pipeline's input features. For a linear model, the local contributions add up to the difference between the profile's prediction and the reference-profile prediction.

The non-linear model explanation uses one-feature-at-a-time perturbations. Those values can overlap and do not necessarily add up to the total prediction change. They are not causal explanations or SHAP values.

## Current results and historical results

The corrected ten-seed classical and neural experiments, the validation-selected PyTorch study and the nested cross-validation benchmark are reported in [RESULTS.md](../RESULTS.md), with links to the successful GitHub Actions runs. The older result tables in that file are retained as a historical record of the previous full-dataset-standardized implementation. They are not results from the current code and should not be compared directly with the corrected results.

## What this change does not establish

This change prevents full-dataset feature statistics from being used before splitting. It does not provide external validation, clinical validation, fairness evidence, transportability to other populations or evidence of clinical benefit. The separate UCI readmission classification experiment is a different task and must not be treated as external validation of the progression model.
