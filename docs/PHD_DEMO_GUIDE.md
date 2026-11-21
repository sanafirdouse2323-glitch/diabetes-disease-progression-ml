# PhD demonstration guide

## How to introduce the project

"I built this project to compare machine-learning methods on the public scikit-learn diabetes disease-progression dataset. I wanted to see whether neural networks or more complex models would improve on simpler regression methods, and I also wanted to evaluate the models carefully rather than rely on a single train/test split."

The dataset contains 442 observations and 10 baseline features. Its target is a continuous measure of disease progression one year after baseline. It is not a diabetes diagnosis label. The demo uses synthetic examples in the dataset's unstandardized source feature representation, not real patient measurements.

## A suggested 8–10 minute walkthrough

### 1. Explain the question

Start with the research question: do more complex methods improve prediction on this small benchmark, and how consistent are the results across different data splits?

### 2. Introduce the data

Show the README and the first notebook. Explain that the dataset is public, contains 442 observations and is designed for regression. Make clear that the target is a one-year progression measure, not a diabetes/no-diabetes label.

### 3. Describe the models

I compared Linear Regression, Ridge, Lasso, ElasticNet, Support Vector Regression, Random Forest, Extra Trees and Histogram Gradient Boosting. I also trained small neural networks with PyTorch and TensorFlow/Keras.

### 4. Explain the evaluation

Show how the code separates model fitting, validation, conformal calibration and final testing. Explain that the classical linear models learn their scaling only from the data used to fit them, and that each cross-validation fold learns its own scaler. The neural models use the same principle for their model-fitting subset.

I used repeated hold-out experiments as well as nested cross-validation. The nested experiment tunes each model family inside the outer training partition and evaluates it on the corresponding held-out fold. The separate family-selection results are reported apart from the per-family comparison.

### 5. Discuss the findings

"In the corrected ten-seed matched-sample experiment, Ridge, Lasso and Linear Regression had mean test MAE values of about 42.8 to 42.9. The fixed PyTorch and TensorFlow baselines had higher mean errors under the settings I tested. A separate validation-selected PyTorch model improved on its default control, but it still did not outperform those three linear models on mean MAE."

"The nested cross-validation experiment gave a similar overall picture. SVR-RBF and Linear Regression had very close mean MAE values, and the procedure that selected a model family inside the inner folds did not improve on using Linear Regression as a fixed strategy."

The standard deviations are descriptive measures across repeated splits, not confidence intervals. The repeated test partitions overlap, and the benchmark is not clinical validation.

### 6. Show the demo and code

Run the Streamlit demo with the instructions in the README. Use the synthetic feature profiles, explain what the prediction interval represents, and point out that the demo is for research demonstration only. Then show the package structure, tests and GitHub Actions workflows.

## A short interview explanation

"I built a machine-learning project using scikit-learn's public diabetes disease-progression dataset. I compared classical regression methods with small PyTorch and TensorFlow neural networks. I also worked on preprocessing, repeated and nested evaluation, conformal prediction intervals and feature analysis. In the corrected repeated-split experiments, the simpler linear models performed strongly. Tuning improved the PyTorch baseline, but it did not outperform the three linear estimators on mean MAE. I treated the work as a research benchmark, not a clinical tool."

## Questions you may be asked

**Why did you choose this dataset?**

It is a public dataset that let me compare regression methods without needing access to private hospital records. It is small, so I could focus on the experiment design and evaluation process.

**Does the model diagnose diabetes?**

No. It predicts a continuous disease-progression measure from the dataset. It does not classify people as having or not having diabetes.

**Why did you include neural networks?**

I wanted to test whether they improved the predictions rather than assume that a more complex model would perform better. The fixed neural baselines had higher mean errors than the classical models. A separate validation-selected PyTorch model improved on its paired control, but it remained behind Ridge, Lasso and Linear Regression on mean MAE.

**What did the nested cross-validation add?**

It allowed me to tune each model family inside the training folds and evaluate that process on held-out outer folds. I also evaluated the full procedure that selects a family using inner-fold results. In this experiment, that selection procedure did not improve on using Linear Regression as a fixed strategy.

**Can the demo be used in a hospital?**

No. The dataset is small, the demos use synthetic inputs, and the models have not been independently or clinically validated. They are not designed for diagnosis or treatment decisions.

## Points to keep clear

The target is a progression measure, not a diabetes diagnosis. A 90% split-conformal interval is a marginal coverage statement under its assumptions, not a 90% probability about one person's outcome. The subgroup comparison does not prove fairness, and the separate readmission classification study is not external validation of the progression model.
