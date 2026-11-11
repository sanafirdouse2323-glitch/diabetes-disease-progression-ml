# Independent ML Research Demonstration

## Project

**Diabetes disease-progression regression with uncertainty estimation and leakage-aware evaluation**

Repository: https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml

This is an independent educational research-software project. Its purpose is to demonstrate a complete and reproducible machine-learning workflow on a small public tabular benchmark—not to provide a clinical prediction system.

## Research question

On a small, preprocessed tabular disease-progression benchmark, how do linear, kernel, tree-ensemble, and compact neural models compare when model selection is separated from final evaluation? What additional information do split-conformal prediction intervals, subgroup error summaries, and feature-attribution methods provide—and what are their limits?

The project deliberately tests whether added model complexity produces a meaningful benefit rather than assuming that a neural or ensemble model should outperform simpler methods.

## Data and prediction target

- Dataset: `sklearn.datasets.load_diabetes`.
- Size: 442 observations and 10 preprocessed baseline features.
- Target: a continuous quantitative measure of disease progression one year after baseline.
- The task is **regression**, not diabetes diagnosis or classification.

The repository does not contain raw medical images, longitudinal electronic health records, genomic data, or a new patient cohort. The model inputs in the interactive demo are synthetic standardized feature values, not measurements from a real person.

## Methodology

### 1. Controlled fitting, validation, calibration, and testing

The repeated-seed neural comparison uses disjoint partitions per seed: 225 model-fitting observations, 57 validation observations, 71 conformal-calibration observations, and 89 test observations. The validation set is used for neural early stopping and configuration selection; the calibration set is not used to fit or select the model; and the test set is reserved for final evaluation.

### 2. Model comparison and hyperparameter selection

The project compares ordinary and regularised linear models, tree ensembles, SVR-RBF, and compact PyTorch/TensorFlow MLPs. A controlled PyTorch study selects from a small, predeclared set of settings using validation data, then compares the selected model and its default control on the same untouched test observations.

The expanded nested benchmark uses five outer folds repeated three times and three inner folds. It evaluates a predeclared set of model families and includes a mean-prediction baseline. Scaling is placed inside a scikit-learn pipeline so it is learned only from the relevant training fold.

Crucially, the benchmark now has two distinct outputs:

- **Per-family comparison:** each model family is tuned within the inner folds and evaluated on the corresponding outer fold. This compares the predeclared family-specific tuning procedures.
- **Inner-fold family selector:** within each outer training partition, the family with the lowest inner-CV MAE is selected; its refitted estimator is then evaluated on the untouched outer fold. This evaluates the process of choosing among the candidate families, rather than selecting a winner by inspecting outer-test results.

### 3. Split-conformal prediction

The project constructs regression intervals from absolute calibration residuals using the finite-sample conformal rank. The stated guarantee is marginal coverage under exchangeability. It is not an individual patient's probability of being inside an interval and does not imply reliable coverage after distribution shift.

### 4. Exploratory subgroup error analysis

The project compares errors across the two encoded values of the dataset's standardized `sex` feature. This is a narrow, small-sample error audit—not a comprehensive fairness assessment, proof of fairness, or evidence that disparities are absent.

### 5. Model interpretation

Linear-model coefficients and reference-relative contributions, together with tree-model feature importance and one-feature perturbation explanations, are used to inspect fitted model behaviour. These are model-dependent associations, not causal effects or clinically validated explanations. The perturbation values for the tree model are not SHAP values and are not guaranteed to sum to the prediction difference.

## Selected empirical evidence

The matched-sample ten-seed comparison gives a useful example of why model choice should be evidence-led. With 225 fitting observations per seed, the classical models had mean test MAE of 42.899 for Linear Regression, 42.927 for Lasso, 44.858 for Random Forest, and 49.015 for Ridge. The default PyTorch and TensorFlow MLPs had mean test MAE of 53.964 and 57.204, respectively.

A separate, validation-selected PyTorch tuning study reduced the MLP's mean test MAE to 44.934, bringing it much closer to the classical models but not establishing superiority over Linear Regression or Lasso. The ten repeated hold-out test partitions overlap; their standard deviations are descriptive split-sensitivity summaries, not independent-sample confidence intervals.

The nested repeated-CV run reported the following mean outer-fold MAE values (15 outer evaluations per family):

| Model family | Mean outer-fold MAE |
|---|---:|
| Mean Baseline | 65.896 |
| Linear Regression | 44.483 |
| Ridge | 44.565 |
| ElasticNet | 44.548 |
| SVR-RBF | 44.414 |
| Random Forest | 46.437 |
| Extra Trees | 45.584 |
| Histogram Gradient Boosting | 46.552 |

The difference between SVR-RBF and Linear Regression is only 0.069 target units. This small numerical difference is not evidence of a reliable winner. The per-family ranking is descriptive; no formal significance test is reported. Each outer split is evaluated once, but an observation can appear in outer-test partitions in more than one repeat.

The experiment also evaluates a selection procedure that chooses the family using inner-fold MAE inside each outer training partition, then evaluates that selected family on the untouched outer fold. Its mean outer-fold MAE was 44.854 ± 3.105, RMSE 55.565 ± 3.677, and R² 0.469 ± 0.079. It selected SVR-RBF in 10/15 folds, Ridge in 2/15, and Extra Trees, ElasticNet, and Linear Regression once each. It did not improve over the fixed Linear Regression per-family strategy (mean MAE 44.483). This is an important negative/neutral result: family selection added complexity without demonstrating a benefit on this benchmark.

The nested run's standard deviations are descriptive split-sensitivity summaries, not independent-sample confidence intervals. Its results are not directly interchangeable with the 225-row matched-sample results: the nested benchmark uses larger outer-training partitions and a different validation protocol. Full metrics, fold-level results, selection frequencies, and metadata are recorded in [RESULTS.md](../RESULTS.md#11-nested-repeated-cv-comparison-and-inner-fold-family-selection).

## What this project demonstrates

The work is useful evidence of practical research skills in:

- formulating a measurable regression question and choosing appropriate metrics;
- separating training, validation, calibration, and test data;
- controlling preprocessing leakage through fold-local pipelines;
- comparing simple and complex model families rather than assuming model complexity is beneficial;
- implementing validation-based model selection and nested evaluation;
- applying split-conformal uncertainty estimation with its assumptions stated;
- conducting limited subgroup error analysis and explaining why it is not a fairness certificate;
- examining model behaviour without presenting association as causation;
- building a Python package, automated tests, reproducible notebooks, CI workflows, a Streamlit demonstration, and a Docker image.

## How to demonstrate the application

1. Read the repository README and the detailed [experimental results](../RESULTS.md).
2. Install the demo extra and run `streamlit run demo/app.py`.
3. Use the synthetic standardized feature controls. Open the prediction tab to see a benchmark-scale point prediction and split-conformal interval.
4. Open local feature attribution to inspect how the selected model changes the synthetic profile's prediction relative to the model-fitting reference.
5. Open the subgroup audit to see the held-out benchmark error comparison and its limitations.
6. For a reproducible methods discussion, run `python -m diabetes_ml.nested_tabular_benchmark --output nested-tabular-benchmark` and inspect the per-family and inner-selected-family CSV outputs.

The demo's profile sliders do not accept real patient measurements and should not be used to guide diagnosis, treatment, or care.

## Limitations and next research steps

- The benchmark is small and preprocessed, with 442 observations and 10 input features.
- No external cohort or prospective validation is included.
- The project does not implement multimodal fusion, longitudinal modelling, raw-image analysis, or electronic-health-record integration.
- The conformal statement depends on exchangeability; it does not guarantee subgroup-conditional coverage or robustness to distribution shift.
- The subgroup audit is limited to one encoded feature and does not establish fairness.
- The model searches are intentionally compact, and no inferential claim is made from the descriptive repeated-fold standard deviations.
- Predictive performance on this benchmark does not establish clinical validity, patient benefit, or readiness for deployment.

A natural next step for a medical machine-learning PhD would be to transfer the same evaluation discipline to a suitably governed, clinically meaningful dataset with clearly defined prediction time, outcomes, patient-level split strategy, external validation, subgroup analysis, and—where relevant—multimodal or longitudinal inputs.

## Suggested explanation in an interview

> I developed an independent machine-learning project to examine disease-progression regression on a small public benchmark. I compared classical models with compact neural networks, separated fitting, validation, conformal calibration and test data, and then extended the comparison to nested repeated cross-validation. One important result was that more complex models did not automatically outperform simpler regressors. I also implemented split-conformal intervals, a limited subgroup error audit, and model-attribution views, while documenting their assumptions and limitations. The project is not a clinical tool; it demonstrates my approach to experimental design, leakage control, model evaluation, uncertainty estimation, interpretation, and reproducible software. I would like to extend these methods to larger, externally validated medical datasets and, where appropriate, multimodal or longitudinal learning.
