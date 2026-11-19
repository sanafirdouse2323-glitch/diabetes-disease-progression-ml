# Technical Audit — 16 November 2026

## Scope and audit boundary

**Repository:** `sanafirdouse2323-glitch/diabetes-disease-progression-ml`  
**Audited baseline:** `main` at commit `12cb45c72d1d19b5489dcad0a1add2d66600c1e7` (15 November 2026).  
**Review type:** source-level review of the primary data, model, evaluation, experiment, demo, test, dependency and CI paths, plus inspection of the recorded GitHub Actions runs for the audited commit.

This is a technical and methodological review, not an independent clinical validation, a formal security audit, or a fresh local execution of every experiment. Workflow results below are GitHub-hosted results. No clinical deployment or patient-care use is implied.

## Executive assessment

The repository is a substantial educational/research portfolio project. It has clear separation between its one-year progression regression benchmark and its separate hospital readmission classification study. It implements classical and neural models, repeated and nested evaluation, split-conformal intervals, exploratory subgroup error analysis, notebooks, tests, locked dependencies and a containerised Streamlit demonstration.

The most important unresolved methodological issue is **dataset-wide feature standardisation before data splitting** in the primary progression study. It is disclosed in the README and results, and it is not target-label leakage. However, it allows information about the full feature distribution to enter every later partition. The current results therefore should not be described as a fully fold-isolated raw-feature evaluation.

**Overall:** sound research-engineering foundations and useful experimental breadth; not a mechanistic diabetes model, not external validation of a clinical prediction model, and not ready for clinical use. Resolve the preprocessing issue and rerun all affected primary-study experiments before making stronger claims.

## Findings

### 1. Scientific representation of the progression target

**Status: appropriately described in most current documentation; interpretation must remain narrow.**

The primary task uses scikit-learn's `load_diabetes` benchmark: 442 observations, 10 baseline features, and a continuous quantitative measure of disease progression one year after baseline. The model estimates this benchmark target from baseline variables.

This is a predictive regression task. It does not model a biological disease mechanism, repeated within-person measurements, a time-varying trajectory, beta-cell function, insulin resistance dynamics, or causal pathways. It also does not classify diabetes diagnosis or prescribe treatment.

Recommended wording:

> Machine learning for estimation of a continuous one-year diabetes disease-progression benchmark measure from baseline features.

Avoid wording such as “mechanistic progression model”, “disease trajectory model”, “diagnosis model”, “clinical risk tool” or “clinically validated system”.

### 2. Preprocessing and leakage

**Priority: high. Confirmed issue.**

File: `src/diabetes_ml/data.py`

The loader calls:

```python
load_diabetes(as_frame=True, scaled=True)
```

With `scaled=True`, scikit-learn returns features standardised using statistics calculated across the complete 442-observation feature matrix. The repository only creates its outer train/test split after this call. The same preprocessed matrix is then used by the repeated-split and nested-cross-validation experiments.

This is **unsupervised full-dataset preprocessing**, not direct target-label leakage. Nevertheless, the feature distribution of future test/outer-fold observations influences preprocessing before the split. Additional `StandardScaler` instances fitted inside later training folds do not undo that initial transformation.

The neural preprocessing code correctly fits its additional scaler on the neural model-fitting subset and transforms validation, calibration and test rows with that fitted scaler. This is good partition discipline, but it still receives the already globally standardised feature representation.

**Required correction:**

1. Load the raw benchmark representation with `scaled=False`.
2. Put scale-sensitive transformations inside scikit-learn pipelines so that each cross-validation training fold learns its own transformation.
3. Fit the neural scaler only on each neural fitting subset, as the existing neural preparation function already does.
4. Redesign the Streamlit demo inputs and training/reference transformations to use the same fold-local or final-training-set preprocessing contract.
5. Recompute feature attribution after the transformation change.
6. Rerun the primary held-out, repeated, matched-sample, nested, neural and conformal experiments; update every affected result table and metadata record.

Do not simply change `scaled=True` to `scaled=False` while keeping old metrics. Keep the published results as a clearly labelled legacy/pre-standardised benchmark if they are retained for comparison.

### 3. Model implementation and comparisons

**Status: model families are present; comparison claims need the experiment design attached.**

The repository contains a mean baseline, Linear Regression, Ridge, Lasso and Random Forest in the main classical experiment. The nested benchmark additionally covers ElasticNet, RBF-SVR, Extra Trees and Histogram Gradient Boosting. PyTorch and TensorFlow/Keras MLP implementations are also present.

The model families are reasonable for a small tabular regression benchmark. The neural architecture is a small two-hidden-layer MLP; it should be presented as a baseline neural model, not a novel architecture.

The first single-split classical-versus-neural experiment uses different model-fitting sample sizes: the classical pathway fits on the post-calibration development subset, while the neural pathway reserves a further early-stopping validation subset. A later ten-seed matched-sample experiment reserves a corresponding unused validation-sized subset for the classical models. The matched-sample experiment is the more appropriate comparison when discussing relative performance.

Keep results from different split designs separate. A lower MAE in one experiment is descriptive and does not establish general superiority.

### 4. Cross-validation and split roles

**Status: core boundaries are correctly structured in the reviewed paths.**

- The primary classical experiment makes a held-out outer test partition and separates calibration rows from the outer training partition.
- Classical CV is performed within the model-fitting partition. The conformal wrapper then fits the estimator on that fitting partition and calibrates on the separate calibration partition.
- Neural preparation separates model fitting, early-stopping validation, conformal calibration and outer testing. Its additional feature scaler is fitted on model-fitting rows only.
- The nested benchmark uses inner-fold MAE for hyperparameter selection and family selection, with each outer fold evaluated after selection.
- Repeated random hold-outs and repeated outer CV reuse observations across different test splits. Their means and standard deviations are descriptive split-sensitivity summaries, not independent replications or confidence intervals.
- The readmission classifier uses patient-grouped outer splitting, and its threshold is selected from grouped out-of-fold predictions generated within the outer training partition.

No direct target-label leakage was identified in the reviewed split/tuning logic. The global feature standardisation issue above remains the principal leakage-related correction for the progression benchmark.

### 5. Metrics and uncertainty

**Status: appropriate for exploratory benchmark evaluation, with limits.**

The progression study reports MAE, RMSE and R², plus conformal coverage, interval width and Winkler score. These provide complementary views of point prediction and interval behaviour.

The split-conformal implementation uses absolute residual scores and the finite-sample order-statistic rank:

```text
k = ceil((n_cal + 1) * (1 - alpha))
```

When the required rank exceeds the calibration sample size, the implementation uses an infinite quantile, giving an unbounded interval rather than an invalid finite threshold. The implementation's interpretation as **marginal** coverage under exchangeability is correct. It is not a patient-specific probability statement and does not guarantee conditional coverage for every subgroup.

Observed test coverage on one small split can be below the nominal level without contradicting the marginal guarantee. Always report nominal coverage, empirical coverage and interval width together, and retain the exchangeability qualification.

For readmission, ROC-AUC, average precision, Brier score, ECE and threshold-based precision/recall/specificity/F2 are relevant descriptive metrics. The reported F2 operating point gives high sensitivity but low precision; that trade-off does not demonstrate clinical utility or justify a clinical threshold.

### 6. Fairness and explainability

**Status: useful exploratory diagnostics, not a complete fairness or explanation framework.**

The progression benchmark subgroup check compares error across the two encoded values of its standardised `sex` feature. These values are not literal demographic labels. The sample is small, only one attribute is examined, and this does not establish fairness or absence of bias.

The permutation test evaluates a difference in mean absolute error under exchangeability of errors across group labels. The results are exploratory; p-values are reported across multiple models without a stated multiplicity adjustment. Do not interpret non-significance as evidence of equality.

Explainability code provides fitted linear coefficients, Random Forest impurity feature importance, additive baseline-relative contributions for linear models, and one-feature-at-a-time perturbations for non-linear models. The latter can overlap and need not sum to the total prediction difference. These are not SHAP values, causal effects or validated clinical explanations.

### 7. Separate UCI readmission study

**Status: appropriately separated and cautious in the current documentation.**

The UCI study uses a different historical dataset (1999–2008), different observation unit (hospital encounters), different endpoint (recorded readmission within 30 days) and a different prediction time (discharge). It is not external validation of the one-year progression regression.

Positive points in the reviewed implementation:
- patient identifiers are used for grouping, not as predictors;
- encounter identifiers are checked for uniqueness and excluded from predictors;
- all encounters for a patient are kept within one partition;
- numeric/categorical preprocessing is fitted within training pipelines;
- the F2 threshold is selected using grouped out-of-fold predictions from the outer training data;
- raw patient-level rows and identifiers are not written to result artifacts.

The published UCI schema available to this experiment does not expose a facility identifier for a hospital-held-out split. Patient grouping therefore does not establish hospital/site generalisation. The historical cohort and source limitations also prevent claims of contemporary clinical performance.

### 8. Tests, CI and reproducibility

**Status: good engineering foundation; reproducibility has defined boundaries.**

The repository has tests for data partition integrity, model instantiation, neural data separation/scaler fitting, conformal quantile behaviour, subgroup metric input contracts, nested benchmarking, readmission patient grouping and notebook structure/execution.

GitHub Actions for the audited main commit `12cb45c72d1d19b5489dcad0a1add2d66600c1e7` showed success for:
- core test matrix on Python 3.10, 3.11, 3.12 and 3.13;
- neural framework checks;
- notebook execution;
- Docker build and application smoke test.

Separate successful recorded runs were also found for repeated classical validation, repeated neural validation and the external UCI readmission experiment, but those workflows ran on earlier commits. The nested benchmark workflow is manually dispatched or runs on eligible pull-request changes; it is not part of every ordinary push.

The `uv.lock` file pins the Python dependency resolution and hashes. This does not make the full container bit-for-bit reproducible: the Docker base image is tag-based rather than pinned to an immutable image digest, and operating-system packages are not pinned to exact package versions. The README correctly notes that interpreter, operating system and external UCI service remain runtime inputs.

This review inspected repository code and GitHub-hosted workflow results; it did not execute the entire suite or long-running experiments locally.

## Prioritised remediation plan

### P0 — required before claiming fully leakage-controlled progression evaluation

- Switch to the raw feature representation (`scaled=False`).
- Put every learned transformation inside the fitting pipeline/split boundary.
- Ensure nested CV scales within each inner training fold.
- Preserve the existing training-only scaler contract for neural networks.
- Update demo transformations and synthetic input semantics.
- Rerun and publish all affected results with new commit/run provenance.
- Add regression tests that fail if full-dataset feature scaling reappears.

### P1 — strengthen scientific evaluation

- Make the matched-sample classical-versus-neural experiment the primary cross-family comparison.
- Report per-seed results and paired differences; do not present repeated-split SD as confidence intervals.
- Add sensitivity to alternative split seeds and, when scientifically justified, bootstrap/permutation uncertainty for performance differences.
- Explain target scale and the benchmark's clinical meaning/limitations rather than treating MAE as a clinical error tolerance.
- Retain conformal coverage and interval width side by side; consider interval-conditional diagnostics without promising subgroup guarantees.

### P2 — strengthen responsible-AI analysis

- Describe current sex-feature analysis as exploratory error disparity, not algorithmic fairness validation.
- If reporting inferential tests across multiple models or groups, pre-specify hypotheses and address multiplicity.
- Evaluate additional clinically relevant groups only when a suitable dataset, sample size, definitions and governance are available.
- Keep model attributions explicitly associational and method-specific.

### P3 — external relevance and engineering hardening

- Seek an independently governed cohort with a comparable one-year progression endpoint before calling any result external validation.
- Do not use the UCI readmission study as progression-model validation.
- For deployment/reproducibility claims, pin the Docker base image by digest and pin operating-system packages or document the intended update policy.
- Keep the demo visibly marked as synthetic and non-clinical.

## Portfolio positioning

A defensible description is:

> An independent medical machine-learning portfolio project comparing classical and neural regression methods on a public one-year diabetes progression benchmark, with nested and repeated evaluation, split-conformal uncertainty estimation, exploratory subgroup error analysis, reproducible experiment tooling and a separate patient-grouped hospital readmission study.

Best demonstrated capabilities: Python ML engineering, experimental design, model comparison, uncertainty estimation, careful interpretation, automated testing and reproducible research workflows.

The project does **not** currently demonstrate mechanistic disease modelling, longitudinal patient trajectory modelling, multimodal medical representation learning, clinical validation, prospective utility or novel ML methodology.

## References

- scikit-learn, `load_diabetes` API: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
- scikit-learn, preprocessing and data leakage guidance: https://scikit-learn.org/stable/common_pitfalls.html
- UCI Diabetes 130-US Hospitals dataset: https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008
- Lei, J. et al. (2018), *Distribution-Free Predictive Inference for Regression*: https://doi.org/10.1080/01621459.2017.1307116
