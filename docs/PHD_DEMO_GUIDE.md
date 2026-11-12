# PhD Application Demo Guide

## Purpose

Use this repository as evidence of independent machine-learning research and research-software engineering. It demonstrates experimental design, leakage-aware evaluation, model comparison, uncertainty estimation, interpretation, testing, and reproducibility.

**Scope:** this is a regression benchmark using scikit-learn's public, preprocessed diabetes disease-progression dataset (442 observations, 10 features). It predicts a continuous progression measure; it does not diagnose diabetes. It is not a clinical decision-support system and has not been externally or prospectively clinically validated.

## Recommended 8–12 minute walkthrough

### 1. Start with the research question (1 minute)

Explain that the project asks whether more complex tabular models offer a meaningful benefit over simpler models, and what uncertainty and error analysis add. Emphasise that model complexity is tested rather than assumed to be better.

### 2. Establish the data and task (1 minute)

Show the README and research brief. State the dataset size, feature count, and continuous regression target. Explain that the demo's feature controls use synthetic standardised values, not real patient measurements.

### 3. Show the evaluation design (2 minutes)

Use the methodology in docs/RESEARCH_BRIEF.md and RESULTS.md:
- Distinguish model fitting, validation, conformal calibration, and final testing.
- Explain that preprocessing is fitted within training folds in the nested benchmark.
- Explain that the nested family selector chooses using inner-fold MAE, then evaluates the selected procedure on an untouched outer fold.
- Note that repeated outer-CV test partitions overlap across repeats; reported standard deviations are descriptive, not independent-sample confidence intervals.

### 4. Discuss the evidence, including neutral results (2 minutes)

Use the measured results in RESULTS.md, not a newly calculated or selectively chosen score.

Key points:
- In the matched-sample ten-seed comparison, Linear Regression and Lasso had mean test MAE of 42.899 and 42.927, compared with 53.964 for the default PyTorch MLP and 57.204 for the default TensorFlow MLP.
- A separate validation-selected PyTorch tuning study reduced mean test MAE to 44.934, but did not establish superiority over the simpler classical models.
- In the nested repeated-CV benchmark, the inner-fold family-selection procedure had mean outer-fold MAE 44.854, compared with 44.483 for the fixed Linear Regression family strategy. The selector therefore did not demonstrate a benefit on this dataset.
- In the target-standardised neural follow-up, the selected model's mean test MAE was 43.408 versus 44.086 for its paired default control under the same training budget. This is a modest descriptive difference, not evidence of neural superiority; Linear Regression and Lasso remained slightly better in the earlier matched-sample comparison.

Do not combine scores from different split protocols as if they were directly comparable. Do not present overlapping-seed or repeated-fold standard deviations as confidence intervals or formal significance tests.

### 5. Demonstrate uncertainty and interpretation (1–2 minutes)

In the Streamlit demo, show the point prediction and split-conformal interval, then the feature-attribution and subgroup-error views. Explain:
- Conformal coverage is marginal and relies on exchangeability; it is not an individual patient's probability of being inside an interval and does not guarantee subgroup-conditional coverage or robustness to distribution shift.
- The subgroup analysis is a limited exploratory audit, not a fairness certificate.
- Feature attributions describe model behaviour; they are not causal effects or clinically validated explanations.

### 6. Show reproducibility and software engineering (1–2 minutes)

Point to the package structure, tests, notebooks, GitHub Actions, and Docker setup. The main-branch commit e47915188b2c802e3a0e94e138470ae0c93f2d7b passed the eight post-merge workflows recorded for that commit: tests, CI, Docker smoke test, neural framework checks, repeated-seed classical validation, repeated-seed neural validation, neural hyperparameter study, and end-to-end notebook execution.

For a live demonstration, follow the current installation and run instructions in the repository README. Do not claim a deployment is clinically usable merely because the Streamlit demo or Docker image runs.

## Suggested interview explanation

> I developed an independent machine-learning project to study disease-progression regression on a small public tabular benchmark. I compared classical regressors with compact neural networks, separated fitting, validation, conformal calibration and final testing, and extended the work to nested repeated cross-validation with inner-fold model-family selection. An important finding was that additional model complexity did not automatically improve performance: simpler linear models remained highly competitive, and the nested selection procedure did not outperform the fixed linear strategy on this dataset. I also implemented split-conformal prediction intervals, a limited subgroup error audit, model-attribution views, automated tests, reproducible notebooks and a Docker-based demo. I treat this as a methodological and engineering demonstration, not a clinical prediction product. My next research step would be to apply these evaluation principles to a larger, appropriately governed medical dataset with patient-level splitting, external validation and clinically meaningful outcomes.

## Questions to prepare for

**Why use a diabetes dataset if this is not a diagnostic model?**  
The dataset provides a compact, public regression benchmark for demonstrating modelling and evaluation methods. The target is disease progression, not diabetes status or diagnosis.

**Why did you compare neural networks if classical models performed well?**  
To test whether added complexity offered measurable benefit rather than assuming it would. The results support a cautious, evidence-led model choice.

**How did you reduce leakage risk?**  
The workflow separates fitting, validation, calibration and test roles. In the nested benchmark, preprocessing and tuning are confined to the relevant training folds; the outer test fold evaluates the complete selection procedure.

**What does the conformal interval mean?**  
Under exchangeability, split conformal prediction targets marginal coverage over future observations. It is not a patient-specific probability and does not guarantee coverage under distribution shift.

**What is missing before medical use?**  
A clinically relevant and sufficiently large cohort, a prespecified prediction task and prediction time, patient-level and temporal validation where appropriate, external validation, subgroup and calibration assessment, clinical review, governance, and prospective evaluation. This repository does not establish clinical utility or deployment readiness.

## Claims to avoid

- “The model diagnoses diabetes.”
- “The model is clinically validated, fair, or ready for hospital deployment.”
- “The neural network outperforms classical models.”
- “The conformal interval is a patient's probability of the outcome.”
- “The subgroup audit proves fairness.”
- “The repeated-split standard deviation is a confidence interval.”

## Files to open during the walkthrough

1. README.md — project overview, installation, demo and experiment commands.
2. docs/RESEARCH_BRIEF.md — research question, methods, results, limitations and next steps.
3. RESULTS.md — detailed experimental metrics, provenance and interpretation.
4. demo/app.py — interactive educational demonstration.
5. tests/ and .github/workflows/ — automated checks and reproducibility infrastructure.
