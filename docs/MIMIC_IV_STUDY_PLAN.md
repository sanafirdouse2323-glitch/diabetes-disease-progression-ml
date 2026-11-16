# MIMIC-IV diabetes readmission study: protocol and access

## Status

**Protocol and cohort-query scaffold only. No MIMIC-IV patient data has been downloaded, queried, or analysed yet.** Execution is blocked until the researcher has their own approved PhysioNet credentialing, completes the required training, and signs the MIMIC-IV Data Use Agreement (DUA).

This is a new, separate clinical EHR study within the diabetes project. It does not replace or relabel the scikit-learn one-year progression benchmark, and it is not an external validation of that continuous regression target.

## Dataset and access

Use **MIMIC-IV v3.1**, the latest version identified in the official PhysioNet project page at the time this protocol was prepared. MIMIC-IV is a de-identified relational EHR dataset from Beth Israel Deaconess Medical Center. Its hospital module includes admissions, diagnoses, laboratory measurements, medication records, procedures, and related hospital data. The ICU module contains higher-frequency critical-care observations.

- Dataset page: https://physionet.org/content/mimiciv/3.1/
- Official documentation: https://mimic.mit.edu/docs/iv/
- Access instructions: https://mimic.mit.edu/docs/gettingstarted/
- PhysioNet credentialing: https://physionet.org/settings/credentialing/
- Required CITI course information: https://physionet.org/about/citi-course/

**Cost:** MIMIC-IV dataset access is free of charge after approval. PhysioNet states that the required CITI "Data or Specimens Only Research" course is free when registered through the MIT Affiliates option for PhysioNet access. Credentialing and DUA approval are mandatory. Optional cloud compute or storage can incur separate charges; local CSV download avoids relying on paid cloud querying.

Access sequence:
1. Create or sign in to a PhysioNet account.
2. Complete the PhysioNet credentialing application and required CITI "Data or Specimens Only Research" training.
3. Submit the CITI completion report and any required affiliation/reference details.
4. Wait for credentialing approval.
5. Open the MIMIC-IV v3.1 project page and read/sign its DUA.
6. Download the required hospital-module CSV files locally, or request approved cloud access.
7. Keep all source files and patient-level derived data outside version control.

Each researcher must obtain their own access. Do not share credentials, raw data, patient-level extracts, or restricted data with collaborators who are not independently authorised.

## Research question

Among adult hospital admissions with a documented diabetes diagnosis, can information available by hospital discharge estimate whether the patient has another **MIMIC-recorded inpatient admission within 30 days after discharge**?

This endpoint is intentionally phrased as a subsequent admission recorded in MIMIC-IV. It does not establish that no readmission occurred at another hospital or outside the data capture system.

## Cohort and outcome definition

The initial cohort query in `sql/mimic_iv_diabetes_readmission_cohort.sql`:

- Identifies admissions with at least one ICD-9-CM diabetes code beginning `250` or ICD-10-CM diabetes code in the E08–E13 families.
- Uses the MIMIC patient anchor-age fields to restrict to adults (18 years or older); the age value for people older than 89 is top-coded and must remain labelled as 90+ rather than treated as an exact age.
- Uses all subsequent admissions for the same `subject_id` to define whether a later admission begins after discharge and within 30 days.
- Excludes index admissions ending in in-hospital death from the readmission-risk cohort.
- Keeps `subject_id` and `hadm_id` only for linkage, grouping, and audit; never use them as model predictors.

The unit of analysis is an eligible hospital admission, not a unique person. A person may contribute more than one eligible admission. Therefore, every train/validation/test split must be grouped by `subject_id` so admissions from one person cannot appear in multiple partitions.

## Prediction time and leakage controls

The intended prediction time is **hospital discharge**. Candidate predictors may include admission characteristics, demographics, diagnoses and procedures known by discharge, length of stay, and laboratory summaries measured during the index admission up to discharge.

Do not use:
- the next-admission timestamp or any variable derived from the readmission outcome;
- post-discharge events or measurements;
- patient or admission identifiers as predictors;
- discharge outcome fields that directly encode death for the readmission-risk cohort;
- any data from a later encounter when constructing index-admission features.

Laboratory extraction must be bounded to the index admission interval and use documented item identifiers and units. Do not treat every laboratory item with a similar name as interchangeable. Preserve missingness and report how each feature is derived.

## Analysis plan

This study will **not** be a simple-model-versus-neural-network comparison or a model-family bake-off. Use one pre-specified primary prediction pipeline after inspecting feature availability and missingness. Any model choice and hyperparameters must be fixed using training data only.

Report, at minimum:
- cohort flow counts and exclusion reasons;
- positive outcome prevalence;
- missingness and feature-availability summaries;
- patient-grouped train/validation/test design;
- AUROC and AUPRC, with uncertainty intervals where justified;
- sensitivity, specificity, positive predictive value and negative predictive value at a threshold selected using training/validation data only;
- Brier score and calibration assessment;
- subgroup performance with small-cell suppression/caution and no unsupported fairness claims;
- data version, query version, package versions, random seeds and a reproducible run manifest.

The held-out test partition must not be used for feature selection, imputation fitting, scaling, model tuning, or threshold selection. If a temporal evaluation is feasible, it should be designed without allowing the same patient to cross partitions and should be reported separately from the primary grouped evaluation.

## Limitations to state

- MIMIC-IV is a single-centre retrospective EHR dataset, not a representative sample of all hospitals or populations.
- The outcome captures admissions recorded in this health system, not all readmissions across all providers.
- Diagnosis coding may miss diabetes cases or include cases identified later in the encounter.
- De-identification shifts dates while preserving within-patient intervals; do not interpret dates as real calendar dates or combine them with external timelines.
- Readmission prediction is not diabetes disease-progression prediction, diagnosis, treatment advice, or proof of clinical utility.
- Retrospective model performance does not establish prospective clinical benefit.

## Data governance

Never commit MIMIC-IV CSVs, SQL query outputs containing patient-level rows, database dumps, or patient-level model predictions to this public repository. The repository's `.gitignore` already excludes `data/`; store authorised data under that directory. Do not publish derived patient-level data or trained models without checking the current PhysioNet DUA and dataset-specific sharing rules. Only publish approved aggregate results and code.

## Current blocker and next execution step

The researcher must complete the PhysioNet access steps above. Once access is approved and the MIMIC-IV v3.1 hospital CSV files are available locally, run the cohort query against a local PostgreSQL import (or an approved cloud instance), validate the resulting cohort counts and label construction, and only then implement and execute the feature pipeline. No cohort size, model result, or clinical conclusion is asserted in this repository until that execution has occurred.
