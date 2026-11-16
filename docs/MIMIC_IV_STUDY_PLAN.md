# MIMIC-IV longitudinal glycaemic-control study: feasibility protocol

## Status

**Feasibility protocol and SQL scaffold only. No MIMIC-IV patient data has been downloaded, queried, or analysed.** Execution is blocked until the researcher has their own approved PhysioNet credentialing, completes the required human-subjects training, and signs the MIMIC-IV Data Use Agreement (DUA).

This is an extension of the diabetes disease-progression project using a clinically observed longitudinal measurement. It does not replace or relabel the scikit-learn benchmark's target, and its outcome is not interchangeable with the benchmark's one-year progression score.

## Dataset and access

Use **MIMIC-IV v3.1**, the latest version identified on the official PhysioNet project page when this protocol was prepared. MIMIC-IV is a de-identified relational EHR dataset from Beth Israel Deaconess Medical Center. Its hospital module contains admissions, diagnoses, laboratory measurements, medications, procedures, and related hospital data. The ICU module contains higher-frequency critical-care observations.

- Dataset page: https://physionet.org/content/mimiciv/3.1/
- Official documentation: https://mimic.mit.edu/docs/iv/
- Access instructions: https://mimic.mit.edu/docs/gettingstarted/
- PhysioNet credentialing: https://physionet.org/settings/credentialing/
- Required CITI course information: https://physionet.org/about/citi-course/

**Cost:** MIMIC-IV dataset access is free of charge after approval. PhysioNet states that the required CITI "Data or Specimens Only Research" course is free when registered through the MIT Affiliates option for PhysioNet access. Credentialing and the dataset-specific DUA are mandatory. Optional cloud compute or storage may incur separate charges; local download is available after approval.

Access sequence:
1. Create or sign in to a PhysioNet account.
2. Complete the PhysioNet credentialing application and required CITI "Data or Specimens Only Research" training.
3. Submit the CITI completion report and required affiliation/reference details.
4. Wait for credentialing approval.
5. Open the MIMIC-IV v3.1 project page and read/sign its DUA.
6. Download the required hospital-module CSV files locally, or request approved cloud access.
7. Keep all source files and patient-level derived data outside version control.

Each researcher must obtain their own access. Do not share credentials, raw data, patient-level extracts, or restricted data with collaborators who are not independently authorised.

## Research question

Among adults with a recorded diabetes diagnosis, how does measured HbA1c change over approximately one year, and can baseline clinical information help estimate that subsequent change?

The initial feasibility endpoint is the difference between a baseline HbA1c measurement and one follow-up HbA1c measurement obtained **180–365 days later**, in the same unit. A negative difference means the measured HbA1c decreased; a positive difference means it increased. This is a change in measured glycaemic control, not a direct measurement of all aspects of diabetes progression and not a causal treatment effect.

## Why a feasibility phase is mandatory

MIMIC-IV is primarily a hospital EHR, not a dedicated longitudinal diabetes registry. HbA1c tests may be sparse and are likely to be measured more often in patients with particular clinical needs. Requiring a baseline and a follow-up test will select a subset of patients and can introduce monitoring/selection bias. Before committing to a predictive study, quantify:
- how many distinct HbA1c item IDs and units are present;
- the number of diabetes-coded adults with at least one HbA1c result;
- the number with a same-unit follow-up result 180–365 days later;
- the distribution of follow-up intervals and HbA1c values;
- the fraction of records from inpatient, emergency, and other contexts where those distinctions are available;
- missingness and the number of repeated measurements per patient.

The SQL file first lists candidate HbA1c lab items and units, then produces candidate baseline/follow-up pairs. Item labels and units must be reviewed against the actual v3.1 data dictionary before treating them as HbA1c measurements. No item ID is hard-coded in advance.

## Cohort and outcome construction

The initial query:
- Identifies patients with at least one ICD-9-CM diabetes code beginning `250` or ICD-10-CM code in the E08–E13 families in `diagnoses_icd`.
- Restricts the baseline measurement to adults using MIMIC's patient anchor-age fields. The `anchor_age=300` convention represents age over 89 and must be reported as 90+, not as an exact age.
- Finds the first candidate HbA1c measurement on or after the first recorded diabetes-coded hospital admission.
- Finds the closest same-unit HbA1c measurement in the 180–365-day window after baseline.
- Retains patient and admission identifiers only for linkage, grouping, and audit; never use them as predictors.

The initial output is a candidate-pair table for feasibility review, not a final analysis dataset. Repeated measurements and patients with multiple encounters require explicit handling before modelling.

## Analysis plan

This study will **not** compare simple models with neural networks and will not run a model-family bake-off. First establish that the paired-measurement cohort is large and sufficiently representative for the proposed question. If feasibility is supported, pre-specify one primary statistical/ML pipeline and its covariates before looking at held-out outcomes.

For any subsequent prediction analysis:
- Define the prediction time at the baseline HbA1c measurement.
- Use only information available at or before that baseline time.
- Exclude all follow-up measurements and post-baseline information from predictors.
- Split by patient so no patient appears in more than one partition.
- Fit imputation, scaling, feature selection, and tuning within training data only.
- Keep a final patient-level held-out test set untouched until the analysis is locked.
- Report MAE/RMSE for the continuous HbA1c change, calibration/uncertainty where appropriate, and subgroup results with small-cell caution.
- Record data version, query version, item IDs, units, feature windows, package versions, random seeds, and a reproducible run manifest.

If the paired cohort is too small or biased by selective retesting, do not force a predictive model. Report a transparent feasibility analysis and explain why MIMIC-IV cannot support the intended progression endpoint adequately.

## Limitations to state

- MIMIC-IV is a single-centre retrospective EHR dataset, not a representative diabetes cohort.
- HbA1c measurements are intermittent and driven by clinical care; absence of a test is not a normal result.
- Requiring repeat HbA1c tests creates selection bias and may over-represent sicker or more closely monitored patients.
- A change in HbA1c is a surrogate measure of glycaemic control, not a complete measure of disease progression, complications, or treatment benefit.
- Diabetes diagnosis codes may be incomplete or recorded after the disease was already present.
- De-identification shifts dates while preserving within-patient intervals; do not interpret dates as real calendar dates or combine them with external timelines.
- Retrospective prediction performance does not establish prospective clinical benefit or clinical utility.

## Data governance

Never commit MIMIC-IV CSVs, SQL query outputs containing patient-level rows, database dumps, or patient-level predictions to this public repository. The repository's `.gitignore` excludes `data/`; store authorised data under that directory. Do not publish derived patient-level data or trained models without checking the current PhysioNet DUA and dataset-specific sharing rules. Only publish approved aggregate results and code.

## Current blocker and next execution step

The researcher must complete the PhysioNet access steps above. Once access is approved and the MIMIC-IV v3.1 hospital CSV files are available locally, run the discovery query and candidate-pair query against an authorised local PostgreSQL import. Review HbA1c item labels/units and the resulting aggregate feasibility counts before implementing any model. No cohort size, model result, or clinical conclusion is asserted in this repository until that execution has occurred.
