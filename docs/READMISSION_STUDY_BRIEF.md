# Separate Research Brief: 30-Day Hospital Readmission

## Scope and relationship to the main project

This document describes a separate classification experiment in `src/diabetes_ml/external_readmission.py`. It is **not** an extension of the one-year disease-progression regression experiment using scikit-learn's 442-row `load_diabetes` dataset, and it is **not external validation** of that regression model. The datasets, units of observation, outcomes, and estimands are different.

## Dataset and endpoint

The experiment retrieves UCI dataset 296, *Diabetes 130-US Hospitals for Years 1999-2008* (DOI: [10.24432/C5230J](https://doi.org/10.24432/C5230J)). UCI describes 101,766 inpatient encounters from 130 US hospitals and integrated delivery networks during 1999–2008, representing 71,518 unique patients.

The target is mapped as:

- `<30`: positive, readmission within 30 days.
- `>30` and `NO`: negative for the within-30-day endpoint.

The unit of analysis is an **encounter**, not a unique patient. All encounters belonging to one patient are assigned to the same outer partition, but patients with more encounters contribute more rows to encounter-level metrics.

## Prediction time

The intended prediction time is **at discharge**. The feature table includes discharge disposition and other encounter-level information that is not necessarily known at admission. The model should therefore not be described as an admission-time readmission prediction model.

## Data handling and evaluation design

- `encounter_id` and `patient_nbr` are not predictors. `patient_nbr` is used only as the group key for splitting.
- Race, gender, and age are excluded from the predictive feature matrix in this pre-specified baseline and retained only for exploratory subgroup reporting.
- The three coded categorical integer fields (`admission_type_id`, `discharge_disposition_id`, and `admission_source_id`) are treated as categorical, not continuous measurements.
- Imputation, scaling, and one-hot encoding are fitted inside each model pipeline using only the relevant training partition.
- The outer evaluation uses a five-fold `StratifiedGroupKFold` split and selects one held-out fold per seed (42, 43, and 44 in the workflow configuration). The same patient cannot occur in both outer training and test partitions.
- The fixed reference models are a prior-only dummy classifier, logistic regression, and random forest. This is not an exhaustive model search.
- Test folds from different seeds can overlap. Across-seed standard deviations are descriptive split-sensitivity summaries, not confidence intervals or independent replications.

## Threshold selection and reported metrics

The positive class is relatively uncommon, so a 0.5 probability threshold can produce high accuracy while missing nearly all positive cases. The 0.5 cutoff is retained only as a reference.

For each outer split and model, the operating threshold is selected by maximising F2 (beta = 2) on three-fold, patient-grouped out-of-fold probabilities generated **only within the outer training partition**. A tie is resolved by choosing the largest threshold among tied candidates. The selected threshold is then applied once to the untouched outer test partition. Test labels are not used to select the threshold.

F2 weights recall more heavily than precision. This makes it a stated research choice for examining a recall-oriented operating point; it is not a clinically validated threshold, utility function, or decision rule. The outputs report ROC-AUC, average precision, Brier score, and threshold-based accuracy, precision, sensitivity, specificity, and F2 at both the 0.5 reference cutoff and the training-selected F2 cutoff.

## Privacy and reproducibility

The workflow uploads aggregate metrics and metadata only. It does not write raw patient-level records or patient identifiers to the repository or the workflow artifact. A SHA-256 fingerprint of the retrieved source tables is recorded to help detect changes in the upstream data. The fingerprint is not a substitute for dataset version control or a formal data-provenance audit.

## Limitations

This is a retrospective, historical benchmark from 1999–2008. It does not establish performance in contemporary hospitals, on new sites, or in prospective workflows. Patient-grouped splitting addresses within-patient overlap between training and test partitions; it does not establish hospital-level generalisation. Metrics are encounter-weighted, and repeated encounters from a patient remain multiple observations within that patient's assigned partition.

Subgroup summaries are exploratory and do not establish fairness. The dataset, task definition, endpoint, and model outputs have not been clinically validated for care decisions. No clinical utility, treatment benefit, deployment readiness, or external validity of the one-year disease-progression regression model is claimed.

Further work would require a pre-specified clinical use case, a justified utility/cost function for threshold choice, calibration and decision-curve analysis, temporal and site-aware validation where identifiers and permissions allow, and prospective evaluation under appropriate governance.
