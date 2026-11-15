# Separate Research Brief: 30-Day Hospital Readmission

## Scope and relationship to the main project

This document describes a separate classification experiment in `src/diabetes_ml/external_readmission.py`. It is **not** an extension of the one-year disease-progression regression experiment using scikit-learn's 442-row `load_diabetes` dataset, and it is **not external validation** of that regression model. The datasets, units of observation, outcomes, and estimands are different.

## Dataset and endpoint

The experiment retrieves UCI dataset 296, *Diabetes 130-US Hospitals for Years 1999-2008* (DOI: [10.24432/C5230J](https://doi.org/10.24432/C5230J)). UCI describes 101,766 inpatient encounters from 130 US hospitals and integrated delivery networks during 1999–2008, representing 71,518 unique patients.

The target is mapped as:

- `<30`: positive, a readmission recorded within 30 days.
- `>30`: negative for the within-30-day endpoint because the recorded readmission occurred after 30 days.
- `NO`: negative because no readmission was recorded in the source data.

The `NO` value must not be interpreted as proof that no readmission occurred anywhere. The original Health Facts paper states that encounters from out-of-network providers were not captured, so some out-of-network readmissions may be absent from this source outcome.

The unit of analysis is an **encounter**, not a unique patient. All encounters belonging to one patient are assigned to the same outer partition, but patients with more encounters contribute more rows to encounter-level metrics.

The source outcome is limited to the contributing Health Facts data. The original publication reports that out-of-network provider encounters were not captured; consequently, `NO` means no readmission recorded in the source data, not confirmed absence of a readmission in every healthcare setting.

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

## Results from the current three-seed run

Across seeds 42–44, the held-out positive prevalence was approximately 11.16%. Mean test metrics at the training-selected F2 thresholds were:

| Model | ROC-AUC | Average precision | Precision | Sensitivity | Specificity | F2 |
|---|---:|---:|---:|---:|---:|---:|
| Prior baseline | 0.5000 | 0.1116 | 0.1116 | 1.0000 | 0.0000 | 0.3858 |
| Logistic Regression | 0.6639 | 0.2127 | 0.1424 | 0.8043 | 0.3916 | 0.4168 |
| Random Forest | 0.6752 | 0.2195 | 0.1477 | 0.7999 | 0.4197 | 0.4246 |

At the 0.5 reference cutoff, mean sensitivity was 2.13% for Logistic Regression and 0% for Random Forest. The selected F2 thresholds raised sensitivity to about 80%, but precision remained about 14–15% and specificity about 39–42%. The recall gain therefore comes with many false-positive flags. These metrics describe an experimental operating point, not clinical readiness. There are only three outer seeds, their test folds overlap, and no statistical superiority claim is made.

## Limitations

This is a retrospective, historical benchmark from 1999–2008. It does not establish performance in contemporary hospitals, on new sites, or in prospective workflows. Patient-grouped splitting addresses within-patient overlap between training and test partitions; it does not establish hospital-level generalisation. Metrics are encounter-weighted, and repeated encounters from a patient remain multiple observations within that patient's assigned partition.

Subgroup summaries are exploratory and do not establish fairness. The dataset, task definition, endpoint, and model outputs have not been clinically validated for care decisions. No clinical utility, treatment benefit, deployment readiness, or external validity of the one-year disease-progression regression model is claimed.

Further work would require a pre-specified clinical use case, a justified utility/cost function for threshold choice, calibration and decision-curve analysis, temporal and site-aware validation where identifiers and permissions allow, and prospective evaluation under appropriate governance.

Dataset source: Clore, J., Cios, K., DeShazo, J., and Strack, B. (2014), *Diabetes 130-US Hospitals for Years 1999-2008*, UCI Machine Learning Repository, DOI [10.24432/C5230J](https://doi.org/10.24432/C5230J). Source-capture limitation: Strack et al. (2014), *Impact of HbA1c Measurement on Hospital Readmission Rates*, DOI [10.1155/2014/781670](https://doi.org/10.1155/2014/781670).
