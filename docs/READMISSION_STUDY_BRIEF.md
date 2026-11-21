# Research brief: 30-day hospital readmission

## How this study relates to the main project

This repository contains a separate classification experiment in
`src/diabetes_ml/external_readmission.py`. It uses a hospital-encounter
dataset and predicts a different outcome from the one-year disease-progression
regression study. It is not an extension of that regression model and it is
not external validation of it.

## Dataset and outcome

The experiment uses the UCI dataset *Diabetes 130-US Hospitals for Years
1999–2008* (dataset 296, DOI
[10.24432/C5230J](https://doi.org/10.24432/C5230J)). The source contains
101,766 inpatient encounters from 130 US hospitals and integrated delivery
networks between 1999 and 2008, representing 71,518 unique patients.

The prediction target is readmission recorded within 30 days:

- `<30` is positive because the source records a readmission within 30 days.
- `>30` is negative for this endpoint because the recorded readmission was
  after 30 days.
- `NO` is negative because no readmission was recorded in the source data.

The `NO` label needs care. It means that the source dataset contains no
recorded readmission for that encounter; it does not prove that the patient
was never readmitted elsewhere. The original Health Facts publication notes
that encounters at out-of-network providers were not captured.

The unit of analysis is an encounter, not a unique patient. A patient can
contribute more than one encounter, so patients with more encounters also
contribute more rows to the encounter-level metrics. The splitting procedure
keeps all of one patient's encounters together.

## When the prediction is made

The intended prediction time is discharge. The feature set includes discharge
disposition and other encounter information that may not be known at
admission. For that reason, this experiment should not be described as an
admission-time prediction model.

## Data preparation and evaluation

The encounter identifier and patient number are not used as predictors.
The patient number is used only to keep a patient's encounters in the same
data partition. Before splitting, the code checks that the feature, target
and identifier tables have matching rows and indexes, and that encounter
identifiers are present and unique.

For this pre-specified baseline, race, gender and age are left out of the
predictive feature matrix and retained only for exploratory subgroup
reporting. Three coded integer fields—admission type, discharge disposition
and admission source—are treated as categorical variables rather than
continuous measurements.

Imputation, scaling and one-hot encoding are fitted inside each model
pipeline, using only the relevant training partition. The outer evaluation
uses a five-fold `StratifiedGroupKFold` split and selects one held-out fold
for each of seeds 42, 43 and 44. A patient cannot appear in both the outer
training and test partitions.

The fixed model comparison includes a prior-only baseline, logistic
regression and random forest. It is a limited, pre-specified comparison rather
than an exhaustive model search. Test folds from different seeds can overlap,
so the across-seed standard deviations describe split sensitivity; they are
not confidence intervals or independent replications.

## How the classification threshold is chosen

The positive class is relatively uncommon. A probability threshold of 0.5
can therefore produce a high accuracy while missing most positive cases. I
keep 0.5 as a reference point, not as a recommended operating threshold.

For each outer split and model, the experiment selects a threshold by
maximising F2 (with beta = 2) using three-fold, patient-grouped
out-of-fold probabilities generated only inside the outer training
partition. When several thresholds have the same F2 score, the largest one
is selected. The chosen threshold is then applied once to the untouched
outer test partition. Test labels are not used to choose the threshold.

F2 gives more weight to recall than to precision. It is a stated research
choice for exploring a recall-oriented operating point, not a clinically
validated threshold or decision rule. The experiment reports ROC-AUC,
average precision, Brier score, 10-bin equal-frequency expected calibration
error (ECE), and threshold-based accuracy, precision, sensitivity,
specificity and F2 at both the 0.5 reference cutoff and the training-selected
cutoff. The per-seed reliability bins are calculated on the held-out test
partition for evaluation only; they do not fit a post-hoc calibration model.
ECE depends on the binning and is descriptive, not proof of clinical
probability calibration.

## Results from the three-seed run

Across seeds 42–44, the held-out positive prevalence was approximately
11.16%. The table shows mean metrics at the training-selected F2 threshold.

| Model | ROC-AUC | Average precision | Brier score | 10-bin ECE | Precision | Sensitivity | Specificity | F2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prior baseline | 0.5000 | 0.1116 | 0.0992 | 0.00003 | 0.1116 | 1.0000 | 0.0000 | 0.3858 |
| Logistic Regression | 0.6639 | 0.2127 | 0.0953 | 0.0098 | 0.1424 | 0.8021 | 0.3931 | 0.4163 |
| Random Forest | 0.6752 | 0.2195 | 0.0961 | 0.0276 | 0.1477 | 0.7999 | 0.4197 | 0.4246 |

At the 0.5 reference threshold, mean sensitivity was 2.13% for logistic
regression and 0% for random forest. The training-selected thresholds raised
sensitivity to about 80%, but precision remained around 14–15%, while
specificity was around 39–42%. In other words, the increase in recall came
with many false-positive flags.

These results describe an experimental operating point, not a clinically
ready model. There are only three seeds and their test folds overlap. The
metrics do not support a statistical claim that one model is superior.

## Privacy, limitations and next steps

The workflow saves aggregate metrics, metadata and the per-seed calibration
reliability table. It does not write raw patient records or patient
identifiers to the repository or workflow artifacts. A SHA-256 fingerprint
of the retrieved source tables is recorded to help detect changes in the
upstream data. This fingerprint is useful for traceability, but it is not a
substitute for a versioned dataset or a formal data-provenance audit.

This is a retrospective benchmark from 1999–2008. It does not show how a
model would perform in a contemporary hospital or a prospective workflow.
Patient-grouped splitting prevents a patient's encounters from crossing the
training/test boundary, but it does not establish generalisation to a new
hospital. Although the records come from 130 hospitals, the public UCI schema
does not provide a hospital or facility identifier that would allow a
hospital-held-out evaluation. The metrics are encounter-weighted, so patients
with multiple encounters contribute multiple observations within their
assigned partition.

The subgroup summaries are exploratory and do not establish fairness. The
study has not been clinically validated and does not demonstrate clinical
utility, treatment benefit or deployment readiness. It also does not validate
the one-year disease-progression regression model.

Further work would need a pre-specified clinical use case, a justified
cost/utility function for threshold selection, appropriate calibration and
decision-curve analysis, temporal and site-aware validation where the data
permit it, and prospective evaluation under suitable research governance.

## Sources

Clore, J., Cios, K., DeShazo, J. and Strack, B. (2014). *Diabetes 130-US
Hospitals for Years 1999–2008*. UCI Machine Learning Repository.
[https://doi.org/10.24432/C5230J](https://doi.org/10.24432/C5230J).

Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission
Rates*. [https://doi.org/10.1155/2014/781670](https://doi.org/10.1155/2014/781670).
