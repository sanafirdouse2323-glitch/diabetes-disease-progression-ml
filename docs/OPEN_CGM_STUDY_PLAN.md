# Open-access CGM study: longitudinal glycaemic dynamics

## Status and data access

This protocol replaces the earlier MIMIC-IV proposal. It uses only datasets listed as **open access** in the Glucose-ML collection. No PhysioNet credentialing, CITI training, MIMIC-IV DUA, or institutional data-access approval is intended for the selected downloads.

Glucose-ML catalog: https://www.glucose-ml-project.com/
Glucose-ML repository: https://github.com/Augmented-Health-Lab/Glucose-ML-Project
Download instructions: https://github.com/Augmented-Health-Lab/Glucose-ML-Project/blob/main/1_Auto-scripts/README.md

The Glucose-ML project reports 20+ datasets overall and provides automated download/harmonisation scripts for its open-access subset. Controlled-access datasets must be excluded even if the project provides a harmonisation script for them.

## Dataset shortlist

The first download batch should be limited to the following catalogued open-access cohorts:

| Dataset | Population in catalog | Participants | Approximate CGM duration | Intended role |
|---|---|---:|---:|---|
| ShanghaiT2DM | Type 2 diabetes | 100 | 13 days | Primary T2D cohort |
| CGMacros Dexcom | Type 2, prediabetes, no diabetes | 45 | 11 days | Device/cohort comparison; verify participant overlap with Libre |
| CGMacros Libre | Type 2, prediabetes, no diabetes | 45 | 12 days | Device/cohort comparison; verify participant overlap with Dexcom |
| Bris-T1D Open | Type 1 diabetes | 20 | 183 days | Longer-duration, separate T1D cohort |
| T1D-UOM | Type 1 diabetes | 17 | 96 days | Separate T1D cohort |
| AZT1D | Type 1 diabetes | 25 | 42 days | Additional T1D cohort |

Participant counts and durations are catalog summaries, not independently verified row counts. Confirm the dataset cards, source terms, actual downloaded files, participant IDs, dates, and usable-data counts before analysis. Do not add all cohorts into one undifferentiated population. Keep type 1 and type 2 analyses distinct, and do not count CGMacros Dexcom and Libre as independent people until participant overlap is checked.

## Research question and scope

**Primary question:** Can continuous glucose monitoring time series support reproducible characterisation and short-horizon forecasting of glucose dynamics in people with diabetes, with evaluation separated by dataset and diabetes type?

This is a study of **glycaemic dynamics**, not long-term diabetes disease progression, diagnosis, treatment response, or causal effect. CGM records measure interstitial glucose over time. They do not, by themselves, establish changes in complications or the underlying progression of diabetes.

## Initial execution workflow

Clone the upstream Glucose-ML repository and install its documented dependencies. Use only the open-access download keys:

```bash
git clone https://github.com/Augmented-Health-Lab/Glucose-ML-Project.git
cd Glucose-ML-Project
pip install -r dependencies.txt
cd 1_Auto-scripts

# T2D-oriented first batch plus independent longitudinal T1D cohorts
python auto-download-open-datasets.py shanghai cgmacros bris-t1d_open t1d-uom azt1d

# Harmonise only the open-access datasets downloaded above
python auto-harmonize-CGM-datasets.py shanghait2dm cgmacros_dexcom cgmacros_libre bris-t1d_open t1d-uom azt1d
```

The upstream downloader asks for confirmation and estimates the download size. The Shanghai download bundle contains both Shanghai T1DM and T2DM; the harmonisation command selects the requested cohort. CGMacros downloads both device datasets. Do not commit downloaded raw data or participant-level records to this repository.

## Feasibility and data-quality checks before modelling

1. Verify the upstream dataset source, access label, licence/terms, citation, and revision for each selected cohort.
2. Record download URL, retrieval date, archive size, SHA-256 checksum, and upstream commit/revision.
3. Inspect schema, units, timestamp timezone conventions, sensor sampling frequency, gaps, duplicate timestamps, impossible values, and missingness.
4. Count participants and valid CGM days after cleaning; report distributions rather than relying on catalog totals.
5. Verify participant overlap between CGMacros Dexcom and Libre before any combined analysis.
6. Confirm diabetes type from documented metadata; do not infer type from glucose traces alone.
7. Keep each dataset and diabetes type identifiable throughout analysis. Do not silently harmonise incompatible variables or protocols.
8. Create participant-level train/validation/test partitions. Never split adjacent time windows from the same participant across partitions.
9. Use chronological ordering within each participant. Fit all preprocessing and any tuning using training participants only.
10. Freeze the analysis protocol before evaluating the held-out participants.

If a selected source unexpectedly requires an account, approval, credentialing, or an additional data-use agreement, stop and exclude it from the no-gated-access analysis unless its terms are independently checked and accepted. Do not attempt to bypass access controls.

## Analysis plan

There will be **no simple-model-versus-neural-network bake-off**. Pre-specify one primary time-series method after inspecting the data structure and missingness. Do not select a method based on held-out test performance.

The first analysis should characterise:
- mean/median glucose and distribution;
- time in range (70–180 mg/dL) and time below/above range, where sampling and clinical context support those metrics;
- within-person glucose variability;
- daily and participant-level patterns;
- missingness, sensor gaps, and effective observation duration.

A secondary forecasting experiment may predict glucose at a pre-specified short horizon (for example, 30 or 60 minutes) using only past observations. Evaluate with participant-held-out and, where possible, cross-dataset testing. Report MAE/RMSE and clinically interpretable error summaries, but do not claim clinical utility from retrospective metrics.

Use appropriate glucose units and document all transformations. Avoid treating every CGM sample as an independent patient-level observation. Confidence intervals should account for participant clustering (e.g., participant-level bootstrap), not sample-level resampling.

## Limitations

- The selected open cohorts are modest in participant count and vary in duration, population, device, protocol, and metadata.
- The strongest long-duration open cohorts in this shortlist are predominantly type 1 diabetes; the T2D-focused Shanghai cohort has a shorter monitoring window.
- CGM is a measure of glucose dynamics, not a complete measure of diabetes progression.
- Short monitoring windows do not establish long-term progression or complication risk.
- Sensor data are subject to gaps, device-specific error, and differences in sampling and collection protocols.
- Cross-dataset generalisation may be limited by population and device differences.
- Results are research analyses, not diagnosis, treatment recommendations, or evidence of clinical effectiveness.

## Reproducibility and governance

Keep downloaded datasets outside Git. Store local files under an ignored data directory and verify that no patient-level data, participant-level predictions, or restricted metadata enter commits, issues, workflow artifacts, or public logs. Follow each dataset's licence, attribution, and usage conditions even when it is openly downloadable. Publish code and only those aggregate outputs permitted by the source terms.

## Completion gate

The dataset selection is considered feasible only after downloads succeed without gated approval, licences/terms are recorded, source files are verified, and participant-level data-quality summaries are generated. No model results or participant counts are claimed in this repository until those steps have actually run.
