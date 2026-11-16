# Open-access CGM study: longitudinal glycaemic dynamics

## Status and research boundary

This is a separate extension to the existing tabular diabetes progression benchmark. It does not replace or rewrite the original 442-observation scikit-learn diabetes benchmark, the separate UCI hospital-readmission analysis, or their recorded results in RESULTS.md.

The new study examines longitudinal glycaemic dynamics from continuous glucose monitoring (CGM). It is not long-term diabetes disease progression, diagnosis, treatment response, complication prediction, or a causal study. CGM observations describe interstitial glucose over monitoring periods; they do not establish long-term changes in the underlying disease.

There will be no simple-model-versus-neural-network bake-off. The first ShanghaiT2DM forecasting run is now complete using one fixed model after acquisition, source-workbook reconciliation, quality assessment, and participant-level window feasibility. Its specification and results are documented in OPEN_CGM_FORECAST_RESULTS.md. A 2025 publication already used ShanghaiT2DM with 20 prior readings and 30/60/90-minute horizons, so this run is a reproducibility/proof-of-pipeline exercise, not a novel forecasting method. Bris-T1D Open remains excluded from modelling until its conflicting duplicate timestamps are resolved.

## First acquisition batch

The first batch is deliberately restricted to two smaller cohorts with source-level open-access evidence:

| Dataset | Population | Catalogue/source description | First-batch role |
|---|---|---|---|
| ShanghaiT2DM | Type 2 diabetes | Glucose-ML catalogue reports 100 participants and about 13 days; NLM catalog records CC BY 4.0 for the Shanghai T1DM/T2DM dataset | Primary T2D feasibility cohort |
| Bris-T1D Open | Type 1 diabetes | University of Bristol marks the processed-state deposit Open Access and CC BY 4.0; its record describes 24 enrolled young adults and a six-month study | Separate T1D feasibility cohort |

Important count discrepancy: Glucose-ML reports 20 participants and about 183 days for Bris-T1D Open, while the official Bristol record describes 24 young adults in a six-month study. Do not choose one count by assumption. The downloaded file count and usable CGM coverage must be computed and reported separately from enrolment/catalogue counts.

Source records, exclusions, rights decisions and upstream processing caveats are documented in OPEN_CGM_DATA_SOURCES.md.

## Explicit exclusions from the first batch

- T1D-UOM: excluded until its public Zenodo record's blank Rights/License field is resolved.
- CGMacros: deferred because the upstream archive is approximately 2.14 GB and its download endpoint is PhysioNet; confirm the source's access route and licence before a later batch.
- AZT1D: deferred from the first batch because the upstream archive is approximately 1.56 GB.
- BIG IDEAs: excluded from the first batch because the upstream estimate is approximately 41.9 GB and its source is PhysioNet.
- UCHTT1DM: excluded pending resolution of its no-derivatives licence restriction.
- T1DiabetesGranada: excluded because it requires permission and acceptance of a data-use agreement.

These decisions are conservative acquisition choices, not claims that all deferred datasets are inaccessible.

## Primary research question

Can open CGM records support reproducible, participant-level characterisation of glycaemic variability and short-horizon glucose dynamics, when analyses are kept separate by dataset and diabetes type?

The first-batch acquisition and aggregate audit completed successfully on 16 November 2026. The workflow downloaded and harmonized ShanghaiT2DM and Bris-T1D Open, and a second source-workbook audit preserved Shanghai's 109 recording sessions across 100 participant filename prefixes. It reconciled 112,475 source rows to 112,462 valid glucose/timestamp pairs after identifying 13 missing CGM values. Full measured counts, session-boundary discrepancies, dataset fingerprints and blockers are in [OPEN_CGM_ACQUISITION_AUDIT.md](OPEN_CGM_ACQUISITION_AUDIT.md). No raw data or participant-level outputs were committed or uploaded.

## Data acquisition

The automated workflow clones the upstream Glucose-ML repository, installs its required packages plus openpyxl for Shanghai Excel files, and downloads only:

- shanghai (the upstream archive contains both ShanghaiT1DM and ShanghaiT2DM; only ShanghaiT2DM is harmonized);
- bris-t1d_open (the processed-state open deposit only).

The source archive is not committed to this repository. The workflow calls the dataset-specific extraction scripts directly. The upstream wrapper currently constructs a path with different letter casing from the actual harmonization directory, which can fail on case-sensitive runners; the workflow avoids that wrapper and uses the verified paths. The upstream Shanghai extractor merges multiple recording workbooks into one participant CSV; do not use those merged files to construct forecasting windows. Use original workbooks as separate sessions, grouping sessions by participant before the train/test split.

## Quality audit

Run the local aggregate-only audit after the upstream harmonization scripts have created participant CSV files:

    PYTHONPATH=src python -m diabetes_ml.cgm_audit \
      --dataset shanghai_t2dm=path/to/Standardized-datasets/ShanghaiT2DM \
      --dataset bris_t1d_open=path/to/Standardized-datasets/Bris-T1D_Open \
      --output cgm-audit-summary/audit-summary.json

The audit reports, per dataset:

1. Number of participant CSV files found and number with the expected schema.
2. Total source rows and valid timestamp/glucose pairs.
3. Missing or unparseable timestamps and missing/non-numeric glucose values.
4. Duplicate timestamps and non-monotonic timestamp steps in source order.
5. Large gaps, where a gap is more than three times that participant file's median positive sampling interval.
6. Median participant sampling interval and median participant observation span.
7. Median across-participant mean, median, and standard deviation of glucose.
8. Median across-participant proportions of readings below 70, from 70 to 180 inclusive, and above 180 mg/dL.
9. Count of readings outside 40–400 mg/dL, flagged but not removed.

The reported 70–180 metric is a fraction of readings, not a time-weighted time-in-range estimate. It must not be labelled TIR unless sampling intervals and duration weighting have been validated. Values outside 40–400 mg/dL are flagged only; the audit does not delete or impute records.

For ShanghaiT2DM, run the source-workbook audit as well. It treats each Excel workbook as one recording session and reports the number of sessions per participant prefix, source rows, missing CGM values, within-session gaps, and reconciliation against the harmonized CSV output:

    PYTHONPATH=src python -m diabetes_ml.cgm_source_audit \
      --raw-dir path/to/Original-Glucose-ML-datasets/Shanghai_raw_data/Shanghai_T2DM \
      --harmonized-dir path/to/Standardized-datasets/ShanghaiT2DM \
      --upstream-revision UPSTREAM_GIT_SHA \
      --output cgm-audit-summary/shanghai-source-audit.json

The source-workbook audit is necessary because the upstream Shanghai extractor concatenates multiple recording workbooks for some participants. For any future forecasting windows, preserve workbook boundaries and split participants—not individual windows—between train and test.

The audit parses timestamps as UTC for elapsed-time calculations. Naive timestamps are interpreted as UTC; no local-calendar-day metrics are computed. This avoids inventing local timezone semantics but does not establish the source's actual timezone. Source timezone conventions must be checked before any calendar-based analysis. The Shanghai source-workbook audit also preserves each workbook as a separate recording session and reports within-session gaps separately from gaps introduced by participant-level concatenation.

The report contains aggregate dataset-level fields only. It does not contain participant identifiers, filenames, per-participant metrics, raw glucose readings, or demographic metadata.

## Modelling and evaluation protocol

The first ShanghaiT2DM model run is complete. It uses one fixed HistGradientBoostingRegressor with 20 past CGM readings as inputs and a 30-minute target horizon. The original Excel workbook is the recording-session boundary. All sessions belonging to one participant are assigned to the same split.

- Participant split: 70 train, 15 validation, 15 held-out test (seed 42).
- Valid windows: 109,972; no window crosses a recording-session boundary.
- Model parameters are fixed in advance; no hyperparameter search is performed.
- Validation data are reported but are not used for fitting, early stopping or selection.
- Test uncertainty is estimated by resampling held-out participants as clusters.
- No model-family comparison and no baseline comparison are performed.
- Full metrics and limitations are in [OPEN_CGM_FORECAST_RESULTS.md](OPEN_CGM_FORECAST_RESULTS.md).

This experiment is a reproducibility/proof-of-pipeline exercise, not a novel forecasting method. A 2025 publication already used ShanghaiT2DM with 20 prior CGM readings and 30/60/90-minute horizons. The protocols differ, so results must not be compared directly. No novelty or superiority claim is made.

The model predicts short-horizon CGM glucose only. It does not predict long-term diabetes progression, diagnosis, complications, treatment response or clinical outcomes. No clinical utility or patient-level decision support is established.

## Remaining research gates

1. **Independent T2D validation:** identify another genuinely open T2D CGM cohort with compatible units, cadence, rights and participant-level identifiers. Bris-T1D Open is T1D and must not be presented as external validation for the Shanghai T2D model.
2. **Bris-T1D duplicate resolution:** trace the 309 conflicting duplicate timestamp groups across 13 files to the source records. Do not deduplicate by first/last/mean/median until the cause and rule are documented.
3. **Source timestamp semantics:** confirm timezone conventions before any local-calendar or circadian analysis. The current forecast uses elapsed intervals only.
4. **Out-of-range sensitivity:** the 359 Shanghai readings outside 40–400 mg/dL were retained. A separate, pre-specified sensitivity analysis may be needed before drawing conclusions.
5. **Independent evaluation:** any further model claims require an untouched external cohort and participant-clustered uncertainty. No production or clinical-readiness claim is justified.

## Reproducibility and governance

- Cite each original dataset and the Glucose-ML harmonization resource.
- Follow source licence, attribution and non-commercial/derivative restrictions.
- Keep raw and participant-level files outside Git and public workflow artifacts.
- The GitHub Actions workflow uploads only aggregate JSON reports, never raw files or participant-level outputs.
- The original benchmark results remain in RESULTS.md and are not mixed with CGM outputs.

## Completion status

The first-batch acquisition, source-workbook reconciliation, aggregate quality audit, session-preserving window-feasibility check and one fixed ShanghaiT2DM forecasting experiment all completed successfully. The Bris-T1D Open duplicate issue and independent T2D validation remain open research gates. The original diabetes progression benchmark and its historical results remain unchanged.
