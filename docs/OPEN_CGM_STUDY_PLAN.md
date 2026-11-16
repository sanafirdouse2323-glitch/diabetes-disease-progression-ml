# Open-access CGM study: longitudinal glycaemic dynamics

## Status and research boundary

This is a separate extension to the existing tabular diabetes progression benchmark. It does not replace or rewrite the original 442-observation scikit-learn diabetes benchmark, the separate UCI hospital-readmission analysis, or their recorded results in RESULTS.md.

The new study examines longitudinal glycaemic dynamics from continuous glucose monitoring (CGM). It is not long-term diabetes disease progression, diagnosis, treatment response, complication prediction, or a causal study. CGM observations describe interstitial glucose over monitoring periods; they do not establish long-term changes in the underlying disease.

There will be no simple-model-versus-neural-network bake-off. No predictive model is to be trained until acquisition, licence checks, data-quality assessment, cohort definition, and participant-level split feasibility have been completed.

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

The audit parses timestamps as UTC for elapsed-time calculations. Naive timestamps are interpreted as UTC; no local-calendar-day metrics are computed. This avoids inventing local timezone semantics but does not establish the source's actual timezone. Source timezone conventions must be checked before any calendar-based analysis. The Shanghai source-workbook audit also preserves each workbook as a separate recording session and reports within-session gaps separately from gaps introduced by participant-level concatenation.

The report contains aggregate dataset-level fields only. It does not contain participant identifiers, filenames, per-participant metrics, raw glucose readings, or demographic metadata.

## Modelling gate (not yet authorised by this protocol)

Only after source terms, actual file counts, schema, units, sampling cadence, timestamp semantics, missingness, duplicate handling, and participant-level cohort feasibility have been reviewed should a single primary forecasting method be selected. The method must be frozen before held-out evaluation. Any forecasting windows must be split by participant, with chronology preserved within each participant. Adjacent windows from one participant must never be split across train/validation/test sets.

A later forecast horizon must be specified before model fitting. Report participant-clustered uncertainty, not sample-level confidence intervals. If an independent cohort supports external testing, preserve it as a separate dataset and report its results separately. Keep T1D and T2D analyses separate.

No model family comparison is planned. No clinical utility, diagnosis, treatment benefit, causal effect, or long-term progression claim may be made from retrospective CGM metrics.

## Reproducibility and governance

- Cite each original dataset and the Glucose-ML harmonization resource.
- Follow source licence, attribution and non-commercial/derivative restrictions.
- Keep raw and participant-level files outside Git and public workflow artifacts.
- The GitHub Actions workflow uploads only an aggregate audit JSON, never raw files or participant-level outputs.
- The original benchmark results remain in RESULTS.md and are not mixed with CGM outputs.

## Completion gate

The first batch acquisition gate passed: both downloads and both dataset-specific harmonization scripts completed successfully, and both aggregate audits completed. The Shanghai source audit reconciled the 13 missing CGM values and identified multiple recording sessions that must remain separate. Bris-T1D Open has unresolved conflicting duplicate timestamps. These issues must be handled before forecasting. No CGM model result is claimed; no model has been trained.
