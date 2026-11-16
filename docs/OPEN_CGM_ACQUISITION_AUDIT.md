# Open CGM first-batch acquisition and quality audit

## Status

**Status: Acquisition and aggregate audit succeeded.**

The first-batch GitHub Actions workflow completed successfully on 16 November 2026. It downloaded the open Shanghai archive and Bris-T1D Open archive, harmonized ShanghaiT2DM and the Bris-T1D Open processed-state data, audited the original Shanghai workbooks as separate recording sessions, ran the aggregate-only audit, and uploaded aggregate JSON reports only.

- Workflow run: https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/43796183352
- Repository commit tested: 34e0511a839ecd9806bbedb85c6be5032ded792f
- Upstream Glucose-ML revision: decf266461dd125ead7715bf524e5ffb9b9469c4
- Audit artifact SHA-256: fefe3f3bedcdaff01ea8ca002c0dfc2e4ad7f8f2c0070c2e31f2e990461e6e3c
- The artifact contains only dataset-level aggregates. Raw data and participant-level CSVs were not committed or uploaded.

The focused CGM audit tests passed. The repository CI, neural-framework, Docker smoke-test, and notebook-execution workflows are being checked against the latest branch commit separately.

## Observed cohort and file counts

| Audit field | ShanghaiT2DM | Bris-T1D Open |
|---|---:|---:|
| Harmonized participant CSV files | 100 | 20 |
| Files with valid schema and records | 100 | 20 |
| Harmonized CGM rows | 112,462 | 848,574 |
| Valid timestamp/glucose pairs | 112,462 | 848,574 |
| Missing/unparseable timestamps | 0 | 0 |
| Missing/non-numeric glucose values in harmonized output | 0 | 0 |
| Median participant sampling interval | 15 minutes | 5 minutes |
| Median min-to-max participant span in harmonized output | 12.68 days | 206.30 days |

### Shanghai source-workbook reconciliation

The original Shanghai study describes 100 T2D participants, 109 recording files, 15-minute CGM measurements, and 112,475 source rows/measurements. The source-workbook audit independently read all original Excel workbooks before relying on the upstream merged participant files.

| Source-workbook audit field | Observed value |
|---|---:|
| Readable recording workbooks | 109 |
| Participant filename prefixes (same grouping convention as upstream harmonizer) | 100 |
| Participants with one recording workbook | 92 |
| Participants with two recording workbooks | 7 |
| Participants with three recording workbooks | 1 |
| Source workbook rows | 112,475 |
| Rows with missing CGM value | 13 |
| Rows with missing or unparseable timestamp | 0 |
| Finite CGM values with valid timestamps | 112,462 |
| Harmonized participant CSV rows | 112,462 |
| Difference between valid source pairs and harmonized rows | 0 |
| Median recording-session span | 12.68 days |
| Median recording-session sampling interval | 15 minutes |
| Duplicate timestamp groups within source workbooks | 0 |
| Non-monotonic timestamp steps within source workbooks | 0 |
| Large gaps within source workbooks under the audit rule | 2 |

The 13-row discrepancy is reconciled: the 112,475 source rows include 13 rows with missing CGM values, leaving 112,462 finite timestamp/glucose pairs. The upstream harmonizer drops rows missing either timestamp or glucose; its 112,462 output rows exactly match the valid source pairs. No valid CGM rows were lost in this reconciliation.

The upstream Shanghai extractor combines multiple Excel recording workbooks into one CSV per participant. The source audit found 109 workbooks across 100 participant filename prefixes, with 8 participants having multiple workbooks. Therefore, the 100 harmonized CSV files are participant-level containers, **not 100 separate recording sessions**. Future time-series windows must preserve the original workbook/session boundaries rather than treating concatenated visits as one uninterrupted trace.

The source article's narrative session distribution states 94 participants with one period, 6 with two, and 1 with three; those counts sum to 101 participant entries. The actual source filenames, grouped using the same prefix rule as the upstream harmonizer, yield 92 with one workbook, 7 with two, and 1 with three (100 prefixes and 109 workbooks). This published-versus-file-level distribution discrepancy is unresolved and is recorded rather than silently reconciled.

For Bris-T1D Open, the audit found 20 harmonized participant CSV files. This matches the Glucose-ML catalogue's 20-participant entry, but the official Bristol deposit describes 24 young adults enrolled in the study. The audit establishes the number of harmonized files, not why the enrolment and file counts differ. The median observed timestamp span of 206.30 days is an elapsed first-to-last timestamp span, not proof of continuous sensor wear.

## Glucose distribution summaries

The following values are medians across participant-level statistics, not pooled statistics across all CGM readings.

| Participant-level statistic | ShanghaiT2DM | Bris-T1D Open |
|---|---:|---:|
| Median of participant mean glucose (mg/dL) | 139.50 | 153.25 |
| Median of participant median glucose (mg/dL) | 131.40 | 146.70 |
| Median of participant glucose SD (mg/dL) | 37.63 | 55.45 |
| Median participant fraction of readings below 70 mg/dL | 0.44% | 1.57% |
| Median participant fraction of readings from 70–180 mg/dL inclusive | 83.61% | 69.54% |
| Median participant fraction of readings above 180 mg/dL | 14.36% | 27.41% |

These are descriptive proportions of recorded readings. They are **not duration-weighted time-in-range (TIR) estimates**, not clinical targets, and not evidence of treatment effectiveness. Medians of separate participant-level fractions need not sum to 100%.

## Data-quality findings

| Quality check | ShanghaiT2DM | Bris-T1D Open |
|---|---:|---:|
| Duplicate timestamp rows | 0 | 309 |
| Duplicate timestamp groups | 0 | 309 |
| Duplicate groups with conflicting glucose values | 0 | 309 |
| Participants with duplicate timestamps | 0 | 13 of 20 |
| Non-monotonic timestamp steps in source order | 2 | 0 |
| Participants with non-monotonic timestamp steps | 2 | 0 |
| Large gaps under the defined rule | 11 | 900 |
| Participants with at least one large gap | 10 of 100 | 20 of 20 |
| Participants with readings outside 40–400 mg/dL | 19 of 100 | 17 of 20 |
| Readings outside 40–400 mg/dL | 359 | 1,006 |

A large gap is an interval greater than three times that participant file's median positive sampling interval. This is a data-continuity flag, not a count of known missing samples or proof of sensor failure.

### Important Bris-T1D Open blocker

All 309 Bris duplicate timestamp groups contain more than one glucose value at the same parsed timestamp, across 13 participant files. The audit did **not** choose the first value, last value, mean, or median; it preserved and counted the observations. These conflicting duplicate timestamps must be traced to the source records and a handling rule pre-specified before Bris data are used for time-series modelling. Until then, Bris-derived glucose distribution summaries remain preliminary.

ShanghaiT2DM has no duplicate timestamps or non-monotonic steps within any of the 109 source workbooks. The merged participant CSVs show two non-monotonic steps and 11 large-gap flags; the session-preserving source audit finds zero within-session reversals and only two within-session large gaps. The nine additional large-gap flags equal the nine extra recording workbooks beyond one per participant, consistent with session boundaries being counted as long gaps after concatenation. The two merged-file reversals also appear only after concatenation. This confirms that the merged participant CSVs must not be used as if each were one continuous session.

Values outside 40–400 mg/dL were flagged, not removed or imputed. Their presence is not automatically evidence that the readings are erroneous; source context and device conventions must be checked.

## Interpretation and next gate

1. **Acquisition passed** for the first batch: both downloads and both dataset-specific extraction scripts completed successfully.
2. **Shanghai source reconciliation passed:** 109 workbooks, 100 participant prefixes, 112,475 source rows, 13 missing CGM values, and 112,462 valid pairs exactly matching the harmonized output.
3. **Shanghai session-boundary gate is now explicit:** 8 participants have multiple recording workbooks. Future windows must be created within individual workbooks/sessions, not across the upstream merged participant CSVs.
4. **Bris-T1D Open is not yet ready for forecasting** because of 309 conflicting duplicate timestamp groups across 13 of 20 files. No deduplication decision has been made.
5. The 359 Shanghai and 1,006 Bris readings outside 40–400 mg/dL were flagged, not filtered. Source-aware review remains necessary.
6. No forecasting model was trained, no model comparison was performed, and no clinical utility or long-term disease-progression claim is made.

The first-batch acquisition and audit are a feasibility result only. The next stage is session-preserving window feasibility for ShanghaiT2DM and source-level duplicate review for Bris-T1D Open—not a model-family bake-off.

## Reproducibility fingerprints

These SHA-256 values identify the harmonized CSV contents processed in this run. The audit does not publish filenames or participant identifiers.

| Dataset | Total harmonized CSV bytes | Content SHA-256 |
|---|---:|---|
| ShanghaiT2DM | 2,989,625 | c8fc31277270935d8bd923af3bb6e7bc3e34d0ec8de0a02d99ca2390efea831a |
| Bris-T1D Open | 21,951,250 | 9f49feef19b199cd801544a0cbad0e619fed40b59d1e7329bf4e35442883fede |

The fingerprint is calculated from sorted participant CSV file contents and byte lengths. It does not expose the filenames. The upstream code revision is recorded above so that future harmonization changes can be distinguished from this run.

## Source records

- Shanghai data: [Figshare DOI 10.6084/m9.figshare.20444397](https://doi.org/10.6084/m9.figshare.20444397). The [NLM Dataset Catalog](https://datasetcatalog.nlm.nih.gov/dataset?q=0002621900) records CC BY 4.0 for the Shanghai T1DM/T2DM data.
- Bris-T1D Open: [University of Bristol data record](https://doi.org/10.5523/bris.33z5jc8fa6tob21ptrugzqog08). The processed-state deposit is Open Access and CC BY 4.0. The separate raw-state deposit is restricted and was not used.
- Upstream harmonization tools: [Glucose-ML-Project](https://github.com/Augmented-Health-Lab/Glucose-ML-Project), revision decf266461dd125ead7715bf524e5ffb9b9469c4.

The original tabular benchmark's historical metrics remain in RESULTS.md. These CGM audit outputs are separate and must not be merged into those benchmark results.
