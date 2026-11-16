# ShanghaiT2DM 30-minute CGM forecasting experiment

## Status

**One fixed-model forecasting experiment completed successfully.** This is a separate CGM extension; it does not replace or update the original tabular diabetes progression benchmark in RESULTS.md.

- Workflow run: https://github.com/sanafirdouse2323-glitch/diabetes-disease-progression-ml/actions/runs/43797065112
- Tested repository commit: da9c25dcfbe879a7a572074fd3f9bd4cb7c8b3fc
- Upstream Glucose-ML revision: decf266461dd125ead7715bf524e5ffb9b9469c4
- Aggregate artifact SHA-256: 2741412c8ced5d8fe8dccf6c15849f56944292dae94b3eecd5fdf257ab6c3a74
- Focused CGM tests: 10 passed.
- Repository CI across Python 3.10–3.13, neural-framework checks, Docker smoke test, and end-to-end notebook execution: passed.

Only aggregate JSON reports were uploaded. Raw source workbooks, participant-level CSVs, participant IDs, individual predictions, and model binaries were not uploaded or committed.

## Research task

Predict the ShanghaiT2DM CGM glucose reading **30 minutes ahead** from the 20 most recent readings in the same original recording workbook/session.

- Source cadence: 15 minutes.
- Input: 20 consecutive CGM readings.
- First-to-last elapsed history span: 285 minutes.
- Horizon: two sampling steps, exactly 30 minutes.
- Units: mg/dL.
- Each window must have exact 15-minute spacing from the first input reading through the target.
- Windows never cross workbook/session boundaries.
- Missing values and irregular intervals break windows; no interpolation or imputation is used.
- Participant filename prefixes are assigned to train/validation/test before windows are assembled. All sessions from one participant remain in one split.

The 20-reading/30-minute configuration is not novel. A published 2025 study, [Interpretable glucose forecasting for type 2 diabetes across traditional, deep, and large language models](https://doi.org/10.1038/s41598-025-32373-4), used 20 prior CGM readings with 30-, 60- and 90-minute horizons and included ShanghaiT2DM in its data. That study also used six static patient features and compared several model families. This repository's experiment is a narrower reproducibility/proof-of-pipeline exercise: one fixed HistGradientBoostingRegressor, CGM lags only, session-preserving windows, and a participant-held-out 70/15/15 split. Its metrics must not be compared directly with the published paper because the inputs, splits and protocols differ. No novelty claim is made.

## Window feasibility and split

| Split | Participants | Recording sessions | Windows |
|---|---:|---:|---:|
| Training | 70 | 75 | 75,045 |
| Validation | 15 | 15 | 16,939 |
| Held-out test | 15 | 19 | 17,988 |
| **Total** | **100** | **109** | **109,972** |

All 109 sessions and all 100 participants produced at least one valid window. There were no overlapping recording-session intervals within participants. The split was fixed with seed 42. Windows were not randomly divided across participants.

The windows overlap in time and are not independent observations. The evaluation therefore reports participant-macro metrics and participant-cluster bootstrap intervals, rather than treating 17,988 test windows as 17,988 independent people.

## Model

One model was fitted: scikit-learn HistGradientBoostingRegressor.

| Parameter | Fixed value |
|---|---:|
| max_iter | 150 |
| learning_rate | 0.05 |
| max_leaf_nodes | 15 |
| l2_regularization | 1.0 |
| early_stopping | False |
| random_state | 42 |

The model uses only the 20 past CGM readings. No demographic or medication features are used. Hyperparameters were fixed in advance; no search or model-family comparison was performed. The validation set was evaluated but not used for fitting, early stopping or selection. The held-out test participants were not used for any training or tuning.

## Results

**Evaluation status: internal participant-held-out test completed.**

| Metric | Validation | Held-out test |
|---|---:|---:|
| Pooled-window MAE (mg/dL) | 11.335 | 9.521 |
| Pooled-window RMSE (mg/dL) | 15.969 | 13.310 |
| Pooled-window R² | 0.918 | 0.938 |
| Participant-macro MAE (mg/dL) | 11.318 | 9.382 |
| Participant-macro RMSE (mg/dL) | 15.541 | 12.949 |

Participant-macro values are the unweighted mean of participant-level errors, so each participant contributes equally regardless of their number of windows.

### Participant-cluster bootstrap uncertainty

The held-out test set contains 15 participant clusters. A percentile bootstrap resampled those 15 participants with replacement for 1,000 replicates.

| Test metric | Estimate | 95% participant-cluster bootstrap interval |
|---|---:|---:|
| Participant-macro MAE (mg/dL) | 9.382 | 8.417–10.305 |
| Participant-macro RMSE (mg/dL) | 12.949 | 11.568–14.189 |

These intervals quantify uncertainty across the 15 held-out participants under this split. They do not represent external-population uncertainty.

## Interpretation and limitations

1. This is one internal participant-held-out experiment on one open ShanghaiT2DM cohort. There is no independent external cohort evaluation.
2. The test set contains 15 participants. Participant-cluster bootstrap intervals are more appropriate than sample-level intervals, but 15 clusters remain a small basis for uncertainty estimation.
3. No persistence baseline or competing model was evaluated. The metrics therefore describe this fixed model's performance; they do not establish that it improves on a simpler method or is superior to another model.
4. The 359 Shanghai readings outside 40–400 mg/dL were flagged in the quality audit but retained in this experiment. No out-of-range sensitivity analysis has been performed.
5. The 13 source rows with missing CGM values do not create forecast targets; exact-cadence window construction also prevents windows crossing the resulting gaps.
6. The model predicts short-horizon CGM glucose. It does not predict long-term diabetes progression, diagnosis, complications, treatment response, or clinical outcomes.
7. No clinical utility, safety, fairness, causal effect, or patient-level decision support is established.

The Bris-T1D Open cohort was not used for model training or evaluation. A source-level audit confirms 309 conflicting duplicate timestamp groups across 13 of 20 files; all 309 are within one recorded device label, and the processed-state files do not preserve enough row-level provenance to select a correct value. The separate raw-state deposit is restricted. This cohort remains excluded until the dataset maintainers clarify the conflicts.

## Reproduction

After downloading the open datasets and preserving the original Shanghai Excel workbooks, run:

    PYTHONPATH=src python -m diabetes_ml.cgm_forecast \
      --raw-dir path/to/Original-Glucose-ML-datasets/Shanghai_raw_data/Shanghai_T2DM \
      --seed 42 \
      --bootstrap-replicates 1000 \
      --upstream-revision decf266461dd125ead7715bf524e5ffb9b9469c4 \
      --output cgm-audit-summary/shanghai-forecast-metrics.json

The workflow also runs the source-workbook audit, window-feasibility counter, and aggregate data-quality audit. The workflow artifact contains only JSON summaries; raw and participant-level data remain in the ephemeral runner.

## Data source

ShanghaiT2DM is part of the open Shanghai T1DM/T2DM dataset. The source Figshare record is DOI 10.6084/m9.figshare.20444397; the NLM Dataset Catalog records CC BY 4.0. The source paper is https://doi.org/10.1038/s41597-023-01940-7. Cite the source dataset and follow its attribution terms when reusing results.
