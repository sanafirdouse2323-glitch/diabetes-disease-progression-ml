# Diabetes Disease Progression with Machine Learning

This repository documents two existing studies and a separate CGM research extension. The original benchmark and its recorded results are preserved; the new CGM work is not a replacement for them.

## Existing studies (preserved)

1. **Tabular diabetes progression benchmark:** scikit-learn's diabetes dataset (442 observations, 10 preprocessed baseline features) with a continuous one-year progression target. The full historical experimental record is in [RESULTS.md](RESULTS.md).
2. **Hospital readmission study:** a separate UCI Diabetes 130-US Hospitals encounter/readmission analysis. It has a different population, prediction task and outcome; it is not external validation of the progression benchmark.

The original benchmark uses a continuous one-year progression measure. It does not diagnose diabetes, predict long-term complications, or establish clinical utility. See [RESULTS.md](RESULTS.md) for experiment-specific metrics, split designs and limitations.

## Open-access longitudinal CGM extension (separate study)

The new extension examines **longitudinal glycaemic dynamics from continuous glucose monitoring (CGM)**. It is not long-term diabetes disease progression, diagnosis, treatment response, or a causal study.

The first automated acquisition batch is deliberately limited to:
- **ShanghaiT2DM:** public Figshare source; CC BY 4.0 is recorded by the NLM Dataset Catalog for the Shanghai T1DM/T2DM data. Only the T2DM subfolder is harmonized.
- **Bris-T1D Open:** University of Bristol processed-state open deposit; official record states Open Access and CC BY 4.0. The separate raw-state deposit is restricted and is not used.

There is a source-count discrepancy for Bris-T1D Open: Glucose-ML lists 20 participants/about 183 days, while the Bristol record describes 24 enrolled young adults/six months. Actual downloaded file counts and usable coverage must be measured rather than inferred from either summary.

T1D-UOM is deferred until its public Zenodo record's Rights/License field is clarified. CGMacros, AZT1D and BIG IDEAs are deferred from the first batch due to download size/source checks; UCHTT1DM and T1DiabetesGranada are excluded for licence/access reasons. See [data-source decisions](docs/OPEN_CGM_DATA_SOURCES.md).

### Automated acquisition and audit

A GitHub Actions workflow on the research branch downloads only the first-batch open datasets, uses the upstream dataset-specific extraction scripts, and runs the local aggregate-only quality audit. The workflow must succeed before acquisition can be described as complete. It uploads only a dataset-level summary JSON; raw and participant-level data are not committed or uploaded as artifacts.

The local audit can also be run manually after harmonization:

    PYTHONPATH=src python -m diabetes_ml.cgm_audit \
      --dataset shanghai_t2dm=path/to/Standardized-datasets/ShanghaiT2DM \
      --dataset bris_t1d_open=path/to/Standardized-datasets/Bris-T1D_Open \
      --output cgm-audit-summary/audit-summary.json

The audit checks schema, missingness, duplicate/non-monotonic timestamps, sampling intervals, long gaps, observation span, glucose distributions and out-of-range flags. Fractions below 70, 70–180 and above 180 mg/dL are fractions of readings, not duration-weighted time-in-range estimates. No model is trained in this acquisition workflow. No simple-versus-neural bake-off is planned.

- [CGM study protocol](docs/OPEN_CGM_STUDY_PLAN.md)
- [Dataset source, licence and exclusion decisions](docs/OPEN_CGM_DATA_SOURCES.md)
- [Upstream Glucose-ML repository](https://github.com/Augmented-Health-Lab/Glucose-ML-Project)
- [Glucose-ML dataset catalogue](https://www.glucose-ml-project.com/)

## Scope and interpretation

The two existing studies and the CGM extension have different datasets, outcomes and analysis methods. Their metrics must never be combined into one table or described as validation of one another. The CGM extension is limited to glycaemic dynamics and must not be presented as long-term diabetes progression or clinical decision support.

For the original project environment, experiment commands and dependency instructions, see the sections below and the project source.
