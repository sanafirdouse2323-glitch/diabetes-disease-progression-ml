# Open CGM data sources and acquisition decisions

This document records the source-level checks used to select the first acquisition batch. Dataset catalogue counts are not treated as observed counts; the aggregate audit must establish what the downloaded files actually contain.

## First acquisition batch

| Cohort | Source record | Access and licence evidence | Decision |
|---|---|---|---|
| ShanghaiT2DM | [Figshare dataset DOI 10.6084/m9.figshare.20444397](https://doi.org/10.6084/m9.figshare.20444397); Glucose-ML catalogue | The NLM Dataset Catalog records CC BY 4.0 for the Shanghai T1DM/T2DM data. The Glucose-ML catalogue lists ShanghaiT2DM as open access and reports 100 participants and about 13 days. | Include in first batch. Download the shared Shanghai archive but harmonize only the ShanghaiT2DM subfolder. |
| Bris-T1D Open | [University of Bristol data record, DOI 10.5523/bris.33z5jc8fa6tob21ptrugzqog08](https://doi.org/10.5523/bris.33z5jc8fa6tob21ptrugzqog08) | The official record labels this component Open Access and states CC BY 4.0. It contains processed-state device data. The separate raw-state deposit is explicitly restricted and is not used. | Include processed-state CGM only. |

## Source discrepancies and exclusions

- Bris-T1D participant count: the Glucose-ML catalogue reports 20 participants and about 183 days, while the official Bristol record describes 24 young adults in a six-month study. These may refer to different denominators, but that explanation is not assumed. The downloaded file count and usable records must be measured; catalogue and enrolment counts must not be substituted for them.
- T1D-UOM: the public Zenodo record identifies version V1.0.3 and the dataset contents, but the record's displayed Rights/License field is blank. It is excluded from the first batch until the rights statement is independently confirmed.
- CGMacros: the Glucose-ML downloader points to PhysioNet and estimates a download of about 2.14 GB for the archive. Although the Glucose-ML project lists it among open-access downloads, it is deferred from the first batch to avoid assuming the access route and licence conditions without checking the source record directly.
- AZT1D: the upstream downloader estimates an archive of about 1.56 GB. It is deferred to a later batch; the first feasibility pass uses smaller sources with explicit source-level access/licence evidence.
- BIG IDEAs: excluded because its downloader points to PhysioNet and the upstream estimate is about 41.9 GB. It is not part of the first batch.
- UCHTT1DM: excluded because a secondary benchmark records a no-derivatives CC BY-NC-ND 4.0 condition. Do not process or redistribute derivatives unless the original terms are independently resolved.
- T1DiabetesGranada: excluded because it requires permission and acceptance of a data-use agreement and is not suitable for the no-approval requirement.

## Upstream processing caveat

The Shanghai data descriptor reports 100 participants, 109 recording files and 112,475 CGM measurements at 15-minute intervals ([source paper](https://doi.org/10.1038/s41597-023-01940-7)). The downloaded source-workbook audit found 109 readable workbooks, 100 filename prefixes, and 112,475 rows; 13 rows had missing CGM values, leaving 112,462 valid pairs. Those 112,462 pairs exactly match the harmonized CSV row count. The source paper's narrative breakdown of 94 one-session, 6 two-session and 1 three-session participants sums to 101; the actual file-prefix audit yields 92/7/1 (100 prefixes, 109 workbooks). The discrepancy is recorded in the acquisition audit rather than silently resolved.

The Glucose-ML downloader and dataset-specific harmonizers are third-party tools. The acquisition workflow calls the dataset-specific extraction scripts directly rather than the upstream harmonization wrapper. The wrapper's current source computes a 2_Harmonize-CGM-datasets path, while the repository directory is named 2_Harmonize-cgm-datasets; this case mismatch can fail on case-sensitive runners. The direct calls use the actual repository path.

The Bris-T1D Open extraction script converts its source bg field from mmol/L to mg/dL by multiplying by 18, as implemented upstream. The ShanghaiT2DM extractor reads the source Excel files and retains the CGM (mg / dl) field (with a fallback column spelling for one participant). These transformations are upstream behavior and must be checked against the generated audit and source documentation; this project does not silently modify the source data.

## Reproducibility and privacy

- Raw downloads and harmonized participant files remain in the ephemeral GitHub Actions runner workspace.
- The workflow uploads only a dataset-level aggregate JSON report. It does not upload raw files, per-participant CSVs, metadata tables, filenames, or participant-level metrics.
- The report's glucose fractions are proportions of readings, not duration-weighted time-in-range estimates.
- Timestamps are parsed in UTC for elapsed-time calculations. Timezone-naive source timestamps are interpreted as UTC; no local-calendar-day analysis is performed.
- The audit flags values outside 40–400 mg/dL but does not delete or impute them.
- Dataset-level summaries are not model results and do not establish clinical utility.
