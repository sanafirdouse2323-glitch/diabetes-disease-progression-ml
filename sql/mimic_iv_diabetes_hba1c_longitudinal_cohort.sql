-- MIMIC-IV v3.1: feasibility extraction of longitudinal HbA1c measurements
-- among adults with a recorded diabetes diagnosis.
--
-- Run against the MIMIC-IV hospital schema (commonly mimiciv_hosp).
-- This is a discovery/feasibility query, not a validated production cohort.
-- Review returned item labels and units against the v3.1 data dictionary.
-- Keep patient-level query outputs in authorised local storage only.
--
-- Outcome candidate: follow-up HbA1c minus baseline HbA1c, measured in the
-- same unit 180-365 days after baseline. It is a glycaemic-control change,
-- not a complete or causal measure of diabetes disease progression.

-- QUERY 1: inspect candidate HbA1c laboratory item IDs, labels, units and
-- observation counts. Do not assume all matching labels/units are equivalent.
SELECT
    dl.itemid,
    dl.label,
    dl.fluid,
    dl.category,
    le.valueuom,
    COUNT(*) AS measurement_count,
    COUNT(DISTINCT le.subject_id) AS patient_count
FROM mimiciv_hosp.labevents AS le
INNER JOIN mimiciv_hosp.d_labitems AS dl
    ON dl.itemid = le.itemid
WHERE
    le.valuenum IS NOT NULL
    AND le.charttime IS NOT NULL
    AND (
        LOWER(dl.label) LIKE '%hemoglobin a1c%'
        OR LOWER(dl.label) LIKE '%hba1c%'
        OR LOWER(dl.label) LIKE '%glycated hemoglobin%'
    )
GROUP BY dl.itemid, dl.label, dl.fluid, dl.category, le.valueuom
ORDER BY patient_count DESC, measurement_count DESC;


-- QUERY 2: construct one candidate baseline/follow-up pair per patient.
WITH diabetes_diagnosis AS (
    SELECT
        a.subject_id,
        MIN(a.admittime) AS first_recorded_diabetes_admission
    FROM mimiciv_hosp.admissions AS a
    INNER JOIN mimiciv_hosp.diagnoses_icd AS d
        ON d.subject_id = a.subject_id
       AND d.hadm_id = a.hadm_id
    WHERE
        (
            d.icd_version = 9
            AND d.icd_code LIKE '250%'
        )
        OR
        (
            d.icd_version = 10
            AND d.icd_code ~ '^E(08|09|10|11|12|13)'
        )
    GROUP BY a.subject_id
),
candidate_hba1c_items AS (
    SELECT
        itemid,
        label,
        fluid,
        category
    FROM mimiciv_hosp.d_labitems
    WHERE
        LOWER(label) LIKE '%hemoglobin a1c%'
        OR LOWER(label) LIKE '%hba1c%'
        OR LOWER(label) LIKE '%glycated hemoglobin%'
),
hba1c_events AS (
    SELECT
        le.subject_id,
        le.hadm_id,
        le.itemid,
        le.charttime,
        le.valuenum,
        NULLIF(BTRIM(le.valueuom), '') AS valueuom,
        ci.label
    FROM mimiciv_hosp.labevents AS le
    INNER JOIN candidate_hba1c_items AS ci
        ON ci.itemid = le.itemid
    WHERE
        le.valuenum IS NOT NULL
        AND le.charttime IS NOT NULL
),
adult_baseline_candidates AS (
    SELECT
        he.subject_id,
        he.hadm_id AS baseline_hadm_id,
        he.itemid AS baseline_itemid,
        he.label AS baseline_item_label,
        he.charttime AS baseline_charttime,
        he.valuenum AS baseline_hba1c,
        he.valueuom AS baseline_unit,
        p.gender,
        CASE
            WHEN p.anchor_age = 300 THEN '90+'
            ELSE (
                p.anchor_age
                + EXTRACT(YEAR FROM he.charttime)::integer
                - p.anchor_year
            )::text
        END AS age_at_baseline,
        ROW_NUMBER() OVER (
            PARTITION BY he.subject_id
            ORDER BY he.charttime, he.itemid
        ) AS baseline_order
    FROM hba1c_events AS he
    INNER JOIN diabetes_diagnosis AS dd
        ON dd.subject_id = he.subject_id
    INNER JOIN mimiciv_hosp.patients AS p
        ON p.subject_id = he.subject_id
    WHERE
        he.charttime >= dd.first_recorded_diabetes_admission
        AND (
            p.anchor_age = 300
            OR (
                p.anchor_age
                + EXTRACT(YEAR FROM he.charttime)::integer
                - p.anchor_year
            ) >= 18
        )
),
baseline AS (
    SELECT *
    FROM adult_baseline_candidates
    WHERE baseline_order = 1
)
SELECT
    b.subject_id,                         -- linkage/grouping only; never a feature
    b.baseline_hadm_id,                   -- audit only; never a feature
    b.baseline_itemid,
    b.baseline_item_label,
    b.baseline_charttime,
    b.baseline_hba1c,
    b.baseline_unit,
    b.gender,
    b.age_at_baseline,
    f.followup_hadm_id,                   -- audit only; never a feature
    f.followup_itemid,
    f.followup_item_label,
    f.followup_charttime,
    f.followup_hba1c,
    f.followup_unit,
    f.followup_days,
    CASE
        WHEN f.followup_hba1c IS NOT NULL
        THEN f.followup_hba1c - b.baseline_hba1c
        ELSE NULL
    END AS hba1c_change_same_unit
FROM baseline AS b
LEFT JOIN LATERAL (
    SELECT
        he.hadm_id AS followup_hadm_id,
        he.itemid AS followup_itemid,
        he.label AS followup_item_label,
        he.charttime AS followup_charttime,
        he.valuenum AS followup_hba1c,
        he.valueuom AS followup_unit,
        EXTRACT(EPOCH FROM (he.charttime - b.baseline_charttime)) / 86400.0
            AS followup_days
    FROM hba1c_events AS he
    WHERE
        he.subject_id = b.subject_id
        AND he.charttime >= b.baseline_charttime + INTERVAL '180 days'
        AND he.charttime <= b.baseline_charttime + INTERVAL '365 days'
        AND he.valueuom IS NOT DISTINCT FROM b.baseline_unit
    ORDER BY
        ABS(
            EXTRACT(EPOCH FROM (
                he.charttime - (b.baseline_charttime + INTERVAL '365 days')
            ))
        ),
        he.charttime,
        he.itemid
    LIMIT 1
) AS f ON TRUE
ORDER BY b.subject_id;
