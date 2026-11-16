-- MIMIC-IV v3.1: adult diabetes-coded admissions and 30-day subsequent
-- MIMIC-recorded inpatient admission outcome.
--
-- Run against the MIMIC-IV hospital schema (commonly named mimiciv_hosp).
-- This query creates a cohort/label table only. It does not build predictors.
-- Keep query outputs containing patient-level rows in authorised local storage.
--
-- Prediction time: discharge from the index admission.
-- Outcome: a subsequent admission for the same subject_id that starts after
-- discharge and within 30 days. This is MIMIC-recorded readmission, not proof
-- that no admission occurred elsewhere when no subsequent MIMIC row is found.

WITH all_admissions_ordered AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.deathtime,
        a.hospital_expire_flag,
        a.admission_type,
        a.admission_location,
        a.discharge_location,
        a.insurance,
        a.race,
        LEAD(a.admittime) OVER (
            PARTITION BY a.subject_id
            ORDER BY a.admittime, a.hadm_id
        ) AS next_mimic_admittime
    FROM mimiciv_hosp.admissions AS a
    WHERE a.admittime IS NOT NULL
),
diabetes_admissions AS (
    SELECT DISTINCT
        ao.subject_id,
        ao.hadm_id
    FROM all_admissions_ordered AS ao
    INNER JOIN mimiciv_hosp.diagnoses_icd AS d
        ON d.subject_id = ao.subject_id
       AND d.hadm_id = ao.hadm_id
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
),
adult_index_admissions AS (
    SELECT
        ao.*,
        p.gender,
        CASE
            WHEN p.anchor_age = 300 THEN '90+'
            ELSE (
                p.anchor_age
                + EXTRACT(YEAR FROM ao.admittime)::integer
                - p.anchor_year
            )::text
        END AS age_at_admission
    FROM all_admissions_ordered AS ao
    INNER JOIN diabetes_admissions AS da
        ON da.subject_id = ao.subject_id
       AND da.hadm_id = ao.hadm_id
    INNER JOIN mimiciv_hosp.patients AS p
        ON p.subject_id = ao.subject_id
    WHERE
        (
            p.anchor_age = 300
            OR (
                p.anchor_age
                + EXTRACT(YEAR FROM ao.admittime)::integer
                - p.anchor_year
            ) >= 18
        )
        AND ao.dischtime IS NOT NULL
)
SELECT
    subject_id,                         -- linkage/grouping only; never a feature
    hadm_id,                            -- row key only; never a feature
    admittime,
    dischtime,
    gender,
    age_at_admission,
    admission_type,
    admission_location,
    discharge_location,
    insurance,
    race,
    CASE
        WHEN next_mimic_admittime > dischtime
         AND next_mimic_admittime <= dischtime + INTERVAL '30 days'
        THEN 1
        ELSE 0
    END::integer AS readmitted_in_mimic_within_30_days,
    next_mimic_admittime                -- audit/outcome derivation only; never a feature
FROM adult_index_admissions
WHERE COALESCE(hospital_expire_flag, 0) = 0
ORDER BY subject_id, admittime, hadm_id;
