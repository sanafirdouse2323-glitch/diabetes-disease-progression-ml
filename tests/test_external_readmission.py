"""Tests for patient-grouped UCI readmission data preparation and splitting."""

import numpy as np
import pandas as pd
import pytest

from diabetes_ml.external_readmission import (
    grouped_holdout_indices,
    prepare_readmission_frames,
)


def _synthetic_frames(n_patients: int = 50):
    patient_ids = np.repeat(np.arange(n_patients), 2)
    labels = np.repeat(
        np.where(np.arange(n_patients) % 5 == 0, "<30", "NO"),
        2,
    )
    features = pd.DataFrame(
        {
            "race": np.where(patient_ids % 2 == 0, "Caucasian", "AfricanAmerican"),
            "gender": np.where(patient_ids % 3 == 0, "Female", "Male"),
            "age": np.where(patient_ids % 4 == 0, "[70-80)", "[50-60)"),
            "time_in_hospital": (patient_ids % 10) + 1,
            "admission_type_id": (patient_ids % 3) + 1,
            "medical_specialty": np.where(patient_ids % 2 == 0, "InternalMedicine", "?"),
        }
    )
    targets = pd.DataFrame({"readmitted": labels})
    identifiers = pd.DataFrame(
        {
            "encounter_id": np.arange(len(patient_ids)),
            "patient_nbr": patient_ids,
        }
    )
    return features, targets, identifiers


def test_prepare_frames_maps_30_day_outcome_and_excludes_demographics():
    features, targets, identifiers = _synthetic_frames()
    X, y, groups, audit = prepare_readmission_frames(features, targets, identifiers)

    assert y.iloc[0] == 1
    assert y.iloc[2] == 0
    assert set(y.unique()) == {0, 1}
    assert not ({"race", "gender", "age", "patient_nbr", "encounter_id"} & set(X.columns))
    assert set(audit.columns) == {"race", "gender", "age"}
    assert groups.nunique() == 50
    assert X["medical_specialty"].isna().any()


def test_prepare_frames_rejects_unknown_outcome_and_missing_patient_id():
    features, targets, identifiers = _synthetic_frames()
    targets.loc[0, "readmitted"] = "unknown"
    with pytest.raises(ValueError, match="Unexpected or missing"):
        prepare_readmission_frames(features, targets, identifiers)

    features, targets, identifiers = _synthetic_frames()
    identifiers = identifiers.drop(columns=["patient_nbr"])
    with pytest.raises(ValueError, match="patient_nbr"):
        prepare_readmission_frames(features, targets, identifiers)


def test_grouped_holdout_has_both_classes_and_no_patient_overlap():
    features, targets, identifiers = _synthetic_frames()
    X, y, groups, _ = prepare_readmission_frames(features, targets, identifiers)
    train, test = grouped_holdout_indices(y, groups, seed=42, n_splits=5)

    assert len(train) + len(test) == len(X)
    assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))
    assert set(y.iloc[train].unique()) == {0, 1}
    assert set(y.iloc[test].unique()) == {0, 1}


@pytest.mark.parametrize("seeds", [[42], [42, 42], [42, True], [42, -1]])
def test_seed_validation_is_strict(seeds):
    from diabetes_ml.external_readmission import _validated_seeds

    with pytest.raises(ValueError):
        _validated_seeds(seeds)
