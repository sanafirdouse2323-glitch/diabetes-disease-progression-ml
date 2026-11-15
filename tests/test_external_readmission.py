"""Tests for patient-grouped UCI readmission data preparation and splitting."""

import numpy as np
import pandas as pd
import pytest

from diabetes_ml.external_readmission import (
    _build_estimators,
    _calculate_binary_metrics,
    grouped_holdout_indices,
    grouped_oof_probabilities,
    prepare_readmission_frames,
    select_f2_threshold,
    _calibration_bins,
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

def test_f2_threshold_is_selected_from_supplied_training_probabilities():
    labels = pd.Series([1, 1, 1, 0, 0, 0], dtype="int8")
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])

    threshold = select_f2_threshold(labels, probabilities)

    assert threshold == pytest.approx(0.7)


def test_threshold_selection_rejects_invalid_probabilities_and_alignment():
    labels = pd.Series([1, 1, 0, 0], dtype="int8")
    with pytest.raises(ValueError, match="equal"):
        select_f2_threshold(labels, np.array([0.8, 0.7]))

    with pytest.raises(ValueError, match="between zero and one"):
        select_f2_threshold(labels, np.array([0.8, 1.2, 0.3, 0.1]))


def test_grouped_oof_predictions_cover_each_encounter_for_threshold_selection():
    features, targets, identifiers = _synthetic_frames()
    X, y, groups, _ = prepare_readmission_frames(features, targets, identifiers)

    probabilities = grouped_oof_probabilities(
        X,
        y,
        groups,
        _build_estimators(42)["Logistic Regression"],
        seed=42,
        n_splits=3,
    )

    assert probabilities.shape == (len(y),)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    threshold = select_f2_threshold(y, probabilities)
    assert 0.0 <= threshold <= 1.0


def test_binary_metrics_report_reference_and_oof_selected_thresholds():
    labels = pd.Series([1, 1, 1, 0, 0, 0], dtype="int8")
    probabilities = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1])

    metrics = _calculate_binary_metrics(
        labels, probabilities, oof_f2_threshold=0.7
    )

    assert metrics["sensitivity_at_0_5"] == pytest.approx(1.0)
    assert metrics["sensitivity_at_oof_f2"] == pytest.approx(1.0)
    assert metrics["precision_at_oof_f2"] == pytest.approx(1.0)
    assert metrics["f2_at_oof_f2"] == pytest.approx(1.0)


def test_prepare_frames_rejects_misaligned_indexes_and_duplicate_encounters():
    features, targets, identifiers = _synthetic_frames()
    shuffled = features.sample(frac=1.0, random_state=7)
    with pytest.raises(ValueError, match="identical row indexes"):
        prepare_readmission_frames(shuffled, targets, identifiers)

    features, targets, identifiers = _synthetic_frames()
    identifiers.loc[1, "encounter_id"] = identifiers.loc[0, "encounter_id"]
    with pytest.raises(ValueError, match="encounter_id values"):
        prepare_readmission_frames(features, targets, identifiers)


def test_equal_frequency_calibration_bins_and_ece_use_supplied_predictions():
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    ece, bins = _calibration_bins(labels, probabilities, n_bins=4)

    assert ece == pytest.approx(0.25)
    assert sum(row["n"] for row in bins) == len(labels)
    assert [row["n"] for row in bins] == [1, 1, 1, 1]
    assert bins[0]["mean_predicted_probability"] == pytest.approx(0.1)
    assert bins[-1]["observed_positive_rate"] == pytest.approx(1.0)


def test_calibration_bins_handle_constant_probabilities_and_reject_invalid_inputs():
    ece, bins = _calibration_bins([1, 0, 0, 0], np.full(4, 0.2), n_bins=10)
    assert ece == pytest.approx(0.05)
    assert len(bins) == 1
    assert bins[0]["n"] == 4
    with pytest.raises(ValueError, match="equal, non-zero lengths"):
        _calibration_bins([0, 1], np.array([0.2]))
    with pytest.raises(ValueError, match="between zero and one"):
        _calibration_bins([0, 1], np.array([0.2, 1.2]))
