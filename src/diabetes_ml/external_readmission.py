"""Group-aware evaluation on the UCI Diabetes 130-US Hospitals dataset.

This is a separate 30-day readmission classification task. It is not an
external validation of the scikit-learn one-year disease-progression target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


UCI_DATASET_ID = 296
EXPECTED_ROWS = 101_766
DEFAULT_SEEDS = (42, 43, 44)
DEMOGRAPHIC_COLUMNS = ("race", "gender", "age")
CATEGORICAL_INTEGER_COLUMNS = (
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
)
METRIC_COLUMNS = (
    "roc_auc",
    "average_precision",
    "brier_score",
    "accuracy_at_0_5",
    "sensitivity_at_0_5",
    "specificity_at_0_5",
)


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Validate at least two unique non-negative scikit-learn seeds."""
    if isinstance(seeds, (str, bytes)):
        raise ValueError("seeds must be a sequence of at least two unique integers.")
    try:
        values = tuple(seeds)
    except TypeError as exc:
        raise ValueError(
            "seeds must be a sequence of at least two unique integers."
        ) from exc
    if len(values) < 2:
        raise ValueError("At least two seeds are required.")
    if any(
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral)
        for seed in values
    ):
        raise ValueError("Every seed must be an integer; booleans are not valid seeds.")
    normalized = tuple(int(seed) for seed in values)
    if any(seed < 0 or seed > 2**32 - 1 for seed in normalized):
        raise ValueError("Every seed must be between 0 and 2**32 - 1.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("seeds must be unique.")
    return normalized


def prepare_readmission_frames(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    identifiers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Validate UCI tables and prepare predictors, target, patient groups, and audit fields.

    Race, gender, and age are retained for subgroup reporting but excluded from
    the predictive features in this pre-specified baseline analysis. Patient
    and encounter identifiers are used for grouping/provenance only, never as
    model inputs.
    """
    if not all(isinstance(frame, pd.DataFrame) for frame in (features, targets, identifiers)):
        raise TypeError("features, targets, and identifiers must be pandas DataFrames.")
    if features.empty or targets.empty or identifiers.empty:
        raise ValueError("The UCI feature, target, and identifier tables must not be empty.")
    if not (len(features) == len(targets) == len(identifiers)):
        raise ValueError("UCI feature, target, and identifier tables must have equal row counts.")
    if "readmitted" not in targets.columns:
        raise ValueError("The UCI target table must contain the 'readmitted' column.")
    if "patient_nbr" not in identifiers.columns:
        raise ValueError("The UCI identifier table must contain 'patient_nbr' for patient-grouped splitting.")

    identifier_leaks = {"encounter_id", "patient_nbr"} & set(features.columns)
    if identifier_leaks:
        raise ValueError(
            "Identifier columns must not be used as predictors: "
            + ", ".join(sorted(identifier_leaks))
        )
    missing_demographics = set(DEMOGRAPHIC_COLUMNS) - set(features.columns)
    if missing_demographics:
        raise ValueError(
            "The UCI feature table is missing required subgroup fields: "
            + ", ".join(sorted(missing_demographics))
        )

    labels = targets["readmitted"].astype("string").str.strip()
    allowed_labels = {"<30", ">30", "NO"}
    observed_labels = set(labels.dropna().unique())
    if labels.isna().any() or not observed_labels.issubset(allowed_labels):
        unexpected = sorted(observed_labels - allowed_labels)
        raise ValueError(
            f"Unexpected or missing readmission labels; unexpected values: {unexpected}"
        )

    # The UCI task defines early readmission as the '<30' category. Both
    # '>30' and 'NO' are negative for the specific within-30-day endpoint.
    y = labels.eq("<30").astype("int8").rename("readmitted_within_30_days")

    raw_features = features.reset_index(drop=True).replace("?", np.nan)
    audit_groups = (
        raw_features.loc[:, list(DEMOGRAPHIC_COLUMNS)]
        .fillna("Unknown")
        .astype(str)
        .reset_index(drop=True)
    )
    X = (
        raw_features.drop(columns=list(DEMOGRAPHIC_COLUMNS))
        .reset_index(drop=True)
    )
    patient_groups = identifiers["patient_nbr"].reset_index(drop=True)
    if patient_groups.isna().any():
        raise ValueError("patient_nbr contains missing values; safe grouped splitting is impossible.")
    patient_groups = patient_groups.astype("string").rename("patient_nbr")

    if y.nunique() != 2:
        raise ValueError("The readmission target must contain both positive and negative cases.")
    if not X.columns.is_unique:
        raise ValueError("Predictor column names must be unique.")

    return X, y.reset_index(drop=True), patient_groups, audit_groups


def _dataset_fingerprint(
    identifiers: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> str:
    """Hash a deterministic CSV serialization without writing patient-level data."""
    combined = pd.concat(
        [
            identifiers.reset_index(drop=True),
            features.reset_index(drop=True),
            targets.reset_index(drop=True),
        ],
        axis=1,
    )
    payload = combined.to_csv(
        index=False,
        na_rep="",
        lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_uci_readmission_data():
    """Fetch UCI dataset 296 and return validated tables plus a data fingerprint."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            'Install the optional dependency with: uv sync --locked --extra external-data'
        ) from exc

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    features = dataset.data.features
    targets = dataset.data.targets
    identifiers = dataset.data.ids
    if features is None or targets is None or identifiers is None:
        raise ValueError("UCI dataset 296 did not return features, targets, and identifier tables.")
    if len(features) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} UCI encounters, received {len(features):,}; "
            "the upstream dataset may have changed."
        )

    X, y, patient_groups, audit_groups = prepare_readmission_frames(
        features, targets, identifiers
    )
    fingerprint = _dataset_fingerprint(identifiers, features, targets)
    return X, y, patient_groups, audit_groups, fingerprint


def grouped_holdout_indices(
    y: pd.Series,
    patient_groups: pd.Series,
    seed: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one stratified group fold and prove no patient crosses the split."""
    if len(y) != len(patient_groups):
        raise ValueError("y and patient_groups must have equal lengths.")
    if y.isna().any() or patient_groups.isna().any():
        raise ValueError("Targets and patient groups must not contain missing values.")
    if y.nunique() != 2:
        raise ValueError("Both target classes are required for stratified group splitting.")
    if isinstance(n_splits, (bool, np.bool_)) or not isinstance(n_splits, Integral) or n_splits < 2:
        raise ValueError("n_splits must be an integer greater than or equal to 2.")
    if patient_groups.nunique() < n_splits:
        raise ValueError("There are fewer unique patients than requested group folds.")

    splitter = StratifiedGroupKFold(
        n_splits=int(n_splits),
        shuffle=True,
        random_state=int(seed),
    )
    placeholder = np.zeros((len(y), 1), dtype=np.float32)
    train_indices, test_indices = next(
        splitter.split(placeholder, y.to_numpy(), groups=patient_groups.to_numpy())
    )
    train_patients = set(patient_groups.iloc[train_indices])
    test_patients = set(patient_groups.iloc[test_indices])
    if train_patients & test_patients:
        raise RuntimeError("Patient overlap detected between training and test partitions.")
    if y.iloc[train_indices].nunique() != 2 or y.iloc[test_indices].nunique() != 2:
        raise ValueError("The selected grouped fold must contain both target classes.")
    return train_indices, test_indices


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build train-fitted preprocessing with nominal treatment of coded categories."""
    categorical_columns = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for column in CATEGORICAL_INTEGER_COLUMNS:
        if column in X.columns and column not in categorical_columns:
            categorical_columns.append(column)
    numeric_columns = [
        column for column in X.columns if column not in categorical_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("No usable predictor columns remain after excluding identifiers and demographics.")
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


def _build_estimators(seed: int) -> dict[str, object]:
    """Return fixed, pre-specified reference models for the external task."""
    return {
        "Prior baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            solver="liblinear",
            max_iter=2_000,
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def _calculate_binary_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities must all be finite.")
    predictions = (probabilities >= 0.5).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "accuracy_at_0_5": float(accuracy_score(y_true, predictions)),
        "sensitivity_at_0_5": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity_at_0_5": float(tn / (tn + fp)) if tn + fp else float("nan"),
    }


def run_external_readmission_evaluation(
    output_dir: str | Path = "external-readmission-results",
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate pre-specified models on patient-disjoint, held-out UCI encounters."""
    seed_values = _validated_seeds(seeds)
    X, y, patient_groups, audit_groups, fingerprint = load_uci_readmission_data()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, int | float | str]] = []
    subgroup_records: list[dict[str, int | float | str]] = []

    for seed in seed_values:
        train_indices, test_indices = grouped_holdout_indices(
            y, patient_groups, seed=seed, n_splits=5
        )
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        groups_train = patient_groups.iloc[train_indices]
        groups_test = patient_groups.iloc[test_indices]
        test_audit = audit_groups.iloc[test_indices]

        for model_name, estimator in _build_estimators(seed).items():
            pipeline = Pipeline(
                steps=[
                    ("preprocess", _build_preprocessor(X)),
                    ("model", estimator),
                ]
            )
            pipeline.fit(X_train, y_train)
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            metrics = _calculate_binary_metrics(y_test, probabilities)
            records.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "train_encounters": int(len(train_indices)),
                    "test_encounters": int(len(test_indices)),
                    "train_patients": int(groups_train.nunique()),
                    "test_patients": int(groups_test.nunique()),
                    "test_positive_prevalence": float(y_test.mean()),
                    **metrics,
                }
            )

            for attribute in DEMOGRAPHIC_COLUMNS:
                for subgroup_value, positions in test_audit.groupby(
                    attribute, sort=True, dropna=False
                ).indices.items():
                    local_positions = np.asarray(positions, dtype=int)
                    # The subgroup index is relative to the test partition.
                    subgroup_y = y_test.iloc[local_positions]
                    subgroup_probabilities = probabilities[local_positions]
                    subgroup_ids = groups_test.iloc[local_positions]
                    subgroup_metrics = (
                        _calculate_binary_metrics(subgroup_y, subgroup_probabilities)
                        if subgroup_y.nunique() == 2
                        else {
                            metric: float("nan")
                            for metric in METRIC_COLUMNS
                        }
                    )
                    subgroup_records.append(
                        {
                            "seed": seed,
                            "model": model_name,
                            "attribute": attribute,
                            "subgroup": str(subgroup_value),
                            "test_encounters": int(len(local_positions)),
                            "test_patients": int(subgroup_ids.nunique()),
                            "positive_prevalence": float(subgroup_y.mean()),
                            **subgroup_metrics,
                        }
                    )

    per_seed = pd.DataFrame.from_records(records)
    summary_records = []
    for model_name, frame in per_seed.groupby("model", sort=False):
        row: dict[str, str | float | int] = {
            "model": str(model_name),
            "n_seeds": int(frame["seed"].nunique()),
        }
        for metric in METRIC_COLUMNS:
            values = frame[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = (
                float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else float("nan")
            )
        summary_records.append(row)

    summary = pd.DataFrame.from_records(summary_records)
    subgroup_summary = pd.DataFrame.from_records(subgroup_records)

    per_seed.to_csv(out_path / "per_seed_metrics.csv", index=False)
    summary.to_csv(out_path / "summary_metrics.csv", index=False)
    subgroup_summary.to_csv(out_path / "subgroup_metrics.csv", index=False)

    metadata = {
        "dataset": "Diabetes 130-US Hospitals for Years 1999-2008",
        "uci_dataset_id": UCI_DATASET_ID,
        "dataset_doi": "10.24432/C5230J",
        "dataset_license": "CC BY 4.0",
        "source_url": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
        "dataset_sha256_of_canonical_tables": fingerprint,
        "expected_encounters": EXPECTED_ROWS,
        "observed_encounters": int(len(X)),
        "unique_patients": int(patient_groups.nunique()),
        "predictor_count": int(X.shape[1]),
        "excluded_model_features": [
            "encounter_id",
            "patient_nbr",
            *DEMOGRAPHIC_COLUMNS,
        ],
        "subgroup_audit_attributes": list(DEMOGRAPHIC_COLUMNS),
        "target": {
            "name": "readmitted_within_30_days",
            "positive_label": "<30",
            "negative_labels": [">30", "NO"],
            "threshold_for_reported_class_metrics": 0.5,
        },
        "split": {
            "method": "StratifiedGroupKFold",
            "folds": 5,
            "selected_fold": 0,
            "group_key": "patient_nbr",
            "patient_overlap_between_train_and_test": False,
            "outer_test_fraction_approximately": 0.20,
        },
        "seeds": list(seed_values),
        "models": list(_build_estimators(seed_values[0]).keys()),
        "preprocessing": (
            "Median imputation and scaling for numeric columns; most-frequent "
            "imputation and one-hot encoding for categorical columns. Preprocessing "
            "is fitted inside each training partition. Rare categories are grouped "
            "using OneHotEncoder(min_frequency=20)."
        ),
        "interpretation": (
            "This is a related but different diabetes classification task, not an "
            "external validation of the one-year disease-progression regression model. "
            "Rows are encounters; grouped splitting keeps all encounters for a patient "
            "within one partition. Metrics are encounter-level. Repeated-seed test "
            "partitions overlap, so across-seed standard deviations are descriptive "
            "split-sensitivity summaries, not confidence intervals. Subgroup results "
            "are exploratory and do not establish fairness or clinical utility. "
            "A 0.5 threshold is a reporting convention, not a clinical decision threshold."
        ),
        "files": [
            "per_seed_metrics.csv",
            "summary_metrics.csv",
            "subgroup_metrics.csv",
            "metadata.json",
        ],
        "raw_patient_level_data_written": False,
    }
    with (out_path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return per_seed, summary, subgroup_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a patient-grouped UCI diabetes readmission classification study. "
            "This is separate from the disease-progression regression benchmark."
        )
    )
    parser.add_argument(
        "--output",
        default="external-readmission-results",
        help="Directory for aggregate evaluation outputs.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="At least two unique integer seeds (default: 42, 43, 44).",
    )
    args = parser.parse_args()
    per_seed, summary, _ = run_external_readmission_evaluation(
        output_dir=args.output,
        seeds=args.seeds,
    )
    print(f"Completed {per_seed['seed'].nunique()} patient-grouped seeds.")
    print("\n=== Across-seed descriptive summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
