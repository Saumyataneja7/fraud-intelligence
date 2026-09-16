from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fraud_intelligence.features.dataset import (
    get_model_feature_columns,
)
from fraud_intelligence.features.leakage_validation import (
    IDENTIFIER_COLUMNS,
    RAW_TRANSACTION_COLUMNS,
    TARGET_COLUMNS,
)
from fraud_intelligence.ml.splits import TemporalSplit

from fraud_intelligence.features.leakage_validation import (
    ENGINEERED_FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    RAW_TRANSACTION_COLUMNS,
    TARGET_COLUMNS,
)


TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "is_fraud"


@dataclass(frozen=True)
class ModelDataset:
    """Prepared model inputs and targets for one temporal split."""

    X_train: pd.DataFrame
    y_train: pd.Series

    X_validation: pd.DataFrame
    y_validation: pd.Series

    X_test: pd.DataFrame
    y_test: pd.Series

    train_ids: pd.Series
    validation_ids: pd.Series
    test_ids: pd.Series

    feature_columns: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)


class ModelDatasetPreparationError(ValueError):
    """Raised when model dataset preparation fails."""


def _validate_split_columns(split: TemporalSplit) -> None:
    """Validate that every split contains the required columns."""

    required = {
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
        "transaction_id",
    }

    for name, dataset in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        missing = required - set(dataset.columns)

        if missing:
            raise ModelDatasetPreparationError(
                f"{name} split is missing required columns: "
                f"{sorted(missing)}"
            )


def _validate_target(series: pd.Series, name: str) -> None:
    """Validate the binary fraud target."""

    if series.isna().any():
        raise ModelDatasetPreparationError(
            f"{name} contains missing target values."
        )

    unique_values = set(series.unique())

    if not unique_values.issubset({0, 1}):
        raise ModelDatasetPreparationError(
            f"{name} must contain only 0/1 values. "
            f"Found: {sorted(unique_values)}"
        )


def _prepare_features(
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Extract and validate the model feature matrix."""

    missing_features = set(feature_columns) - set(dataset.columns)

    if missing_features:
        raise ModelDatasetPreparationError(
            "Dataset is missing model features: "
            f"{sorted(missing_features)}"
        )

    X = dataset.loc[:, feature_columns].copy()

    forbidden = (
        set(TARGET_COLUMNS)
        | set(IDENTIFIER_COLUMNS)
        | set(RAW_TRANSACTION_COLUMNS)
        | {TIMESTAMP_COLUMN}
    )

    leaked_columns = forbidden & set(X.columns)

    if leaked_columns:
        raise ModelDatasetPreparationError(
            "Forbidden columns found in model features: "
            f"{sorted(leaked_columns)}"
        )

    if X.columns.duplicated().any():
        duplicated = X.columns[X.columns.duplicated()].tolist()

        raise ModelDatasetPreparationError(
            f"Duplicate model feature columns found: {duplicated}"
        )

    return X


def prepare_model_dataset(
    split: TemporalSplit,
) -> ModelDataset:
    """
    Convert temporal feature splits into model-ready datasets.

    Only the approved engineered model features are included in X.
    The fraud target is returned separately as y.

    Transaction IDs are retained separately for downstream error analysis
    and explainability, but are never included in X.
    """

    _validate_split_columns(split)

    # Phase 4 defines the authoritative model feature list.
    feature_columns = get_model_feature_columns(
        ENGINEERED_FEATURE_COLUMNS
    )

    if not feature_columns:
        raise ModelDatasetPreparationError(
            "No model features were found."
        )

    # Ensure every temporal split contains the same model features.
    for name, dataset in (
        ("validation", split.validation),
        ("test", split.test),
    ):
        missing = set(feature_columns) - set(dataset.columns)

        if missing:
            raise ModelDatasetPreparationError(
                f"{name} split is missing model features: "
                f"{sorted(missing)}"
            )

    X_train = _prepare_features(
        split.train,
        feature_columns,
    )

    X_validation = _prepare_features(
        split.validation,
        feature_columns,
    )

    X_test = _prepare_features(
        split.test,
        feature_columns,
    )

    y_train = split.train[TARGET_COLUMN].copy()
    y_validation = split.validation[TARGET_COLUMN].copy()
    y_test = split.test[TARGET_COLUMN].copy()

    _validate_target(y_train, "y_train")
    _validate_target(y_validation, "y_validation")
    _validate_target(y_test, "y_test")

    train_ids = split.train["transaction_id"].copy()
    validation_ids = split.validation["transaction_id"].copy()
    test_ids = split.test["transaction_id"].copy()

    # Reset indexes so X/y/IDs have clean aligned positional indices.
    X_train = X_train.reset_index(drop=True)
    X_validation = X_validation.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_validation = y_validation.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train_ids = train_ids.reset_index(drop=True)
    validation_ids = validation_ids.reset_index(drop=True)
    test_ids = test_ids.reset_index(drop=True)

    result = ModelDataset(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        feature_columns=tuple(feature_columns),
    )

    validate_model_dataset(result)

    return result


def validate_model_dataset(
    dataset: ModelDataset,
) -> None:
    """Validate the prepared model dataset."""

    expected_features = set(dataset.feature_columns)

    for name, X, y, ids in (
        (
            "train",
            dataset.X_train,
            dataset.y_train,
            dataset.train_ids,
        ),
        (
            "validation",
            dataset.X_validation,
            dataset.y_validation,
            dataset.validation_ids,
        ),
        (
            "test",
            dataset.X_test,
            dataset.y_test,
            dataset.test_ids,
        ),
    ):
        if len(X) != len(y):
            raise ModelDatasetPreparationError(
                f"{name}: X/y row count mismatch."
            )

        if len(X) != len(ids):
            raise ModelDatasetPreparationError(
                f"{name}: X/transaction ID row count mismatch."
            )

        if set(X.columns) != expected_features:
            raise ModelDatasetPreparationError(
                f"{name}: model feature columns do not match "
                "the expected feature set."
            )

        forbidden = (
            set(TARGET_COLUMNS)
            | set(IDENTIFIER_COLUMNS)
            | set(RAW_TRANSACTION_COLUMNS)
            | {TIMESTAMP_COLUMN}
        )

        if forbidden & set(X.columns):
            raise ModelDatasetPreparationError(
                f"{name}: forbidden columns present in X."
            )

        if ids.duplicated().any():
            raise ModelDatasetPreparationError(
                f"{name}: duplicate transaction IDs found."
            )

    # Ensure the temporal datasets do not share transactions.
    train_ids = set(dataset.train_ids)
    validation_ids = set(dataset.validation_ids)
    test_ids = set(dataset.test_ids)

    if train_ids & validation_ids:
        raise ModelDatasetPreparationError(
            "Transaction IDs overlap between train and validation."
        )

    if train_ids & test_ids:
        raise ModelDatasetPreparationError(
            "Transaction IDs overlap between train and test."
        )

    if validation_ids & test_ids:
        raise ModelDatasetPreparationError(
            "Transaction IDs overlap between validation and test."
        )


def build_model_dataset_summary(
    dataset: ModelDataset,
) -> dict:
    """Build metadata for the prepared model datasets."""

    return {
        "n_features": dataset.n_features,
        "feature_columns": list(dataset.feature_columns),
        "train_rows": len(dataset.X_train),
        "validation_rows": len(dataset.X_validation),
        "test_rows": len(dataset.X_test),
        "train_fraud_count": int(dataset.y_train.sum()),
        "validation_fraud_count": int(
            dataset.y_validation.sum()
        ),
        "test_fraud_count": int(dataset.y_test.sum()),
        "train_fraud_rate": float(dataset.y_train.mean()),
        "validation_fraud_rate": float(
            dataset.y_validation.mean()
        ),
        "test_fraud_rate": float(dataset.y_test.mean()),
    }