from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "is_fraud"

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VALIDATION_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TemporalSplitError(ValueError):
    """Raised when a temporal train/validation/test split is invalid."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalSplit:
    """Container for chronological train/validation/test datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_rows(self) -> int:
        return len(self.train)

    @property
    def validation_rows(self) -> int:
        return len(self.validation)

    @property
    def test_rows(self) -> int:
        return len(self.test)

    @property
    def total_rows(self) -> int:
        return (
            self.train_rows
            + self.validation_rows
            + self.test_rows
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    """Validate train/validation/test split ratios."""

    ratios = {
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
    }

    for name, ratio in ratios.items():
        if not 0 < ratio < 1:
            raise TemporalSplitError(
                f"{name} must be between 0 and 1, got {ratio}."
            )

    total = train_ratio + validation_ratio + test_ratio

    if abs(total - 1.0) > 1e-9:
        raise TemporalSplitError(
            "train_ratio + validation_ratio + test_ratio must equal 1.0. "
            f"Got {total}."
        )


def _validate_input_dataset(df: pd.DataFrame) -> None:
    """Validate the input feature dataset."""

    if not isinstance(df, pd.DataFrame):
        raise TemporalSplitError(
            "Input dataset must be a pandas DataFrame."
        )

    if df.empty:
        raise TemporalSplitError(
            "Input dataset cannot be empty."
        )

    required_columns = {
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise TemporalSplitError(
            "Input dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df[TIMESTAMP_COLUMN].isna().any():
        raise TemporalSplitError(
            "Timestamp column contains missing values."
        )

    if df[TARGET_COLUMN].isna().any():
        raise TemporalSplitError(
            "Target column contains missing values."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        df[TIMESTAMP_COLUMN]
    ):
        raise TemporalSplitError(
            "Timestamp column must contain datetime values."
        )


# ---------------------------------------------------------------------------
# Main split function
# ---------------------------------------------------------------------------


def temporal_train_validation_test_split(
    df: pd.DataFrame,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> TemporalSplit:
    """
    Split a feature dataset chronologically.

    Rows are ordered by timestamp and divided into train, validation,
    and test sets without randomization.

    Default split:
        70% train
        15% validation
        15% test

    The original DataFrame is never modified.
    """

    _validate_ratios(
        train_ratio,
        validation_ratio,
        test_ratio,
    )

    _validate_input_dataset(df)

    # Preserve the original DataFrame and its row ordering metadata.
    working = df.copy()

    # Stable sorting ensures deterministic behavior when multiple
    # transactions share the same timestamp.
    working = working.sort_values(
        by=TIMESTAMP_COLUMN,
        kind="mergesort",
    ).reset_index(drop=True)

    n_rows = len(working)

    train_end = int(n_rows * train_ratio)
    validation_end = int(
        n_rows * (train_ratio + validation_ratio)
    )

    if train_end <= 0:
        raise TemporalSplitError(
            "Training split would contain zero rows."
        )

    if validation_end <= train_end:
        raise TemporalSplitError(
            "Validation split would contain zero rows."
        )

    if validation_end >= n_rows:
        raise TemporalSplitError(
            "Test split would contain zero rows."
        )

    train = working.iloc[:train_end].copy()
    validation = working.iloc[
        train_end:validation_end
    ].copy()
    test = working.iloc[validation_end:].copy()

    result = TemporalSplit(
        train=train,
        validation=validation,
        test=test,
    )

    validate_temporal_split(result)

    return result


# ---------------------------------------------------------------------------
# Split validation
# ---------------------------------------------------------------------------


def validate_temporal_split(split: TemporalSplit) -> None:
    """
    Validate temporal ordering and row preservation.

    Ensures:

    - all splits are non-empty
    - total row count is preserved
    - train ends no later than validation
    - validation ends no later than test
    - each split is internally chronological
    """

    if split.train.empty:
        raise TemporalSplitError("Training split is empty.")

    if split.validation.empty:
        raise TemporalSplitError("Validation split is empty.")

    if split.test.empty:
        raise TemporalSplitError("Test split is empty.")

    total = (
        len(split.train)
        + len(split.validation)
        + len(split.test)
    )

    if total != split.total_rows:
        raise TemporalSplitError(
            "Split row counts are inconsistent."
        )

    for name, dataset in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        timestamps = dataset[TIMESTAMP_COLUMN]

        if not timestamps.is_monotonic_increasing:
            raise TemporalSplitError(
                f"{name} split is not chronologically ordered."
            )

    train_end = split.train[TIMESTAMP_COLUMN].max()
    validation_start = split.validation[TIMESTAMP_COLUMN].min()
    validation_end = split.validation[TIMESTAMP_COLUMN].max()
    test_start = split.test[TIMESTAMP_COLUMN].min()

    if train_end > validation_start:
        raise TemporalSplitError(
            "Temporal ordering violation between train and validation."
        )

    if validation_end > test_start:
        raise TemporalSplitError(
            "Temporal ordering violation between validation and test."
        )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def build_split_metadata(split: TemporalSplit) -> dict:
    """Build reproducible metadata describing the temporal split."""

    return {
        "split_strategy": "chronological",
        "train_ratio": DEFAULT_TRAIN_RATIO,
        "validation_ratio": DEFAULT_VALIDATION_RATIO,
        "test_ratio": DEFAULT_TEST_RATIO,
        "train_rows": split.train_rows,
        "validation_rows": split.validation_rows,
        "test_rows": split.test_rows,
        "total_rows": split.total_rows,
        "train_start": split.train[TIMESTAMP_COLUMN].min().isoformat(),
        "train_end": split.train[TIMESTAMP_COLUMN].max().isoformat(),
        "validation_start": (
            split.validation[TIMESTAMP_COLUMN].min().isoformat()
        ),
        "validation_end": (
            split.validation[TIMESTAMP_COLUMN].max().isoformat()
        ),
        "test_start": split.test[TIMESTAMP_COLUMN].min().isoformat(),
        "test_end": split.test[TIMESTAMP_COLUMN].max().isoformat(),
        "train_fraud_count": int(split.train[TARGET_COLUMN].sum()),
        "validation_fraud_count": int(
            split.validation[TARGET_COLUMN].sum()
        ),
        "test_fraud_count": int(split.test[TARGET_COLUMN].sum()),
    }


# ---------------------------------------------------------------------------
# File-based convenience function
# ---------------------------------------------------------------------------


def load_and_split_feature_dataset(
    dataset_path: str | Path,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> TemporalSplit:
    """Load the frozen Phase 4 feature dataset and split it chronologically."""

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {dataset_path}"
        )

    df = pd.read_parquet(dataset_path)

    return temporal_train_validation_test_split(
        df,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )