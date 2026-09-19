from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd


TIMESTAMP_COLUMN: Final[str] = "timestamp"
TRANSACTION_ID_COLUMN: Final[str] = "transaction_id"

TRAIN_FRACTION: Final[float] = 0.70
VALIDATION_FRACTION: Final[float] = 0.15
TEST_FRACTION: Final[float] = 0.15


class TemporalGraphSplitError(ValueError):
    """Raised when a temporal Graph ML split is invalid."""


@dataclass(frozen=True)
class TemporalGraphSplit:
    """Chronological transaction split for Graph ML."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def total_size(self) -> int:
        return (
            len(self.train)
            + len(self.validation)
            + len(self.test)
        )

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def validation_size(self) -> int:
        return len(self.validation)

    @property
    def test_size(self) -> int:
        return len(self.test)


def _validate_input(
    transactions: pd.DataFrame,
) -> None:
    if not isinstance(
        transactions,
        pd.DataFrame,
    ):
        raise TemporalGraphSplitError(
            "transactions must be a pandas DataFrame."
        )

    if transactions.empty:
        raise TemporalGraphSplitError(
            "transactions cannot be empty."
        )

    required = {
        TIMESTAMP_COLUMN,
        TRANSACTION_ID_COLUMN,
    }

    missing = required - set(
        transactions.columns
    )

    if missing:
        raise TemporalGraphSplitError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if transactions[
        TRANSACTION_ID_COLUMN
    ].isna().any():
        raise TemporalGraphSplitError(
            "transaction_id cannot contain null values."
        )

    if transactions[
        TRANSACTION_ID_COLUMN
    ].duplicated().any():
        raise TemporalGraphSplitError(
            "transaction_id must be unique."
        )

    if transactions[
        TIMESTAMP_COLUMN
    ].isna().any():
        raise TemporalGraphSplitError(
            "timestamp cannot contain null values."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        transactions[TIMESTAMP_COLUMN]
    ):
        raise TemporalGraphSplitError(
            "timestamp must be a datetime column."
        )


def _validate_fractions() -> None:
    total = (
        TRAIN_FRACTION
        + VALIDATION_FRACTION
        + TEST_FRACTION
    )

    if abs(total - 1.0) > 1e-12:
        raise TemporalGraphSplitError(
            "Train, validation, and test fractions "
            "must sum to 1."
        )


def _sorted_transactions(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a stable chronological copy.

    Timestamp is the primary ordering key and transaction_id
    is the deterministic tie-breaker.
    """

    return (
        transactions
        .sort_values(
            by=[
                TIMESTAMP_COLUMN,
                TRANSACTION_ID_COLUMN,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )


def _calculate_boundaries(
    total_rows: int,
) -> tuple[int, int]:
    """
    Calculate deterministic row boundaries.

    Boundaries are based on row positions after chronological
    ordering, matching the Phase 5 temporal split semantics.
    """

    train_end = int(
        total_rows * TRAIN_FRACTION
    )

    validation_end = (
        train_end
        + int(
            total_rows
            * VALIDATION_FRACTION
        )
    )

    if train_end <= 0:
        raise TemporalGraphSplitError(
            "Training split would be empty."
        )

    if validation_end <= train_end:
        raise TemporalGraphSplitError(
            "Validation split would be empty."
        )

    if validation_end >= total_rows:
        raise TemporalGraphSplitError(
            "Test split would be empty."
        )

    return train_end, validation_end


def split_transactions(
    transactions: pd.DataFrame,
) -> TemporalGraphSplit:
    """
    Split transactions chronologically into 70/15/15 partitions.

    No randomization is performed.
    """

    _validate_input(transactions)
    _validate_fractions()

    ordered = _sorted_transactions(
        transactions
    )

    train_end_index, validation_end_index = (
        _calculate_boundaries(len(ordered))
    )

    train = (
        ordered
        .iloc[:train_end_index]
        .copy()
    )

    validation = (
        ordered
        .iloc[
            train_end_index:validation_end_index
        ]
        .copy()
    )

    test = (
        ordered
        .iloc[validation_end_index:]
        .copy()
    )

    train_end = train[
        TIMESTAMP_COLUMN
    ].max()

    validation_end = validation[
        TIMESTAMP_COLUMN
    ].max()

    test_end = test[
        TIMESTAMP_COLUMN
    ].max()

    result = TemporalGraphSplit(
        train=train,
        validation=validation,
        test=test,
        train_end=pd.Timestamp(train_end),
        validation_end=pd.Timestamp(
            validation_end
        ),
        test_end=pd.Timestamp(test_end),
    )

    validate_temporal_split(result)

    return result


def validate_temporal_split(
    split: TemporalGraphSplit,
) -> None:
    """Validate chronological and partition invariants."""

    if not isinstance(
        split,
        TemporalGraphSplit,
    ):
        raise TemporalGraphSplitError(
            "split must be a TemporalGraphSplit."
        )

    frames = (
        split.train,
        split.validation,
        split.test,
    )

    for frame in frames:
        _validate_input(frame)

    train_ids = set(
        split.train[TRANSACTION_ID_COLUMN]
    )

    validation_ids = set(
        split.validation[TRANSACTION_ID_COLUMN]
    )

    test_ids = set(
        split.test[TRANSACTION_ID_COLUMN]
    )

    if train_ids & validation_ids:
        raise TemporalGraphSplitError(
            "Train and validation transactions overlap."
        )

    if train_ids & test_ids:
        raise TemporalGraphSplitError(
            "Train and test transactions overlap."
        )

    if validation_ids & test_ids:
        raise TemporalGraphSplitError(
            "Validation and test transactions overlap."
        )

    if (
        split.train[TIMESTAMP_COLUMN].max()
        >= split.validation[
            TIMESTAMP_COLUMN
        ].min()
    ):
        raise TemporalGraphSplitError(
            "Training timestamps must be strictly earlier "
            "than validation timestamps."
        )

    if (
        split.validation[
            TIMESTAMP_COLUMN
        ].max()
        >= split.test[TIMESTAMP_COLUMN].min()
    ):
        raise TemporalGraphSplitError(
            "Validation timestamps must be strictly earlier "
            "than test timestamps."
        )

    if (
        split.train_end
        != split.train[TIMESTAMP_COLUMN].max()
    ):
        raise TemporalGraphSplitError(
            "Invalid train_end boundary."
        )

    if (
        split.validation_end
        != split.validation[TIMESTAMP_COLUMN].max()
    ):
        raise TemporalGraphSplitError(
            "Invalid validation_end boundary."
        )

    if (
        split.test_end
        != split.test[TIMESTAMP_COLUMN].max()
    ):
        raise TemporalGraphSplitError(
            "Invalid test_end boundary."
        )


def validate_complete_coverage(
    transactions: pd.DataFrame,
    split: TemporalGraphSplit,
) -> None:
    """Verify every input transaction appears exactly once."""

    _validate_input(transactions)
    validate_temporal_split(split)

    expected = set(
        transactions[TRANSACTION_ID_COLUMN]
    )

    actual = (
        set(split.train[TRANSACTION_ID_COLUMN])
        | set(
            split.validation[
                TRANSACTION_ID_COLUMN
            ]
        )
        | set(
            split.test[TRANSACTION_ID_COLUMN]
        )
    )

    if expected != actual:
        missing = expected - actual
        extra = actual - expected

        raise TemporalGraphSplitError(
            "Temporal split does not provide complete "
            "transaction coverage. "
            f"Missing={len(missing)}, Extra={len(extra)}."
        )

    if split.total_size != len(transactions):
        raise TemporalGraphSplitError(
            "Temporal split contains an unexpected "
            "number of transactions."
        )


def validate_split_fractions(
    split: TemporalGraphSplit,
    tolerance: float = 0.01,
) -> None:
    """Validate split sizes are approximately 70/15/15."""

    if tolerance < 0:
        raise TemporalGraphSplitError(
            "tolerance cannot be negative."
        )

    total = split.total_size

    actual = {
        "train": split.train_size / total,
        "validation": split.validation_size / total,
        "test": split.test_size / total,
    }

    expected = {
        "train": TRAIN_FRACTION,
        "validation": VALIDATION_FRACTION,
        "test": TEST_FRACTION,
    }

    for name in expected:
        if abs(actual[name] - expected[name]) > tolerance:
            raise TemporalGraphSplitError(
                f"{name} fraction {actual[name]:.4f} "
                f"is outside tolerance {tolerance:.4f}."
            )


def validate_deterministic_split(
    transactions: pd.DataFrame,
) -> None:
    """Verify repeated splitting produces identical IDs."""

    first = split_transactions(
        transactions
    )

    second = split_transactions(
        transactions
    )

    for first_frame, second_frame in zip(
        (
            first.train,
            first.validation,
            first.test,
        ),
        (
            second.train,
            second.validation,
            second.test,
        ),
    ):
        if not first_frame[
            TRANSACTION_ID_COLUMN
        ].equals(
            second_frame[
                TRANSACTION_ID_COLUMN
            ]
        ):
            raise TemporalGraphSplitError(
                "Temporal split is not deterministic."
            )