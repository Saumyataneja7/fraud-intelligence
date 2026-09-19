from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class GraphTemporalError(ValueError):
    """Raised when temporal graph controls are violated."""


TRANSACTION_EDGE_TYPES = frozenset(
    {
        "customer_transaction",
        "account_transaction",
        "card_transaction",
        "transaction_merchant",
        "transaction_device",
        "transaction_ip",
    }
)


TRANSACTION_TIMESTAMP_COLUMN = "timestamp"


@dataclass(frozen=True)
class TemporalGraphResult:
    """Result of applying a point-in-time graph cutoff."""

    dataframe: pd.DataFrame
    cutoff_timestamp: pd.Timestamp
    original_edge_count: int
    retained_edge_count: int

    @property
    def removed_edge_count(self) -> int:
        return self.original_edge_count - self.retained_edge_count


def _validate_cutoff_timestamp(cutoff_timestamp: pd.Timestamp) -> pd.Timestamp:
    """Validate and normalize a graph cutoff timestamp."""

    cutoff = pd.Timestamp(cutoff_timestamp)

    if cutoff.tzinfo is None:
        raise GraphTemporalError(
            "cutoff_timestamp must be timezone-aware."
        )

    return cutoff


def _validate_timestamp_column(
    dataframe: pd.DataFrame,
    timestamp_column: str,
) -> None:
    if timestamp_column not in dataframe.columns:
        raise GraphTemporalError(
            f"Missing required timestamp column: {timestamp_column}"
        )

    timestamps = pd.to_datetime(
        dataframe[timestamp_column],
        utc=True,
        errors="coerce",
    )

    if timestamps.isna().any():
        raise GraphTemporalError(
            f"{timestamp_column} contains invalid or null timestamps."
        )


def filter_transactions_as_of(
    transactions: pd.DataFrame,
    cutoff_timestamp: pd.Timestamp,
    *,
    timestamp_column: str = TRANSACTION_TIMESTAMP_COLUMN,
) -> TemporalGraphResult:
    """
    Return transactions strictly earlier than the cutoff timestamp.

    Strict inequality is intentional.

    Transactions occurring at exactly the cutoff timestamp are excluded
    because no explicit event ordering exists within the dataset.
    """

    cutoff = _validate_cutoff_timestamp(cutoff_timestamp)

    _validate_timestamp_column(transactions, timestamp_column)

    timestamps = pd.to_datetime(
        transactions[timestamp_column],
        utc=True,
    )

    mask = timestamps < cutoff

    filtered = transactions.loc[mask].copy()

    return TemporalGraphResult(
        dataframe=filtered,
        cutoff_timestamp=cutoff,
        original_edge_count=len(transactions),
        retained_edge_count=len(filtered),
    )


def validate_no_future_transactions(
    transactions: pd.DataFrame,
    cutoff_timestamp: pd.Timestamp,
    *,
    timestamp_column: str = TRANSACTION_TIMESTAMP_COLUMN,
) -> None:
    """
    Validate that a transaction set contains no records at or after
    the supplied point-in-time cutoff.
    """

    cutoff = _validate_cutoff_timestamp(cutoff_timestamp)

    _validate_timestamp_column(transactions, timestamp_column)

    timestamps = pd.to_datetime(
        transactions[timestamp_column],
        utc=True,
    )

    future_or_same = timestamps >= cutoff

    if future_or_same.any():
        offending_count = int(future_or_same.sum())

        raise GraphTemporalError(
            "Temporal leakage detected: "
            f"{offending_count} transaction(s) occur at or after "
            f"cutoff {cutoff.isoformat()}."
        )


def validate_transaction_temporal_order(
    transactions: pd.DataFrame,
    *,
    timestamp_column: str = TRANSACTION_TIMESTAMP_COLUMN,
) -> None:
    """
    Validate that transaction timestamps are timezone-aware and valid.

    This does not require the dataframe to be sorted.
    """

    _validate_timestamp_column(transactions, timestamp_column)

    timestamps = pd.to_datetime(
        transactions[timestamp_column],
        utc=True,
    )

    if timestamps.dt.tz is None:
        raise GraphTemporalError(
            f"{timestamp_column} must be timezone-aware."
        )


def filter_edge_table_as_of(
    edge_table: pd.DataFrame,
    cutoff_timestamp: pd.Timestamp,
    *,
    timestamp_column: str = TRANSACTION_TIMESTAMP_COLUMN,
) -> TemporalGraphResult:
    """
    Apply a point-in-time filter to an edge table that contains timestamps.

    Only edges whose event timestamp is strictly earlier than the cutoff
    are retained.
    """

    cutoff = _validate_cutoff_timestamp(cutoff_timestamp)

    _validate_timestamp_column(edge_table, timestamp_column)

    timestamps = pd.to_datetime(
        edge_table[timestamp_column],
        utc=True,
    )

    mask = timestamps < cutoff
    filtered = edge_table.loc[mask].copy()

    return TemporalGraphResult(
        dataframe=filtered,
        cutoff_timestamp=cutoff,
        original_edge_count=len(edge_table),
        retained_edge_count=len(filtered),
    )


def validate_edge_table_as_of(
    edge_table: pd.DataFrame,
    cutoff_timestamp: pd.Timestamp,
    *,
    timestamp_column: str = TRANSACTION_TIMESTAMP_COLUMN,
) -> None:
    """
    Validate that a timestamped edge table contains no future or
    same-time events relative to the cutoff.
    """

    validate_no_future_transactions(
        edge_table,
        cutoff_timestamp,
        timestamp_column=timestamp_column,
    )


def is_temporally_safe(
    transactions: pd.DataFrame,
    cutoff_timestamp: pd.Timestamp,
    *,
    timestamp_column: str = TRANSACTION_TIMESTAMP_COLUMN,
) -> bool:
    """
    Return True only when all transactions are strictly before cutoff.
    """

    try:
        validate_no_future_transactions(
            transactions,
            cutoff_timestamp,
            timestamp_column=timestamp_column,
        )
    except GraphTemporalError:
        return False

    return True