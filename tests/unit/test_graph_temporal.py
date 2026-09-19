from __future__ import annotations

import pandas as pd
import pytest

from fraud_intelligence.graph.temporal import (
    GraphTemporalError,
    filter_edge_table_as_of,
    filter_transactions_as_of,
    is_temporally_safe,
    validate_edge_table_as_of,
    validate_no_future_transactions,
    validate_transaction_temporal_order,
)


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [
                "txn_001",
                "txn_002",
                "txn_003",
                "txn_004",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 09:59:59+00:00",
                    "2025-01-01 10:00:00+00:00",
                    "2025-01-01 10:00:00+00:00",
                    "2025-01-01 10:00:01+00:00",
                ],
                utc=True,
            ),
            "amount": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_filter_excludes_cutoff_and_future_transactions() -> None:
    transactions = _transactions()

    result = filter_transactions_as_of(
        transactions,
        pd.Timestamp("2025-01-01 10:00:00+00:00"),
    )

    assert result.dataframe["transaction_id"].tolist() == [
        "txn_001"
    ]

    assert result.original_edge_count == 4
    assert result.retained_edge_count == 1
    assert result.removed_edge_count == 3


def test_filter_keeps_only_strictly_earlier_transactions() -> None:
    transactions = _transactions()

    result = filter_transactions_as_of(
        transactions,
        pd.Timestamp("2025-01-01 10:00:01+00:00"),
    )

    assert result.dataframe["transaction_id"].tolist() == [
        "txn_001",
        "txn_002",
        "txn_003",
    ]


def test_same_timestamp_is_excluded() -> None:
    transactions = _transactions()

    result = filter_transactions_as_of(
        transactions,
        pd.Timestamp("2025-01-01 10:00:00+00:00"),
    )

    assert "txn_002" not in result.dataframe["transaction_id"].tolist()
    assert "txn_003" not in result.dataframe["transaction_id"].tolist()


def test_future_transactions_are_rejected() -> None:
    transactions = _transactions()

    with pytest.raises(GraphTemporalError, match="Temporal leakage"):
        validate_no_future_transactions(
            transactions,
            pd.Timestamp("2025-01-01 10:00:00+00:00"),
        )


def test_temporally_safe_dataset_passes() -> None:
    transactions = _transactions().iloc[[0]].copy()

    assert is_temporally_safe(
        transactions,
        pd.Timestamp("2025-01-01 10:00:00+00:00"),
    )


def test_temporally_unsafe_dataset_fails() -> None:
    transactions = _transactions()

    assert not is_temporally_safe(
        transactions,
        pd.Timestamp("2025-01-01 10:00:00+00:00"),
    )


def test_naive_cutoff_is_rejected() -> None:
    transactions = _transactions()

    with pytest.raises(
        GraphTemporalError,
        match="timezone-aware",
    ):
        filter_transactions_as_of(
            transactions,
            pd.Timestamp("2025-01-01 10:00:00"),
        )


def test_missing_timestamp_column_is_rejected() -> None:
    transactions = _transactions().drop(columns=["timestamp"])

    with pytest.raises(
        GraphTemporalError,
        match="Missing required timestamp column",
    ):
        filter_transactions_as_of(
            transactions,
            pd.Timestamp("2025-01-01 10:00:00+00:00"),
        )


def test_invalid_timestamp_is_rejected() -> None:
    transactions = _transactions()
    transactions.loc[0, "timestamp"] = "not-a-timestamp"

    with pytest.raises(
        GraphTemporalError,
        match="invalid or null timestamps",
    ):
        filter_transactions_as_of(
            transactions,
            pd.Timestamp("2025-01-01 10:00:00+00:00"),
        )


def test_transaction_temporal_order_validation() -> None:
    validate_transaction_temporal_order(_transactions())


def test_edge_table_temporal_filter() -> None:
    edges = pd.DataFrame(
        {
            "source_node_id": [0, 0, 1],
            "target_node_id": [1, 2, 2],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 09:00:00+00:00",
                    "2025-01-01 10:00:00+00:00",
                    "2025-01-01 11:00:00+00:00",
                ],
                utc=True,
            ),
        }
    )

    result = filter_edge_table_as_of(
        edges,
        pd.Timestamp("2025-01-01 11:00:00+00:00"),
    )

    assert len(result.dataframe) == 2
    assert result.retained_edge_count == 2


def test_edge_table_validation_passes_for_historical_edges() -> None:
    edges = pd.DataFrame(
        {
            "source_node_id": [0],
            "target_node_id": [1],
            "timestamp": pd.to_datetime(
                ["2025-01-01 09:00:00+00:00"],
                utc=True,
            ),
        }
    )

    validate_edge_table_as_of(
        edges,
        pd.Timestamp("2025-01-01 10:00:00+00:00"),
    )


def test_edge_table_validation_rejects_future_edges() -> None:
    edges = pd.DataFrame(
        {
            "source_node_id": [0],
            "target_node_id": [1],
            "timestamp": pd.to_datetime(
                ["2025-01-01 11:00:00+00:00"],
                utc=True,
            ),
        }
    )

    with pytest.raises(GraphTemporalError, match="Temporal leakage"):
        validate_edge_table_as_of(
            edges,
            pd.Timestamp("2025-01-01 10:00:00+00:00"),
        )


def test_empty_transaction_table_is_temporally_safe() -> None:
    transactions = _transactions().iloc[0:0].copy()

    result = filter_transactions_as_of(
        transactions,
        pd.Timestamp("2025-01-01 10:00:00+00:00"),
    )

    assert result.dataframe.empty
    assert result.retained_edge_count == 0


def test_original_dataframe_is_not_modified() -> None:
    transactions = _transactions()
    original = transactions.copy(deep=True)

    filter_transactions_as_of(
        transactions,
        pd.Timestamp("2025-01-01 10:00:00+00:00"),
    )

    pd.testing.assert_frame_equal(transactions, original)