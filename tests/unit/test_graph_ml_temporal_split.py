from __future__ import annotations

import pandas as pd
import pytest

from fraud_intelligence.graph_ml.temporal_split import (
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    TemporalGraphSplit,
    TemporalGraphSplitError,
    split_transactions,
    validate_complete_coverage,
    validate_deterministic_split,
    validate_split_fractions,
    validate_temporal_split,
)


def make_transactions(
    n: int = 100,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01",
        periods=n,
        freq="min",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "transaction_id": [
                f"TXN_{i:06d}"
                for i in range(n)
            ],
            "timestamp": timestamps,
            "amount": [float(i + 1) for i in range(n)],
            "is_fraud": [
                int(i % 10 == 0)
                for i in range(n)
            ],
        }
    )


def test_split_returns_temporal_graph_split() -> None:
    transactions = make_transactions()

    result = split_transactions(
        transactions
    )

    assert isinstance(
        result,
        TemporalGraphSplit,
    )


def test_split_sizes_are_70_15_15() -> None:
    transactions = make_transactions(
        1000
    )

    result = split_transactions(
        transactions
    )

    assert result.train_size == 700
    assert result.validation_size == 150
    assert result.test_size == 150


def test_split_fractions_match_configuration() -> None:
    transactions = make_transactions(
        1000
    )

    result = split_transactions(
        transactions
    )

    assert result.train_size / result.total_size == (
        TRAIN_FRACTION
    )

    assert result.validation_size / result.total_size == (
        VALIDATION_FRACTION
    )

    assert result.test_size / result.total_size == (
        TEST_FRACTION
    )


def test_training_is_strictly_before_validation() -> None:
    result = split_transactions(
        make_transactions()
    )

    assert (
        result.train["timestamp"].max()
        < result.validation["timestamp"].min()
    )


def test_validation_is_strictly_before_test() -> None:
    result = split_transactions(
        make_transactions()
    )

    assert (
        result.validation["timestamp"].max()
        < result.test["timestamp"].min()
    )


def test_transaction_ids_do_not_overlap() -> None:
    result = split_transactions(
        make_transactions()
    )

    train_ids = set(
        result.train["transaction_id"]
    )

    validation_ids = set(
        result.validation["transaction_id"]
    )

    test_ids = set(
        result.test["transaction_id"]
    )

    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids


def test_complete_transaction_coverage() -> None:
    transactions = make_transactions()

    result = split_transactions(
        transactions
    )

    validate_complete_coverage(
        transactions,
        result,
    )


def test_temporal_split_validation_passes() -> None:
    result = split_transactions(
        make_transactions()
    )

    validate_temporal_split(result)


def test_split_fraction_validation_passes() -> None:
    result = split_transactions(
        make_transactions(
            1000
        )
    )

    validate_split_fractions(
        result
    )


def test_split_is_deterministic() -> None:
    transactions = make_transactions()

    validate_deterministic_split(
        transactions
    )


def test_input_dataframe_is_not_modified() -> None:
    transactions = make_transactions()

    original = transactions.copy(
        deep=True
    )

    split_transactions(
        transactions
    )

    pd.testing.assert_frame_equal(
        transactions,
        original,
    )


def test_unsorted_input_is_sorted_chronologically() -> None:
    transactions = make_transactions()

    shuffled = transactions.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    result = split_transactions(
        shuffled
    )

    assert result.train[
        "timestamp"
    ].is_monotonic_increasing

    assert result.validation[
        "timestamp"
    ].is_monotonic_increasing

    assert result.test[
        "timestamp"
    ].is_monotonic_increasing


def test_transaction_id_is_used_as_deterministic_tiebreaker() -> None:
    timestamps = pd.date_range(
        "2025-01-01 00:00:00",
        periods=99,
        freq="min",
        tz="UTC",
    )

    transactions = pd.DataFrame(
        {
            "transaction_id": [
                "TXN_B",
                "TXN_A",
            ]
            + [
                f"TXN_{i:03d}"
                for i in range(2, 100)
            ],
            "timestamp": [
                timestamps[0],
                timestamps[0],
            ]
            + list(timestamps[1:]),
        }
    )

    result = split_transactions(transactions)

    ordered_ids = (
        pd.concat(
            [
                result.train,
                result.validation,
                result.test,
            ]
        )["transaction_id"]
        .tolist()
    )

    assert ordered_ids[:2] == [
        "TXN_A",
        "TXN_B",
    ]


def test_missing_timestamp_is_rejected() -> None:
    transactions = make_transactions().drop(
        columns=["timestamp"]
    )

    with pytest.raises(
        TemporalGraphSplitError,
        match="timestamp",
    ):
        split_transactions(
            transactions
        )


def test_missing_transaction_id_is_rejected() -> None:
    transactions = make_transactions().drop(
        columns=["transaction_id"]
    )

    with pytest.raises(
        TemporalGraphSplitError,
        match="transaction_id",
    ):
        split_transactions(
            transactions
        )


def test_duplicate_transaction_ids_are_rejected() -> None:
    transactions = make_transactions()

    transactions.loc[
        1,
        "transaction_id",
    ] = transactions.loc[
        0,
        "transaction_id",
    ]

    with pytest.raises(
        TemporalGraphSplitError,
        match="unique",
    ):
        split_transactions(
            transactions
        )


def test_null_timestamp_is_rejected() -> None:
    transactions = make_transactions()

    transactions.loc[
        0,
        "timestamp",
    ] = pd.NaT

    with pytest.raises(
        TemporalGraphSplitError,
        match="null",
    ):
        split_transactions(
            transactions
        )


def test_non_datetime_timestamp_is_rejected() -> None:
    transactions = make_transactions()

    transactions["timestamp"] = (
        transactions["timestamp"]
        .astype(str)
    )

    with pytest.raises(
        TemporalGraphSplitError,
        match="datetime",
    ):
        split_transactions(
            transactions
        )


def test_empty_dataframe_is_rejected() -> None:
    transactions = pd.DataFrame(
        columns=[
            "transaction_id",
            "timestamp",
        ]
    )

    with pytest.raises(
        TemporalGraphSplitError,
        match="empty",
    ):
        split_transactions(
            transactions
        )


def test_invalid_temporal_split_is_rejected() -> None:
    transactions = make_transactions()

    result = split_transactions(
        transactions
    )

    invalid = TemporalGraphSplit(
        train=result.validation,
        validation=result.train,
        test=result.test,
        train_end=result.train_end,
        validation_end=result.validation_end,
        test_end=result.test_end,
    )

    with pytest.raises(
        TemporalGraphSplitError,
    ):
        validate_temporal_split(
            invalid
        )


def test_incomplete_coverage_is_rejected() -> None:
    transactions = make_transactions()

    result = split_transactions(
        transactions
    )

    incomplete = TemporalGraphSplit(
        train=result.train.iloc[:-1],
        validation=result.validation,
        test=result.test,
        train_end=result.train_end,
        validation_end=result.validation_end,
        test_end=result.test_end,
    )

    with pytest.raises(
        TemporalGraphSplitError,
    ):
        validate_complete_coverage(
            transactions,
            incomplete,
        )