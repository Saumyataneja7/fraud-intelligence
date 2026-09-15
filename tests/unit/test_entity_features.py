import pandas as pd
import pytest

from fraud_intelligence.features.entity_features import (
    add_entity_historical_features,
)


def make_transactions():
    return pd.DataFrame(
        {
            "transaction_id": [
                "T3",
                "T1",
                "T4",
                "T2",
                "T5",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 12:00:00+00:00",
                    "2025-01-01 10:00:00+00:00",
                    "2025-01-01 12:00:00+00:00",
                    "2025-01-01 11:00:00+00:00",
                    "2025-01-01 13:00:00+00:00",
                ],
                utc=True,
            ),
            "customer_id": [
                "C1",
                "C1",
                "C2",
                "C1",
                "C2",
            ],
            "account_id": [
                "A1",
                "A1",
                "A2",
                "A1",
                "A2",
            ],
            "card_id": [
                "CARD1",
                "CARD1",
                "CARD2",
                "CARD1",
                "CARD2",
            ],
            "merchant_id": [
                "M2",
                "M1",
                "M3",
                "M1",
                "M3",
            ],
            "device_id": [
                "D1",
                "D1",
                "D2",
                "D1",
                "D2",
            ],
            "ip_id": [
                "IP1",
                "IP1",
                "IP2",
                "IP1",
                "IP2",
            ],
            "amount": [
                100.0,
                20.0,
                500.0,
                50.0,
                700.0,
            ],
        }
    )


def test_first_entity_transaction_has_zero_history():
    result = add_entity_historical_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T1"
    ].iloc[0]

    assert row[
        "account_txn_count_before"
    ] == 0

    assert row[
        "card_txn_count_before"
    ] == 0

    assert row[
        "merchant_txn_count_before"
    ] == 0

    assert row[
        "device_txn_count_before"
    ] == 0

    assert row[
        "ip_txn_count_before"
    ] == 0


def test_entity_history_uses_previous_transactions():
    result = add_entity_historical_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T2"
    ].iloc[0]

    assert row[
        "account_txn_count_before"
    ] == 1

    assert row[
        "card_txn_count_before"
    ] == 1

    assert row[
        "device_txn_count_before"
    ] == 1

    assert row[
        "ip_txn_count_before"
    ] == 1

    assert row[
        "account_amount_sum_before"
    ] == pytest.approx(20.0)

    assert row[
        "card_amount_sum_before"
    ] == pytest.approx(20.0)


def test_same_timestamp_entities_are_excluded():
    transactions = make_transactions()

    # Add another transaction at exactly the same timestamp
    # and with the same device as T3.
    same_timestamp = transactions[
        transactions["transaction_id"] == "T3"
    ].copy()

    same_timestamp["transaction_id"] = "T3_SAME_TIME"
    same_timestamp["amount"] = 250.0

    transactions = pd.concat(
        [
            transactions,
            same_timestamp,
        ],
        ignore_index=True,
    )

    result = add_entity_historical_features(
        transactions
    )

    rows = result[
        (
            result["timestamp"]
            == pd.Timestamp(
                "2025-01-01 12:00:00+00:00"
            )
        )
        & (
            result["device_id"] == "D1"
        )
    ]

    # T3 and T3_SAME_TIME share device D1 and timestamp 12:00.
    #
    # Neither transaction is allowed to see the other.
    #
    # Historical D1 transactions are only:
    # T1 = 20
    # T2 = 50
    #
    # Therefore:
    # count = 2
    # amount = 70
    assert len(rows) == 2

    assert (
        rows["device_txn_count_before"] == 2
    ).all()

    assert (
        rows["device_amount_sum_before"] == 70.0
    ).all()


def test_future_transactions_do_not_leak():
    result = add_entity_historical_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T3"
    ].iloc[0]

    # Device D1 has T1 and T2 before T3.
    # T4 is the same timestamp and T5 is future.
    assert row[
        "device_txn_count_before"
    ] == 2

    assert row[
        "device_amount_sum_before"
    ] == pytest.approx(70.0)


def test_entities_are_isolated():
    result = add_entity_historical_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T5"
    ].iloc[0]

    # T5 uses A2/CARD2/D2/IP2.
    # T4 at 12:00 uses the same entities.
    assert row[
        "account_txn_count_before"
    ] == 1

    assert row[
        "card_txn_count_before"
    ] == 1

    assert row[
        "device_txn_count_before"
    ] == 1

    assert row[
        "ip_txn_count_before"
    ] == 1

    assert row[
        "account_amount_sum_before"
    ] == pytest.approx(500.0)


def test_original_row_order_is_preserved():
    transactions = make_transactions()

    result = add_entity_historical_features(
        transactions
    )

    assert result[
        "transaction_id"
    ].tolist() == transactions[
        "transaction_id"
    ].tolist()


def test_missing_entity_column_is_rejected():
    transactions = make_transactions().drop(
        columns=["device_id"]
    )

    with pytest.raises(ValueError):
        add_entity_historical_features(
            transactions
        )


def test_naive_full_dataset_aggregate_would_be_wrong():
    result = add_entity_historical_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T3"
    ].iloc[0]

    # Device D1 appears in T1 and T2.
    # T3 itself must not be counted.
    # T4 has the same timestamp and T5 is future.
    assert row[
        "device_txn_count_before"
    ] == 2

    assert row[
        "device_txn_count_before"
    ] != 4