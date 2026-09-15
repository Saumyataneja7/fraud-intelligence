import pandas as pd
import pytest

from fraud_intelligence.features.velocity_features import (
    add_velocity_features,
)


def make_transactions():
    return pd.DataFrame(
        {
            "transaction_id": [
                "T1",
                "T2",
                "T3",
                "T4",
                "T5",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 10:00:00+00:00",
                    "2025-01-01 10:00:30+00:00",
                    "2025-01-01 10:01:00+00:00",
                    "2025-01-01 10:05:00+00:00",
                    "2025-01-01 10:10:00+00:00",
                ],
                utc=True,
            ),
            "customer_id": [
                "C1",
                "C1",
                "C1",
                "C1",
                "C2",
            ],
            "amount": [
                10.0,
                20.0,
                30.0,
                40.0,
                100.0,
            ],
            "merchant_id": [
                "M1",
                "M1",
                "M2",
                "M3",
                "M4",
            ],
            "device_id": [
                "D1",
                "D1",
                "D2",
                "D3",
                "D4",
            ],
            "ip_id": [
                "IP1",
                "IP1",
                "IP2",
                "IP3",
                "IP4",
            ],
        }
    )


def test_first_transaction_has_zero_velocity():
    result = add_velocity_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T1"
    ].iloc[0]

    assert row["customer_txn_count_1m"] == 0
    assert row["customer_txn_count_5m"] == 0
    assert row["customer_txn_count_15m"] == 0

    assert row["customer_amount_sum_1m"] == 0
    assert row["customer_amount_sum_5m"] == 0
    assert row["customer_amount_sum_15m"] == 0


def test_one_minute_window_excludes_current_transaction():
    result = add_velocity_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T3"
    ].iloc[0]

    # T1 at 10:00:00 and T2 at 10:00:30
    # are both strictly before T3 at 10:01:00.
    assert row["customer_txn_count_1m"] == 2

    assert row[
        "customer_amount_sum_1m"
    ] == pytest.approx(30.0)


def test_same_timestamp_transactions_are_excluded():
    transactions = make_transactions()

    transactions.loc[
        transactions["transaction_id"] == "T2",
        "timestamp",
    ] = pd.Timestamp(
        "2025-01-01 10:01:00+00:00"
    )

    result = add_velocity_features(
        transactions
    )

    row = result[
        result["transaction_id"] == "T3"
    ].iloc[0]

    # T2 now has the same timestamp as T3
    # and therefore must not contribute.
    assert row["customer_txn_count_1m"] == 1
    assert row["customer_amount_sum_1m"] == pytest.approx(
        10.0
    )


def test_five_minute_window():
    result = add_velocity_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T4"
    ].iloc[0]

    # T4 = 10:05:00.
    # T1, T2 and T3 are inside the 5-minute window.
    assert row[
        "customer_txn_count_5m"
    ] == 3

    assert row[
        "customer_amount_sum_5m"
    ] == pytest.approx(60.0)


def test_fifteen_minute_window_does_not_mix_customers():
    result = add_velocity_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T5"
    ].iloc[0]

    assert row[
        "customer_txn_count_15m"
    ] == 0

    assert row[
        "customer_amount_sum_15m"
    ] == 0


def test_unique_entity_velocity():
    result = add_velocity_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T3"
    ].iloc[0]

    # Previous transactions:
    # T1 -> M1/D1/IP1
    # T2 -> M1/D1/IP1
    assert row[
        "customer_unique_merchants_5m"
    ] == 1

    assert row[
        "customer_unique_devices_5m"
    ] == 1

    assert row[
        "customer_unique_ips_5m"
    ] == 1


def test_original_row_order_is_preserved():
    transactions = make_transactions()

    result = add_velocity_features(
        transactions
    )

    assert result[
        "transaction_id"
    ].tolist() == transactions[
        "transaction_id"
    ].tolist()


def test_missing_column_is_rejected():
    transactions = make_transactions().drop(
        columns=["device_id"]
    )

    with pytest.raises(ValueError):
        add_velocity_features(
            transactions
        )