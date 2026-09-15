import pandas as pd
import pytest

from fraud_intelligence.features.historical_features import (
    add_customer_historical_features,
)


def make_transactions():
    return pd.DataFrame(
        {
            "transaction_id": [
                "T3",
                "T1",
                "T4",
                "T2",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 12:00:00+00:00",
                    "2025-01-01 10:00:00+00:00",
                    "2025-01-01 12:00:00+00:00",
                    "2025-01-01 11:00:00+00:00",
                ],
                utc=True,
            ),
            "customer_id": [
                "C1",
                "C1",
                "C2",
                "C1",
            ],
            "amount": [
                100.0,
                20.0,
                500.0,
                50.0,
            ],
            "merchant_id": [
                "M2",
                "M1",
                "M3",
                "M1",
            ],
            "device_id": [
                "D2",
                "D1",
                "D3",
                "D1",
            ],
            "ip_id": [
                "IP2",
                "IP1",
                "IP3",
                "IP1",
            ],
        }
    )


def test_first_customer_transaction_has_zero_history():
    result = add_customer_historical_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T1"
    ].iloc[0]

    assert row[
        "customer_txn_count_before"
    ] == 0

    assert row[
        "customer_amount_sum_before"
    ] == 0.0

    assert row[
        "customer_amount_mean_before"
    ] == 0.0

    assert row[
        "customer_amount_median_before"
    ] == 0.0

    assert row[
        "customer_amount_max_before"
    ] == 0.0


def test_historical_features_use_only_previous_transactions():
    result = add_customer_historical_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T2"
    ].iloc[0]

    assert row[
        "customer_txn_count_before"
    ] == 1

    assert row[
        "customer_amount_sum_before"
    ] == pytest.approx(20.0)

    assert row[
        "customer_amount_mean_before"
    ] == pytest.approx(20.0)

    assert row[
        "customer_amount_median_before"
    ] == pytest.approx(20.0)

    assert row[
        "customer_amount_max_before"
    ] == pytest.approx(20.0)


def test_same_timestamp_transactions_are_not_history():
    result = add_customer_historical_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T3"
    ].iloc[0]

    assert row[
        "customer_txn_count_before"
    ] == 2

    assert row[
        "customer_amount_sum_before"
    ] == pytest.approx(70.0)


def test_future_transactions_do_not_leak():
    result = add_customer_historical_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T3"
    ].iloc[0]

    assert row[
        "customer_amount_sum_before"
    ] != pytest.approx(1070.0)

    assert row[
        "customer_amount_sum_before"
    ] == pytest.approx(70.0)


def test_unique_entity_history():
    result = add_customer_historical_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T3"
    ].iloc[0]

    assert row[
        "customer_unique_merchants_before"
    ] == 1

    assert row[
        "customer_unique_devices_before"
    ] == 1

    assert row[
        "customer_unique_ips_before"
    ] == 1


def test_customer_history_is_preserved_in_original_row_order():
    transactions = make_transactions()

    result = add_customer_historical_features(
        transactions
    )

    assert result[
        "transaction_id"
    ].tolist() == transactions[
        "transaction_id"
    ].tolist()


def test_missing_required_column_is_rejected():
    transactions = make_transactions().drop(
        columns=["merchant_id"]
    )

    with pytest.raises(ValueError):
        add_customer_historical_features(
            transactions
        )