import pandas as pd
import pytest

from fraud_intelligence.features.pipeline import (
    build_feature_dataset,
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
                    "2025-01-01 11:00:00+00:00",
                    "2025-01-01 12:00:00+00:00",
                    "2025-01-01 12:00:00+00:00",
                    "2025-01-01 13:00:00+00:00",
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
            "account_id": [
                "A1",
                "A1",
                "A1",
                "A1",
                "A2",
            ],
            "card_id": [
                "CARD1",
                "CARD1",
                "CARD1",
                "CARD2",
                "CARD3",
            ],
            "merchant_id": [
                "M1",
                "M1",
                "M2",
                "M2",
                "M3",
            ],
            "device_id": [
                "D1",
                "D1",
                "D2",
                "D2",
                "D3",
            ],
            "ip_id": [
                "IP1",
                "IP1",
                "IP2",
                "IP2",
                "IP3",
            ],
            "amount": [
                20.0,
                40.0,
                100.0,
                200.0,
                500.0,
            ],
            "currency": [
                "USD",
                "USD",
                "USD",
                "USD",
                "USD",
            ],
            "payment_method": [
                "card",
                "card",
                "wallet",
                "wallet",
                "bank_transfer",
            ],
            "transaction_type": [
                "purchase",
                "purchase",
                "transfer",
                "transfer",
                "withdrawal",
            ],
            "is_fraud": [
                0,
                0,
                1,
                1,
                0,
            ],
            "fraud_scenario": [
                None,
                None,
                "SHARED_DEVICE",
                "SHARED_DEVICE",
                None,
            ],
        }
    )


def test_complete_feature_pipeline_adds_all_feature_families():
    transactions = make_transactions()

    result = build_feature_dataset(
        transactions
    )

    expected_features = [
        # 4.1
        "log_amount",
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "is_weekend",
        "is_night",
        "is_card_payment",
        "is_bank_transfer",
        "is_wallet",
        "is_online",
        "is_purchase",
        "is_transfer",
        "is_withdrawal",
        "is_payment",

        # 4.2
        "customer_txn_count_before",
        "customer_amount_sum_before",
        "customer_amount_mean_before",
        "customer_amount_median_before",
        "customer_amount_max_before",
        "customer_unique_merchants_before",
        "customer_unique_devices_before",
        "customer_unique_ips_before",

        # 4.3
        "customer_txn_count_1m",
        "customer_txn_count_5m",
        "customer_txn_count_15m",
        "customer_amount_sum_1m",
        "customer_amount_sum_5m",
        "customer_amount_sum_15m",
        "customer_unique_merchants_5m",
        "customer_unique_devices_5m",
        "customer_unique_ips_5m",

        # 4.4
        "account_txn_count_before",
        "card_txn_count_before",
        "merchant_txn_count_before",
        "device_txn_count_before",
        "ip_txn_count_before",
        "account_amount_sum_before",
        "card_amount_sum_before",
        "merchant_amount_sum_before",
        "device_amount_sum_before",
        "ip_amount_sum_before",

        # 4.5
        "is_new_device",
        "is_new_ip",
        "is_new_merchant",
        "is_new_card",

        # 4.6
        "amount_vs_customer_mean",
        "amount_vs_customer_median",
        "amount_vs_customer_max",
        "amount_deviation_from_customer_mean",
        "is_new_payment_method",
        "is_new_transaction_type",
    ]

    for feature in expected_features:
        assert feature in result.columns


def test_pipeline_preserves_row_count():
    transactions = make_transactions()

    result = build_feature_dataset(
        transactions
    )

    assert len(result) == len(transactions)


def test_pipeline_preserves_original_row_order():
    transactions = make_transactions()

    result = build_feature_dataset(
        transactions
    )

    assert result[
        "transaction_id"
    ].tolist() == transactions[
        "transaction_id"
    ].tolist()


def test_pipeline_does_not_modify_input():
    transactions = make_transactions()
    original = transactions.copy(
        deep=True
    )

    build_feature_dataset(
        transactions
    )

    pd.testing.assert_frame_equal(
        transactions,
        original,
    )


def test_pipeline_keeps_targets_but_does_not_create_target_features():
    transactions = make_transactions()

    result = build_feature_dataset(
        transactions
    )

    # Labels remain available for downstream splitting/evaluation.
    assert "is_fraud" in result.columns
    assert "fraud_scenario" in result.columns

    # Labels must not appear as engineered feature columns.
    engineered_columns = [
        column
        for column in result.columns
        if column not in transactions.columns
    ]

    assert "is_fraud" not in engineered_columns
    assert "fraud_scenario" not in engineered_columns


def test_pipeline_is_leakage_safe_for_same_timestamp():
    transactions = make_transactions()

    result = build_feature_dataset(
        transactions
    )

    rows = result[
        result["timestamp"]
        == pd.Timestamp(
            "2025-01-01 12:00:00+00:00"
        )
    ]

    # T3 and T4 belong to C1 at the same timestamp.
    #
    # Customer historical amount baseline:
    # T1 = 20
    # T2 = 40
    #
    # Mean = 30
    #
    # Neither T3 nor T4 may enter that baseline.

    assert rows["customer_amount_mean_before"].tolist() == pytest.approx(
        [30.0, 30.0]
    )

    # Both same-timestamp transactions are new
    # to their shared D2/IP2/M2 entities.
    assert (
        rows["is_new_device"] == 1
    ).all()

    assert (
        rows["is_new_ip"] == 1
    ).all()

    assert (
        rows["is_new_merchant"] == 1
    ).all()