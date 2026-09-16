import pandas as pd
import pytest

from fraud_intelligence.features.behavioral_features import (
    add_behavioral_deviation_features,
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
                "T6",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 10:00:00+00:00",
                    "2025-01-01 11:00:00+00:00",
                    "2025-01-01 12:00:00+00:00",
                    "2025-01-01 12:00:00+00:00",
                    "2025-01-01 13:00:00+00:00",
                    "2025-01-01 14:00:00+00:00",
                ],
                utc=True,
            ),
            "customer_id": [
                "C1",
                "C1",
                "C1",
                "C1",
                "C1",
                "C2",
            ],
            "amount": [
                20.0,
                40.0,
                100.0,
                200.0,
                60.0,
                500.0,
            ],
            "payment_method": [
                "card",
                "card",
                "wallet",
                "wallet",
                "card",
                "bank_transfer",
            ],
            "transaction_type": [
                "purchase",
                "purchase",
                "transfer",
                "transfer",
                "purchase",
                "withdrawal",
            ],
        }
    )


def test_first_transaction_has_no_historical_baseline():
    result = add_behavioral_deviation_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T1"
    ].iloc[0]

    assert row["amount_vs_customer_mean"] == 0
    assert row["amount_vs_customer_median"] == 0
    assert row["amount_vs_customer_max"] == 0
    assert row[
        "amount_deviation_from_customer_mean"
    ] == 0

    assert row["is_new_payment_method"] == 1
    assert row["is_new_transaction_type"] == 1


def test_amount_deviation_uses_previous_customer_history():
    result = add_behavioral_deviation_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T3"
    ].iloc[0]

    # Historical amounts for C1:
    # 20, 40
    #
    # Mean = 30
    # Median = 30
    # Max = 40
    #
    # Current amount = 100

    assert row[
        "amount_vs_customer_mean"
    ] == pytest.approx(100 / 30)

    assert row[
        "amount_vs_customer_median"
    ] == pytest.approx(100 / 30)

    assert row[
        "amount_vs_customer_max"
    ] == pytest.approx(100 / 40)

    assert row[
        "amount_deviation_from_customer_mean"
    ] == pytest.approx((100 - 30) / 30)


def test_reused_payment_method_is_not_new():
    result = add_behavioral_deviation_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T3"
    ].iloc[0]

    # C1 previously used card but never wallet.
    assert row["is_new_payment_method"] == 1


def test_reused_transaction_type_is_not_new():
    result = add_behavioral_deviation_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T5"
    ].iloc[0]

    # C1 previously used purchase.
    assert row["is_new_transaction_type"] == 0


def test_same_timestamp_transactions_do_not_see_each_other():
    result = add_behavioral_deviation_features(
        make_transactions()
    )

    rows = result[
        (
            result["timestamp"]
            == pd.Timestamp(
                "2025-01-01 12:00:00+00:00"
            )
        )
    ].sort_values(
        "transaction_id"
    )

    # T3 and T4 must both use only:
    # T1 = 20
    # T2 = 40
    #
    # Historical mean = 30
    #
    # Neither 100 nor 200 should enter the baseline.

    assert len(rows) == 2

    values = rows[
        "amount_vs_customer_mean"
    ].tolist()

    assert values[0] == pytest.approx(
        100 / 30
    )

    assert values[1] == pytest.approx(
        200 / 30
    )


def test_future_transactions_do_not_leak():
    result = add_behavioral_deviation_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T3"
    ].iloc[0]

    # T5 occurs later and has amount 60.
    # It must not affect T3's historical baseline.

    assert row[
        "amount_vs_customer_mean"
    ] == pytest.approx(100 / 30)


def test_customer_histories_are_isolated():
    result = add_behavioral_deviation_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T6"
    ].iloc[0]

    # C2 has no prior transactions.
    assert row["amount_vs_customer_mean"] == 0
    assert row["amount_vs_customer_median"] == 0
    assert row["amount_vs_customer_max"] == 0
    assert row[
        "amount_deviation_from_customer_mean"
    ] == 0

    assert row["is_new_payment_method"] == 1
    assert row["is_new_transaction_type"] == 1


def test_original_row_order_is_preserved():
    transactions = make_transactions()

    result = add_behavioral_deviation_features(
        transactions
    )

    assert result[
        "transaction_id"
    ].tolist() == transactions[
        "transaction_id"
    ].tolist()


def test_missing_required_column_is_rejected():
    transactions = make_transactions().drop(
        columns=["payment_method"]
    )

    with pytest.raises(ValueError):
        add_behavioral_deviation_features(
            transactions
        )