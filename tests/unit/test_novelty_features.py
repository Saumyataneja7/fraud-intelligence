import pandas as pd
import pytest

from fraud_intelligence.features.novelty_features import (
    add_novelty_features,
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
            "account_id": [
                "A1",
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
                "CARD2",
                "CARD3",
            ],
            "merchant_id": [
                "M1",
                "M1",
                "M2",
                "M2",
                "M3",
                "M1",
            ],
            "device_id": [
                "D1",
                "D1",
                "D2",
                "D2",
                "D3",
                "D1",
            ],
            "ip_id": [
                "IP1",
                "IP1",
                "IP2",
                "IP2",
                "IP3",
                "IP1",
            ],
            "amount": [
                20.0,
                50.0,
                100.0,
                150.0,
                200.0,
                300.0,
            ],
        }
    )


def test_first_customer_transaction_has_all_entities_new():
    result = add_novelty_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T1"
    ].iloc[0]

    assert row["is_new_device"] == 1
    assert row["is_new_ip"] == 1
    assert row["is_new_merchant"] == 1
    assert row["is_new_card"] == 1


def test_reused_entities_are_not_new():
    result = add_novelty_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T2"
    ].iloc[0]

    assert row["is_new_device"] == 0
    assert row["is_new_ip"] == 0
    assert row["is_new_merchant"] == 0
    assert row["is_new_card"] == 0


def test_new_entities_are_detected():
    result = add_novelty_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T3"
    ].iloc[0]

    assert row["is_new_device"] == 1
    assert row["is_new_ip"] == 1
    assert row["is_new_merchant"] == 1
    assert row["is_new_card"] == 0


def test_same_timestamp_transactions_do_not_see_each_other():
    result = add_novelty_features(
        make_transactions()
    )

    rows = result[
        (
            result["timestamp"]
            == pd.Timestamp(
                "2025-01-01 12:00:00+00:00"
            )
        )
    ]

    # T3 and T4 both belong to C1 and use D2/IP2/M2.
    #
    # Neither transaction should make the other transaction
    # non-new.
    assert len(rows) == 2

    assert (
        rows["is_new_device"] == 1
    ).all()

    assert (
        rows["is_new_ip"] == 1
    ).all()

    assert (
        rows["is_new_merchant"] == 1
    ).all()

    # T3 uses CARD1, which C1 already used.
    # T4 uses CARD2, which is new.
    card_rows = rows.sort_values(
        "transaction_id"
    )

    assert card_rows.iloc[0]["is_new_card"] == 0
    assert card_rows.iloc[1]["is_new_card"] == 1


def test_future_transactions_do_not_leak():
    result = add_novelty_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T3"
    ].iloc[0]

    # T5 uses D3/IP3/M3, but those are future observations.
    # They must not affect T3.
    assert row["is_new_device"] == 1
    assert row["is_new_ip"] == 1
    assert row["is_new_merchant"] == 1


def test_customer_histories_are_isolated():
    result = add_novelty_features(
        make_transactions()
    )

    row = result[
        result["transaction_id"] == "T6"
    ].iloc[0]

    # C2 has never used CARD3, D1, IP1 or M1.
    # Even though C1 has used M1/D1/IP1 extensively,
    # C1's history must not affect C2.
    assert row["is_new_device"] == 1
    assert row["is_new_ip"] == 1
    assert row["is_new_merchant"] == 1
    assert row["is_new_card"] == 1


def test_original_row_order_is_preserved():
    transactions = make_transactions()

    result = add_novelty_features(
        transactions
    )

    assert result[
        "transaction_id"
    ].tolist() == transactions[
        "transaction_id"
    ].tolist()


def test_missing_required_column_is_rejected():
    transactions = make_transactions().drop(
        columns=["device_id"]
    )

    with pytest.raises(ValueError):
        add_novelty_features(
            transactions
        )