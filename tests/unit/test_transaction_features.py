import pandas as pd
import pytest

from fraud_intelligence.features.transaction_features import (
    add_transaction_features,
)


def test_transaction_features_are_created():
    transactions = pd.DataFrame(
        {
            "amount": [100.0, 250.0],
            "payment_method": [
                "card",
                "wallet",
            ],
            "transaction_type": [
                "purchase",
                "transfer",
            ],
        }
    )

    result = add_transaction_features(
        transactions
    )

    expected_columns = {
        "log_amount",
        "is_card_payment",
        "is_bank_transfer",
        "is_wallet",
        "is_online",
        "is_purchase",
        "is_transfer",
        "is_withdrawal",
        "is_payment",
    }

    assert expected_columns.issubset(
        result.columns
    )


def test_log_amount_is_correct():
    transactions = pd.DataFrame(
        {
            "amount": [1.0, 100.0],
            "payment_method": [
                "card",
                "wallet",
            ],
            "transaction_type": [
                "purchase",
                "transfer",
            ],
        }
    )

    result = add_transaction_features(
        transactions
    )

    assert result.loc[0, "log_amount"] == pytest.approx(
        0.6931471805599453
    )

    assert result.loc[1, "log_amount"] == pytest.approx(
        4.61512051684126
    )


def test_transaction_indicators():
    transactions = pd.DataFrame(
        {
            "amount": [100.0],
            "payment_method": ["card"],
            "transaction_type": ["purchase"],
        }
    )

    result = add_transaction_features(
        transactions
    )

    assert result.loc[0, "is_card_payment"] == 1
    assert result.loc[0, "is_wallet"] == 0
    assert result.loc[0, "is_purchase"] == 1
    assert result.loc[0, "is_transfer"] == 0


def test_negative_amount_is_rejected():
    transactions = pd.DataFrame(
        {
            "amount": [-10.0],
            "payment_method": ["card"],
            "transaction_type": ["purchase"],
        }
    )

    with pytest.raises(ValueError):
        add_transaction_features(
            transactions
        )