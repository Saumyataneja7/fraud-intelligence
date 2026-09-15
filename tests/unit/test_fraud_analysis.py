import pandas as pd

from fraud_intelligence.analysis.fraud_analysis import (
    amount_statistics_by_fraud,
    fraud_class_distribution,
    fraud_rate_by_category,
    fraud_scenario_distribution,
)


def make_test_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [
                "T1",
                "T2",
                "T3",
                "T4",
            ],
            "is_fraud": [
                0,
                0,
                1,
                1,
            ],
            "amount": [
                100.0,
                200.0,
                500.0,
                1000.0,
            ],
            "payment_method": [
                "card",
                "card",
                "wallet",
                "wallet",
            ],
            "fraud_scenario": [
                "LEGITIMATE",
                "LEGITIMATE",
                "ACCOUNT_TAKEOVER",
                "SHARED_IP",
            ],
        }
    )


def test_fraud_class_distribution():
    df = make_test_transactions()

    result = fraud_class_distribution(df)

    fraud = result[
        result["label"] == "FRAUD"
    ].iloc[0]

    assert fraud["transaction_count"] == 2
    assert fraud["percentage"] == 50.0


def test_fraud_rate_by_category():
    df = make_test_transactions()

    result = fraud_rate_by_category(
        df,
        "payment_method",
    )

    wallet = result[
        result["payment_method"] == "wallet"
    ].iloc[0]

    assert wallet["transaction_count"] == 2
    assert wallet["fraud_count"] == 2
    assert wallet["fraud_rate_pct"] == 100.0


def test_amount_statistics_by_fraud():
    df = make_test_transactions()

    result = amount_statistics_by_fraud(df)

    fraud = result[
        result["label"] == "FRAUD"
    ].iloc[0]

    assert fraud["count"] == 2
    assert fraud["median"] == 750.0
    assert fraud["max"] == 1000.0


def test_fraud_scenario_distribution():
    df = make_test_transactions()

    result = fraud_scenario_distribution(df)

    assert len(result) == 2
    assert result["fraud_count"].sum() == 2