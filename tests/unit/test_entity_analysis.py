import pandas as pd

from fraud_intelligence.analysis.entity_analysis import (
    entity_fraud_concentration,
    entity_fraud_summary,
    entity_type_overview,
    high_fraud_rate_entities,
    shared_entity_customer_counts,
    top_fraud_entities,
)


def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [
                "TX001",
                "TX002",
                "TX003",
                "TX004",
                "TX005",
            ],
            "customer_id": [
                "C001",
                "C001",
                "C002",
                "C002",
                "C003",
            ],
            "account_id": [
                "A001",
                "A001",
                "A002",
                "A002",
                "A003",
            ],
            "card_id": [
                "CARD001",
                "CARD001",
                "CARD002",
                "CARD002",
                "CARD003",
            ],
            "device_id": [
                "D001",
                "D001",
                "D001",
                "D002",
                "D003",
            ],
            "ip_id": [
                "IP001",
                "IP001",
                "IP001",
                "IP002",
                "IP003",
            ],
            "merchant_id": [
                "M001",
                "M001",
                "M001",
                "M002",
                "M003",
            ],
            "amount": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
            ],
            "is_fraud": [
                0,
                1,
                1,
                0,
                0,
            ],
        }
    )


def test_entity_fraud_summary():
    transactions = sample_transactions()

    result = entity_fraud_summary(
        transactions,
        "customer_id",
    )

    customer_one = result[
        result["customer_id"] == "C001"
    ].iloc[0]

    assert customer_one[
        "transaction_count"
    ] == 2

    assert customer_one[
        "fraud_count"
    ] == 1

    assert customer_one[
        "fraud_rate_pct"
    ] == 50.0


def test_top_fraud_entities():
    transactions = sample_transactions()

    result = top_fraud_entities(
        transactions,
        "customer_id",
        top_n=2,
    )

    assert len(result) == 2
    assert result.iloc[0]["fraud_count"] >= 1


def test_high_fraud_rate_entities():
    transactions = sample_transactions()

    result = high_fraud_rate_entities(
        transactions,
        "customer_id",
        min_transactions=2,
        top_n=10,
    )

    assert len(result) == 2
    assert (
        result["transaction_count"] >= 2
    ).all()


def test_entity_fraud_concentration():
    transactions = sample_transactions()

    result = entity_fraud_concentration(
        transactions,
        "customer_id",
    )

    assert (
        result["fraud_contribution_pct"]
        .sum()
        == 100.0
    )


def test_shared_entity_customer_counts():
    transactions = sample_transactions()

    result = shared_entity_customer_counts(
        transactions,
        "device_id",
    )

    device_one = result[
        result["device_id"] == "D001"
    ].iloc[0]

    assert device_one[
        "unique_customer_count"
    ] == 2

    assert device_one[
        "fraud_count"
    ] == 2


def test_entity_type_overview():
    transactions = sample_transactions()

    result = entity_type_overview(
        transactions
    )

    assert len(result) == 6

    assert (
        result["total_transactions"] == 5
    ).all()

    assert (
        result["total_fraud"] == 2
    ).all()