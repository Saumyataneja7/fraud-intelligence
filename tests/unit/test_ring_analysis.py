import pandas as pd

from fraud_intelligence.analysis.ring_analysis import (
    build_fraud_customer_graph,
    component_fraud_ring_rate,
    fraud_connected_components,
    fraud_ring_transaction_summary,
    shared_device_fraud_connections,
    shared_ip_fraud_connections,
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
                "C002",
                "C003",
                "C004",
                "C005",
            ],
            "account_id": [
                "A001",
                "A002",
                "A003",
                "A004",
                "A005",
            ],
            "card_id": [
                "CARD001",
                "CARD002",
                "CARD003",
                "CARD004",
                "CARD005",
            ],
            "device_id": [
                "D001",
                "D001",
                "D002",
                "D003",
                "D004",
            ],
            "ip_id": [
                "IP001",
                "IP001",
                "IP002",
                "IP003",
                "IP004",
            ],
            "merchant_id": [
                "M001",
                "M001",
                "M002",
                "M003",
                "M004",
            ],
            "amount": [
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
            ],
            "is_fraud": [
                1,
                1,
                1,
                0,
                1,
            ],
            "fraud_scenario": [
                "FRAUD_RING",
                "FRAUD_RING",
                "SHARED_DEVICE",
                "LEGITIMATE",
                "SHARED_IP",
            ],
        }
    )


def test_fraud_ring_transaction_summary():
    transactions = sample_transactions()

    result = fraud_ring_transaction_summary(
        transactions
    )

    assert result.iloc[0]["transaction_count"] == 2
    assert result.iloc[0]["customer_count"] == 2
    assert result.iloc[0]["device_count"] == 1


def test_build_fraud_customer_graph():
    transactions = sample_transactions()

    graph = build_fraud_customer_graph(
        transactions
    )

    assert graph.number_of_nodes() == 4
    assert graph.has_edge(
        "C001",
        "C002",
    )


def test_fraud_connected_components():
    transactions = sample_transactions()

    result = fraud_connected_components(
        transactions
    )

    assert len(result) >= 1
    assert result.iloc[0][
        "customer_count"
    ] >= 2


def test_shared_device_connections():
    transactions = sample_transactions()

    result = shared_device_fraud_connections(
        transactions
    )

    assert len(result) == 1
    assert result.iloc[0][
        "device_id"
    ] == "D001"

    assert result.iloc[0][
        "customer_count"
    ] == 2


def test_shared_ip_connections():
    transactions = sample_transactions()

    result = shared_ip_fraud_connections(
        transactions
    )

    assert len(result) == 1
    assert result.iloc[0][
        "ip_id"
    ] == "IP001"

    assert result.iloc[0][
        "customer_count"
    ] == 2


def test_component_fraud_ring_rate():
    transactions = sample_transactions()

    result = component_fraud_ring_rate(
        transactions
    )

    assert "fraud_ring_pct" in result.columns
    assert (
        result["fraud_ring_pct"]
        >= 0
    ).all()