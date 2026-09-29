from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_api_health_and_metadata_flow() -> None:
    health_response = client.get("/health")

    assert health_response.status_code == 200

    health_payload = health_response.json()

    assert health_payload["status"] == "ok"

    metadata_response = client.get("/metadata")

    assert metadata_response.status_code == 200

    metadata_payload = metadata_response.json()

    assert "project" in metadata_payload
    assert "version" in metadata_payload


def test_model_metrics_endpoint_is_integrated() -> None:
    response = client.get("/model-metrics")

    assert response.status_code == 200

    payload = response.json()

    assert payload["feature_count"] == 53
    assert len(payload["results"]) == 6


def test_transaction_investigation_unknown_id_returns_404() -> None:
    response = client.get(
        "/transaction/transaction-that-does-not-exist"
    )

    assert response.status_code == 404


def test_customer_investigation_unknown_id_returns_404() -> None:
    response = client.get(
        "/customer/customer-that-does-not-exist"
    )

    assert response.status_code == 404


def test_fraud_ring_unknown_id_returns_404() -> None:
    response = client.get(
        "/fraud-ring/ring-that-does-not-exist"
    )

    assert response.status_code == 404


def test_graph_invalid_node_returns_404() -> None:
    response = client.get(
        "/graph/999999999?node_type=customer"
    )

    assert response.status_code == 404


def test_explanation_unknown_transaction_returns_404() -> None:
    response = client.get(
        "/explanation/transaction-that-does-not-exist"
    )

    assert response.status_code == 404