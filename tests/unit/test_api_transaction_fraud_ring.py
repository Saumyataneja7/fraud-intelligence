from fastapi.testclient import TestClient

from api.main import app
from api.services.transaction_fraud_ring import (
    TransactionFraudRingServiceError,
)


client = TestClient(app)


def test_transaction_fraud_ring_unknown_transaction_returns_404():
    response = client.get(
        "/transaction/TXN_DOES_NOT_EXIST/fraud-ring"
    )

    assert response.status_code == 404


def test_transaction_fraud_ring_service_error_returns_503(
    monkeypatch,
):
    from api import main

    class FailingService:
        def investigate(self, transaction_id):
            raise TransactionFraudRingServiceError(
                "Service unavailable."
            )

    monkeypatch.setattr(
        main,
        "transaction_fraud_ring_service",
        FailingService(),
    )

    response = client.get(
        "/transaction/TXN_000000000/fraud-ring"
    )

    assert response.status_code == 503


def test_transaction_fraud_ring_route_is_registered():
    routes = {
        route.path
        for route in app.routes
    }

    assert (
        "/transaction/{transaction_id}/fraud-ring"
        in routes
    )