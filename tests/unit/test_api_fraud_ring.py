from fastapi.testclient import TestClient

from api.main import app
from api.services.fraud_ring import (
    FraudRingInvestigationServiceError,
)


client = TestClient(app)


def test_unknown_fraud_ring_returns_404():
    response = client.get(
        "/fraud-ring/ring-does-not-exist"
    )

    assert response.status_code == 404


def test_fraud_ring_service_error_returns_503(monkeypatch):
    from api import main

    def failing_service(ring_id):
        raise FraudRingInvestigationServiceError(
            "Service unavailable."
        )

    class FailingService:
        def investigate(self, ring_id):
            raise FraudRingInvestigationServiceError(
                "Service unavailable."
            )


    monkeypatch.setattr(
        main,
        "fraud_ring_investigation_service",
        FailingService(),
    )

    response = client.get(
        "/fraud-ring/ring-test"
    )

    assert response.status_code == 503


def test_fraud_ring_route_is_registered():
    routes = {
        route.path
        for route in app.routes
    }

    assert "/fraud-ring/{ring_id}" in routes