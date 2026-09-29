from __future__ import annotations

from unittest.mock import Mock

from fastapi.testclient import TestClient

from api.main import app
from api.services.transaction_investigation import (
    TransactionInvestigationService,
    TransactionInvestigationServiceError,
)


client = TestClient(app)


def test_transaction_endpoint_returns_404_for_unknown_transaction():
    response = client.get(
        "/transaction/non-existent-transaction"
    )

    assert response.status_code == 404


def test_transaction_endpoint_returns_503_when_service_fails():
    mock_service = Mock(
        spec=TransactionInvestigationService
    )

    mock_service.investigate.side_effect = (
        TransactionInvestigationServiceError(
            "service unavailable"
        )
    )

    original_service = __import__(
        "api.main",
        fromlist=["transaction_investigation_service"],
    ).transaction_investigation_service

    try:
        import api.main as main_module

        main_module.transaction_investigation_service = (
            mock_service
        )

        response = client.get(
            "/transaction/tx-000001"
        )

        assert response.status_code == 503

    finally:
        main_module.transaction_investigation_service = (
            original_service
        )


def test_transaction_endpoint_is_registered():
    routes = {
        route.path
        for route in app.routes
    }

    assert "/transaction/{transaction_id}" in routes