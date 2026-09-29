from __future__ import annotations

from unittest.mock import Mock

from fastapi.testclient import TestClient

from api.main import app
from api.services.entity_investigation import (
    EntityInvestigationService,
    EntityInvestigationServiceError,
)


client = TestClient(app)


def test_customer_endpoint_returns_404_for_unknown_customer():
    response = client.get(
        "/customer/non-existent-customer"
    )

    assert response.status_code == 404


def test_customer_endpoint_returns_503_when_service_fails():
    mock_service = Mock(
        spec=EntityInvestigationService
    )

    mock_service.investigate.side_effect = (
        EntityInvestigationServiceError(
            "service unavailable"
        )
    )

    import api.main as main_module

    original_service = (
        main_module.entity_investigation_service
    )

    try:
        main_module.entity_investigation_service = (
            mock_service
        )

        response = client.get(
            "/customer/customer_001"
        )

        assert response.status_code == 503

    finally:
        main_module.entity_investigation_service = (
            original_service
        )


def test_customer_endpoint_is_registered():
    routes = {
        route.path
        for route in app.routes
    }

    assert "/customer/{customer_id}" in routes