from __future__ import annotations

from fastapi.testclient import TestClient

from api import main
from api.main import app
from api.services.graph import (
    GraphInvestigationService,
    GraphInvestigationServiceError,
)


client = TestClient(app)


def test_graph_endpoint_is_registered():
    response = client.get(
        "/graph/0",
        params={"node_type": "customer"},
    )

    assert response.status_code != 404 or response.json().get(
        "detail"
    ) != "Not Found"


def test_graph_unknown_node_type_returns_404():
    response = client.get(
        "/graph/0",
        params={"node_type": "unknown"},
    )

    assert response.status_code == 404


def test_graph_negative_node_id_returns_422():
    response = client.get(
        "/graph/-1",
        params={"node_type": "customer"},
    )

    assert response.status_code == 422


def test_graph_unknown_node_id_returns_404():
    response = client.get(
        "/graph/999999999",
        params={"node_type": "customer"},
    )

    assert response.status_code == 404


def test_graph_service_failure_returns_503(monkeypatch):
    class FailingService:
        def investigate(
            self,
            *,
            node_type,
            node_id,
        ):
            raise GraphInvestigationServiceError(
                "Graph unavailable."
            )

    monkeypatch.setattr(
        main,
        "graph_investigation_service",
        FailingService(),
    )

    response = client.get(
        "/graph/0",
        params={"node_type": "customer"},
    )

    assert response.status_code == 503

    data = response.json()

    assert data["detail"] == (
        "Graph investigation service is unavailable."
    )


def test_graph_response_shape():
    service = GraphInvestigationService()

    response = service.investigate(
        node_type="customer",
        node_id=0,
    )

    assert response.target_node_type == "customer"
    assert response.target_node_id == 0

    assert response.nodes
    assert any(
        node.node_type == "customer"
        and node.node_id == 0
        for node in response.nodes
    )

    for edge in response.edges:
        assert edge.relationship_type
        assert edge.source_node_type
        assert edge.target_node_type
        assert edge.source_node_id >= 0
        assert edge.target_node_id >= 0


def test_graph_target_node_is_always_included():
    service = GraphInvestigationService()

    response = service.investigate(
        node_type="transaction",
        node_id=0,
    )

    assert (
        response.target_node_type,
        response.target_node_id,
    ) in {
        (
            node.node_type,
            node.node_id,
        )
        for node in response.nodes
    }