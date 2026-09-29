from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint_returns_200():
    response = client.get("/health")

    assert response.status_code == 200


def test_health_endpoint_response():
    response = client.get("/health")

    assert response.json() == {
        "status": "ok",
        "service": "fraud-intelligence-api",
        "version": "1.0.0",
    }


def test_health_endpoint_content_type():
    response = client.get("/health")

    assert response.headers["content-type"].startswith(
        "application/json"
    )


def test_metadata_endpoint_returns_200():
    response = client.get("/metadata")

    assert response.status_code == 200


def test_metadata_contains_service_information():
    response = client.get("/metadata")
    data = response.json()

    assert data["service"] == "fraud-intelligence-api"
    assert data["version"] == "1.0.0"
    assert data["project"] == "Fraud Intelligence"


def test_metadata_contains_phase():
    response = client.get("/metadata")
    data = response.json()

    assert data["phase"] == "10.2"
    assert data["status"] == "operational"


def test_metadata_contains_dataset():
    response = client.get("/metadata")
    data = response.json()

    assert data["dataset"] == {
        "name": "fraud-intelligence-synthetic",
        "version": "1.0.0",
    }


def test_metadata_contains_architecture():
    response = client.get("/metadata")
    data = response.json()

    architecture = data["architecture"]

    assert architecture["feature_engineering"] == "Phase 4"
    assert architecture["classical_ml"] == "Phase 5"
    assert architecture["graph"] == "Phase 6"
    assert architecture["graph_ml"] == "Phase 7"
    assert architecture["explainability"] == "Phase 8"
    assert architecture["fraud_intelligence"] == "Phase 9"
    assert architecture["api"] == "Phase 10"


def test_metadata_contains_frozen_phases():
    response = client.get("/metadata")
    data = response.json()

    assert data["frozen_phases"] == [
        "Phase 1",
        "Phase 2",
        "Phase 3",
        "Phase 4",
        "Phase 5",
        "Phase 6",
        "Phase 7",
        "Phase 8",
        "Phase 9",
    ]


def test_unknown_endpoint_returns_404():
    response = client.get("/does-not-exist")

    assert response.status_code == 404