from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.prediction import (
    PredictionService,
    PredictionServiceError,
)


client = TestClient(app)


def test_predict_endpoint_exists():
    response = client.post(
        "/predict",
        json={"transaction_id": "txn_001"},
    )

    assert response.status_code in {404, 503}


def test_prediction_request_validation():
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422


def test_prediction_request_rejects_extra_fields():
    response = client.post(
        "/predict",
        json={
            "transaction_id": "txn_001",
            "unexpected": "value",
        },
    )

    assert response.status_code == 422


def test_prediction_service_rejects_empty_transaction_id():
    service = PredictionService(
        feature_dataset_path=Path("/does/not/exist"),
        model_path=Path("/does/not/exist"),
    )

    with pytest.raises(PredictionServiceError):
        service.predict("")


def test_prediction_service_missing_dataset():
    service = PredictionService(
        feature_dataset_path=Path("/does/not/exist"),
        model_path=Path("/does/not/exist"),
    )

    with pytest.raises(PredictionServiceError):
        service.predict("txn_001")


def test_prediction_service_missing_model(tmp_path):
    dataset_path = tmp_path / "features.parquet"

    import pandas as pd

    pd.DataFrame(
        {
            "transaction_id": ["txn_001"],
        }
    ).to_parquet(dataset_path)

    service = PredictionService(
        feature_dataset_path=dataset_path,
        model_path=tmp_path / "missing_model.joblib",
    )

    with pytest.raises(PredictionServiceError):
        service.predict("txn_001")


def test_predict_unknown_transaction_returns_404(monkeypatch):
    class FakeService:
        def predict(self, transaction_id):
            raise KeyError(
                f"Transaction not found: {transaction_id}"
            )

    import api.main

    monkeypatch.setattr(
        api.main,
        "prediction_service",
        FakeService(),
    )

    response = client.post(
        "/predict",
        json={"transaction_id": "does_not_exist"},
    )

    assert response.status_code == 404


def test_predict_service_failure_returns_503(monkeypatch):
    class FakeService:
        def predict(self, transaction_id):
            raise PredictionServiceError(
                "Model artifact unavailable."
            )

    import api.main

    monkeypatch.setattr(
        api.main,
        "prediction_service",
        FakeService(),
    )

    response = client.post(
        "/predict",
        json={"transaction_id": "txn_001"},
    )

    assert response.status_code == 503