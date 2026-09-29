from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.explanation import ExplanationService


client = TestClient(app)


def test_explanation_unknown_transaction_returns_404(
    monkeypatch,
    tmp_path: Path,
):
    dataframe = pd.DataFrame(
        {
            "transaction_id": ["TXN_001"],
            "timestamp": [
                pd.Timestamp("2025-01-01", tz="UTC")
            ],
            "feature_a": [1.0],
        }
    )

    dataset_path = tmp_path / "features.parquet"
    dataframe.to_parquet(dataset_path)

    service = ExplanationService(
        feature_dataset_path=dataset_path,
        model_path=tmp_path / "model.joblib",
    )

    monkeypatch.setattr(
        "api.main.explanation_service",
        service,
    )

    response = client.get(
        "/explanation/TXN_UNKNOWN"
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_explanation_missing_model_returns_503(
    monkeypatch,
    tmp_path: Path,
):
    dataframe = pd.DataFrame(
        {
            "transaction_id": ["TXN_001"],
            "timestamp": [
                pd.Timestamp("2025-01-01", tz="UTC")
            ],
            "feature_a": [1.0],
        }
    )

    dataset_path = tmp_path / "features.parquet"
    dataframe.to_parquet(dataset_path)

    service = ExplanationService(
        feature_dataset_path=dataset_path,
        model_path=tmp_path / "missing_model.joblib",
    )

    monkeypatch.setattr(
        "api.main.explanation_service",
        service,
    )

    response = client.get(
        "/explanation/TXN_001"
    )

    assert response.status_code == 503
    assert "model artifact" in response.json()["detail"].lower()


def test_explanation_response_mapping(
    monkeypatch,
):
    class FakeService:
        def explain(self, transaction_id: str):
            from api.contracts.explanation import (
                ExplanationResponse,
                FeatureAttributionResponse,
            )

            return ExplanationResponse(
                transaction_id=transaction_id,
                prediction_probability=0.91,
                prediction_label=1,
                feature_attributions=[
                    FeatureAttributionResponse(
                        feature="velocity_1h",
                        attribution=0.42,
                        absolute_attribution=0.42,
                        rank=1,
                    )
                ],
                graph_findings=[],
                summary=(
                    "Transaction TXN_001 is suspicious."
                ),
            )

    monkeypatch.setattr(
        "api.main.explanation_service",
        FakeService(),
    )

    response = client.get(
        "/explanation/TXN_001"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["transaction_id"] == "TXN_001"
    assert body["prediction_probability"] == 0.91
    assert body["prediction_label"] == 1
    assert body["feature_attributions"][0]["feature"] == (
        "velocity_1h"
    )
    assert body["feature_attributions"][0]["rank"] == 1
    assert body["graph_findings"] == []
    assert body["summary"]


def test_explanation_empty_transaction_id_is_rejected():
    response = client.get(
        "/explanation/"
    )

    assert response.status_code == 404
