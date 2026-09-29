from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from api.services.model_metrics import (
    ModelMetricsService,
)


client = TestClient(app)


def test_model_metrics_endpoint_returns_frozen_metrics():
    response = client.get("/model-metrics")

    assert response.status_code == 200

    body = response.json()

    assert body["feature_dataset"] == (
        "data/processed/features/feature_dataset.parquet"
    )

    assert body["feature_count"] == 53

    assert body["models"] == [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ]

    assert body["splits"] == [
        "validation",
        "test",
    ]

    assert len(body["results"]) == 6


def test_model_metrics_contains_xgboost_test_metrics():
    response = client.get("/model-metrics")

    assert response.status_code == 200

    results = response.json()["results"]

    xgboost_test = next(
        result
        for result in results
        if result["model_name"] == "xgboost"
        and result["split_name"] == "test"
    )

    metrics = xgboost_test["metrics"]

    assert metrics["precision"] == 0.23275862068965517
    assert metrics["recall"] == 0.391304347826087
    assert metrics["f1"] == 0.2918918918918919
    assert metrics["pr_auc"] == 0.3330420050841898
    assert metrics["roc_auc"] == 0.9043255925529363

    assert metrics["true_negatives"] == 14684
    assert metrics["false_positives"] == 178
    assert metrics["false_negatives"] == 84
    assert metrics["true_positives"] == 54

    assert metrics["support"] == 15000
    assert metrics["predicted_positive_count"] == 232
    assert metrics["actual_positive_count"] == 138


def test_model_metrics_missing_artifact_returns_503(
    monkeypatch,
    tmp_path: Path,
):
    service = ModelMetricsService(
        metrics_path=tmp_path / "missing.json"
    )

    monkeypatch.setattr(
        "api.main.model_metrics_service",
        service,
    )

    response = client.get("/model-metrics")

    assert response.status_code == 503
    assert "not found" in (
        response.json()["detail"].lower()
    )


def test_model_metrics_invalid_artifact_returns_503(
    monkeypatch,
    tmp_path: Path,
):
    metrics_path = tmp_path / "invalid.json"

    metrics_path.write_text(
        json.dumps(
            {
                "feature_dataset": "x",
            }
        ),
        encoding="utf-8",
    )

    service = ModelMetricsService(
        metrics_path=metrics_path
    )

    monkeypatch.setattr(
        "api.main.model_metrics_service",
        service,
    )

    response = client.get("/model-metrics")

    assert response.status_code == 503
    assert "missing required" in (
        response.json()["detail"].lower()
    )
