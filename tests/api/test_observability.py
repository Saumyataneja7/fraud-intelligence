from __future__ import annotations

import json
import logging
import re

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_request_id_is_generated_and_returned() -> None:
    response = client.get("/health")

    request_id = response.headers["X-Request-ID"]
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        request_id,
    )


def test_valid_request_id_is_preserved() -> None:
    request_id = "test-request-123"

    response = client.get(
        "/health",
        headers={"X-Request-ID": request_id},
    )

    assert response.headers["X-Request-ID"] == request_id


def test_invalid_request_id_is_replaced() -> None:
    response = client.get(
        "/health",
        headers={"X-Request-ID": "bad request id"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "bad request id"
    assert " " not in response.headers["X-Request-ID"]


def test_request_log_contains_status_and_latency(
    caplog,
) -> None:
    with caplog.at_level(
        logging.INFO,
        logger="fraud_intelligence.api",
    ):
        response = client.get("/health")

    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "fraud_intelligence.api"
    ]
    assert records

    log_record = records[-1]
    assert log_record["event"] == "http_request"
    assert log_record["request_id"] == response.headers["X-Request-ID"]
    assert log_record["method"] == "GET"
    assert log_record["route"] == "/health"
    assert log_record["status_code"] == 200
    assert isinstance(log_record["duration_ms"], (int, float))
    assert log_record["duration_ms"] >= 0


def test_unknown_route_is_logged_with_404(caplog) -> None:
    with caplog.at_level(
        logging.INFO,
        logger="fraud_intelligence.api",
    ):
        response = client.get("/observability-test-missing-route")

    assert response.status_code == 404

    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "fraud_intelligence.api"
    ]
    assert records
    assert records[-1]["status_code"] == 404


def test_request_logs_do_not_include_transaction_ids(caplog) -> None:
    transaction_id = "private-transaction-12345"

    with caplog.at_level(
        logging.INFO,
        logger="fraud_intelligence.api",
    ):
        response = client.get(f"/transaction/{transaction_id}")

    assert response.status_code == 404

    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "fraud_intelligence.api"
    ]
    assert records

    log_record = records[-1]
    assert log_record["route"] == "/transaction/{transaction_id}"
    assert transaction_id not in json.dumps(log_record)