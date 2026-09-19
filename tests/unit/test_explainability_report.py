from __future__ import annotations

from fraud_intelligence.explainability.comparison import (
    build_explainability_comparison,
)
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "explainability"
    / "EXPLAINABILITY_REPORT.md"
)

FREEZE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "explainability"
    / "PHASE_8_FREEZE.txt"
)


def test_explainability_comparison_has_expected_methods() -> None:
    methods = {
        item.method
        for item in build_explainability_comparison()
    }

    assert methods == {
        "Classical Attribution",
        "SHAP",
        "GNN Neighborhood",
        "GNN Importance",
    }


def test_report_exists() -> None:
    assert REPORT_PATH.exists()


def test_freeze_marker_exists() -> None:
    assert FREEZE_PATH.exists()


def test_report_is_markdown() -> None:
    assert REPORT_PATH.suffix == ".md"


def test_report_contains_phase_title() -> None:
    content = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert "# Explainability Report" in content
    assert "Phase 8 — Explainability" in content


def test_report_contains_all_subphases() -> None:
    content = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    for phase in (
        "8.1",
        "8.2",
        "8.3",
        "8.4",
        "8.5",
        "8.6",
        "8.7",
        "8.8",
        "8.9",
        "8.10",
        "8.11",
    ):
        assert phase in content


def test_report_contains_temporal_rule() -> None:
    content = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert (
        "neighbor_timestamp < target_timestamp"
        in content
    )


def test_report_contains_forbidden_fields() -> None:
    content = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    for field in (
        "is_fraud",
        "fraud_scenario",
        "customer_id",
        "account_id",
        "device_id",
        "ip_id",
        "timestamp",
    ):
        assert field in content


def test_report_documents_no_ranking() -> None:
    content = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert (
        "without ranking methods"
        in content
    )


def test_report_documents_fraud_limitation() -> None:
    content = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert (
        "Proof of fraud"
        in content
    )


def test_freeze_marker_contains_all_subphases() -> None:
    content = FREEZE_PATH.read_text(
        encoding="utf-8"
    )

    for phase in (
        "8.1",
        "8.2",
        "8.3",
        "8.4",
        "8.5",
        "8.6",
        "8.7",
        "8.8",
        "8.9",
        "8.10",
        "8.11",
        "8.12",
    ):
        assert phase in content


def test_freeze_marker_declares_frozen() -> None:
    content = FREEZE_PATH.read_text(
        encoding="utf-8"
    )

    assert "COMPLETE / FROZEN" in content