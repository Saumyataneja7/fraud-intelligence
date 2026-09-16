from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "feature_engineering"
    / "FEATURE_ENGINEERING_REPORT.md"
)


def test_feature_engineering_report_exists():
    assert REPORT_PATH.exists()


def test_feature_engineering_report_contains_required_sections():
    content = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    required_sections = [
        "# Phase 4 — Feature Engineering Report",
        "## 1. Executive Summary",
        "## 2. Phase 4 Architecture",
        "## 3. Dataset Profile",
        "## 4. Feature Family Summary",
        "## 5. Complete Feature Dictionary",
        "## 6. Temporal Semantics",
        "## 7. Cold-Start Behavior",
        "## 8. Model Feature Boundary",
        "## 9. Leakage Controls",
        "## 10. Reproducibility",
        "## 11. Validation",
        "## 12. Known Design Limitations",
        "## 13. Phase 4 Deliverables",
        "## 14. Final Phase 4 Conclusion",
    ]

    for section in required_sections:
        assert section in content


def test_feature_engineering_report_documents_53_features():
    content = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert "**53 engineered model features**" in content


def test_feature_engineering_report_documents_leakage_controls():
    content = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert "Same-timestamp leakage validation" in content
    assert "Future-transaction invariance" in content
    assert "Target columns excluded from model features" in content


def test_feature_engineering_report_documents_phase_completion():
    content = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert "Phase 4 is complete and frozen." in content