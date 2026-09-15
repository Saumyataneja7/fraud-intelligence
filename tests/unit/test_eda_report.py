from pathlib import Path


def test_eda_report_exists():
    report_path = (
        Path(__file__).resolve().parents[2]
        / "reports"
        / "eda"
        / "EDA_REPORT.md"
    )

    assert report_path.exists()


def test_eda_report_contains_required_sections():
    report_path = (
        Path(__file__).resolve().parents[2]
        / "reports"
        / "eda"
        / "EDA_REPORT.md"
    )

    content = report_path.read_text(
        encoding="utf-8",
    )

    required_sections = [
        "# Fraud Intelligence — Exploratory Data Analysis",
        "## 1. Executive Summary",
        "## 2. Dataset Profile",
        "## 3. Data Quality",
        "## 4. Fraud Distribution",
        "## 5. Fraud Scenario Distribution",
        "## 6. Transaction Amount Analysis",
        "## 7. Fraud by Payment Method",
        "## 8. Fraud by Transaction Type",
        "## 9. Temporal Analysis",
        "## 10. Velocity Analysis",
        "## 11. Entity-Level Analysis",
        "## 12. Fraud Concentration",
        "## 13. Shared Infrastructure",
        "## 14. Fraud-Ring Exploration",
        "## 15. High Fraud-Rate Entities",
        "## 16. Top Fraud-Associated Entities",
        "## 17. Leakage Audit",
        "## 18. Feature Engineering Rules",
        "## 19. Key Findings",
        "## 20. Phase 4 Requirements",
        "## 21. Limitations",
        "## 22. Phase 3 Conclusion",
    ]

    for section in required_sections:
        assert section in content


def test_eda_report_documents_leakage_rules():
    report_path = (
        Path(__file__).resolve().parents[2]
        / "reports"
        / "eda"
        / "EDA_REPORT.md"
    )

    content = report_path.read_text(
        encoding="utf-8",
    )

    for rule_id in [
        "L001",
        "L002",
        "L003",
        "L004",
        "L005",
        "L006",
        "L007",
        "L008",
    ]:
        assert rule_id in content