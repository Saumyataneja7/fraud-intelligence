from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "fraud_intelligence"
    / "FRAUD_INTELLIGENCE_REPORT.md"
)

FREEZE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "fraud_intelligence"
    / "PHASE_9_FREEZE.txt"
)


def test_fraud_intelligence_report_exists():
    assert REPORT_PATH.exists()


def test_phase_9_freeze_exists():
    assert FREEZE_PATH.exists()


def test_report_contains_phase_status():
    text = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert "Phase 9" in text
    assert "COMPLETE / FROZEN" in text


def test_report_contains_final_test_count():
    text = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert "252 tests passed" in text


def test_report_contains_all_phase_9_components():
    text = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    expected_sections = (
        "9.1 Fraud Intelligence Data Contract",
        "9.2 Suspicious Transaction Intelligence",
        "9.3 Entity Relationship Intelligence",
        "9.4 Fraud Ring Candidate Detection",
        "9.5 Ring-Level Network Scoring",
        "9.6 Fraud Ring Evidence Assembly",
        "9.7 Customer / Entity Investigation View",
        "9.8 Transaction Investigation View",
        "9.9 Fraud Ring Investigation View",
        "9.10 Intelligence Validation & Leakage Controls",
        "9.11 Fraud Intelligence Integration Tests",
        "9.12 Fraud Intelligence Report & Freeze",
    )

    for section in expected_sections:
        assert section in text


def test_report_documents_leakage_controls():
    text = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert "is_fraud" in text
    assert "fraud_scenario" in text
    assert "future context" in text.lower()
    assert "same-timestamp" in text


def test_report_documents_phase_8_phase_9_separation():
    text = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert "Why did the model make this prediction?" in text
    assert (
        "What does the surrounding fraud network tell an investigator?"
        in text
    )


def test_report_documents_structural_score_constraint():
    text = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert "NOT:" in text
    assert "a fraud probability" in text


def test_report_documents_frozen_dependencies():
    text = REPORT_PATH.read_text(
        encoding="utf-8"
    )

    assert "Phase 4 feature engineering" in text
    assert "Phase 5 classical ML" in text
    assert "Phase 6 heterogeneous graph" in text
    assert "Phase 7 graph ML" in text
    assert "Phase 8 explainability" in text


def test_freeze_contains_final_result():
    text = FREEZE_PATH.read_text(
        encoding="utf-8"
    )

    assert "PHASE 9 IS FROZEN." in text
    assert "252 passed" in text


def test_freeze_contains_leakage_controls():
    text = FREEZE_PATH.read_text(
        encoding="utf-8"
    )

    assert "No is_fraud predictive leakage" in text
    assert "No fraud_scenario predictive leakage" in text
    assert "No future temporal context" in text
    assert "No same-timestamp event context" in text