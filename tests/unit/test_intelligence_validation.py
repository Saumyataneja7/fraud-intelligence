from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fraud_intelligence.intelligence.contracts import (
    FraudIntelligenceContractError,
)
from fraud_intelligence.intelligence.entity_relationships import (
    EntityReference,
    EntityRelationship,
)
from fraud_intelligence.intelligence.fraud_ring_candidates import (
    FraudRingCandidate,
)
from fraud_intelligence.intelligence.fraud_ring_evidence import (
    FraudRingEvidence,
    RingEvidenceItem,
)
from fraud_intelligence.intelligence.ring_investigation import (
    assemble_fraud_ring_investigation_view,
)
from fraud_intelligence.intelligence.ring_network_scoring import (
    RingNetworkScore,
)
from fraud_intelligence.intelligence.suspicious_transaction import (
    SuspicionSignal,
    SuspiciousTransactionIntelligence,
)
from fraud_intelligence.intelligence.transaction_investigation import (
    assemble_transaction_investigation_view,
)
from fraud_intelligence.intelligence.validation import (
    IntelligenceLeakageIssue,
    IntelligenceValidationReport,
    assert_fraud_intelligence_valid,
    validate_cross_component_consistency,
    validate_fraud_intelligence,
    validate_no_forbidden_predictive_fields,
    validate_relationship_temporal_context,
    validate_ring_evidence_leakage,
    validate_ring_investigation_leakage,
    validate_suspicious_transaction_intelligence_leakage,
    validate_transaction_investigation_leakage,
)


TARGET_TIME = datetime(
    2025,
    6,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)

HISTORICAL_TIME = TARGET_TIME - timedelta(hours=1)
SAME_TIME = TARGET_TIME
FUTURE_TIME = TARGET_TIME + timedelta(hours=1)


def relationship(
    relationship_type: str,
    source_type: str,
    source_id: str,
    target_type: str,
    target_id: str,
    timestamp=None,
):
    return EntityRelationship(
        relationship_type=relationship_type,
        source=EntityReference(source_type, source_id),
        target=EntityReference(target_type, target_id),
        timestamp=timestamp,
    )


def make_transaction_intelligence():
    return SuspiciousTransactionIntelligence(
        transaction_id="txn-1",
        timestamp=TARGET_TIME,
        prediction_probability=0.82,
        prediction_label=1,
        signals=(
            SuspicionSignal(
                name="velocity",
                value=4,
                description="Historical velocity signal.",
                source="feature",
            ),
        ),
        entity_counts=(("customer", 1),),
    )


def make_candidate():
    relationships = (
        relationship(
            "customer_transaction",
            "customer",
            "c1",
            "transaction",
            "txn-1",
            HISTORICAL_TIME,
        ),
        relationship(
            "transaction_device",
            "transaction",
            "txn-1",
            "device",
            "d1",
            HISTORICAL_TIME,
        ),
    )

    return FraudRingCandidate(
        candidate_id="ring-1",
        entities=(
            EntityReference("customer", "c1"),
            EntityReference("device", "d1"),
            EntityReference("transaction", "txn-1"),
        ),
        relationships=relationships,
        connected_component_size=3,
        shared_relationship_count=2,
    )


def make_score():
    return RingNetworkScore(
        candidate_id="ring-1",
        entity_count=3,
        relationship_count=2,
        relationship_density=0.5,
        entity_type_count=3,
        relationship_type_count=2,
        non_transaction_entity_count=2,
        non_transaction_connectivity=1.0,
        structural_score=65.0,
    )


def make_evidence():
    candidate = make_candidate()
    score = make_score()

    return FraudRingEvidence(
        candidate_id="ring-1",
        target_transaction_id="txn-1",
        entities=candidate.entities,
        relationships=candidate.relationships,
        network_score=score,
        evidence_items=(
            RingEvidenceItem(
                evidence_type="entity_connectivity",
                description="Historical entity connectivity.",
                entity_types=("customer", "device", "transaction"),
                relationship_types=(
                    "customer_transaction",
                    "transaction_device",
                ),
                strength="moderate",
            ),
        ),
        temporal_context="Historical context only.",
    )


def make_transaction_view(
    relationships=None,
):
    intelligence = make_transaction_intelligence()

    if relationships is None:
        relationships = (
            relationship(
                "customer_transaction",
                "customer",
                "c1",
                "transaction",
                "txn-1",
                HISTORICAL_TIME,
            ),
        )

    return assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TARGET_TIME,
        suspicious_transaction=intelligence,
        relationships=relationships,
    )


def test_forbidden_field_detection():
    issues = validate_no_forbidden_predictive_fields(
        {
            "feature": "is_fraud",
        },
        component="test",
    )

    assert len(issues) == 1
    assert issues[0].code == "IL001"


def test_clean_value_has_no_forbidden_field():
    issues = validate_no_forbidden_predictive_fields(
        {
            "description": "Historical velocity signal",
        },
        component="test",
    )

    assert issues == ()


def test_nested_forbidden_field_detection():
    issues = validate_no_forbidden_predictive_fields(
        {
            "nested": (
                {
                    "value": "fraud_scenario",
                },
            ),
        },
        component="test",
    )

    assert len(issues) == 1
    assert issues[0].code == "IL001"


def test_historical_relationship_is_allowed():
    relationships = (
        relationship(
            "customer_transaction",
            "customer",
            "c1",
            "transaction",
            "txn-1",
            HISTORICAL_TIME,
        ),
    )

    issues = validate_relationship_temporal_context(
        relationships=relationships,
        target_timestamp=TARGET_TIME,
        component="test",
    )

    assert issues == ()


def test_untimestamped_relationship_is_allowed():
    relationships = (
        relationship(
            "customer_device",
            "customer",
            "c1",
            "device",
            "d1",
            None,
        ),
    )

    issues = validate_relationship_temporal_context(
        relationships=relationships,
        target_timestamp=TARGET_TIME,
        component="test",
    )

    assert issues == ()


def test_same_timestamp_is_rejected():
    relationships = (
        relationship(
            "customer_transaction",
            "customer",
            "c1",
            "transaction",
            "txn-1",
            SAME_TIME,
        ),
    )

    issues = validate_relationship_temporal_context(
        relationships=relationships,
        target_timestamp=TARGET_TIME,
        component="test",
    )

    assert len(issues) == 1
    assert issues[0].code == "IL003"


def test_future_timestamp_is_rejected():
    relationships = (
        relationship(
            "customer_transaction",
            "customer",
            "c1",
            "transaction",
            "txn-1",
            FUTURE_TIME,
        ),
    )

    issues = validate_relationship_temporal_context(
        relationships=relationships,
        target_timestamp=TARGET_TIME,
        component="test",
    )

    assert len(issues) == 1
    assert issues[0].code == "IL002"


def test_suspicious_transaction_clean():
    issues = validate_suspicious_transaction_intelligence_leakage(
        make_transaction_intelligence()
    )

    assert issues == ()


def test_suspicious_transaction_invalid_probability():
    intelligence = make_transaction_intelligence()

    invalid = SuspiciousTransactionIntelligence(
        transaction_id=intelligence.transaction_id,
        timestamp=intelligence.timestamp,
        prediction_probability=2.0,
        prediction_label=intelligence.prediction_label,
        signals=intelligence.signals,
        entity_counts=intelligence.entity_counts,
    )

    issues = validate_suspicious_transaction_intelligence_leakage(
        invalid
    )

    assert any(issue.code == "IL004" for issue in issues)


def test_suspicious_transaction_invalid_label():
    intelligence = make_transaction_intelligence()

    invalid = SuspiciousTransactionIntelligence(
        transaction_id=intelligence.transaction_id,
        timestamp=intelligence.timestamp,
        prediction_probability=0.5,
        prediction_label=3,
        signals=intelligence.signals,
        entity_counts=intelligence.entity_counts,
    )

    issues = validate_suspicious_transaction_intelligence_leakage(
        invalid
    )

    assert any(issue.code == "IL005" for issue in issues)


def test_transaction_view_clean():
    view = make_transaction_view()

    issues = validate_transaction_investigation_leakage(view)

    assert issues == ()


def test_transaction_view_rejects_future_relationship():
    view = make_transaction_view(
        relationships=(
            relationship(
                "customer_transaction",
                "customer",
                "c1",
                "transaction",
                "txn-1",
                FUTURE_TIME,
            ),
        )
    )

    issues = validate_transaction_investigation_leakage(view)

    assert any(issue.code == "IL002" for issue in issues)


def test_transaction_view_rejects_same_timestamp_relationship():
    view = make_transaction_view(
        relationships=(
            relationship(
                "customer_transaction",
                "customer",
                "c1",
                "transaction",
                "txn-1",
                SAME_TIME,
            ),
        )
    )

    issues = validate_transaction_investigation_leakage(view)

    assert any(issue.code == "IL003" for issue in issues)


def test_transaction_view_relationships_must_contain_target():
    intelligence = make_transaction_intelligence()

    view = assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TARGET_TIME,
        suspicious_transaction=intelligence,
        relationships=(
            relationship(
                "customer_device",
                "customer",
                "c1",
                "device",
                "d1",
                HISTORICAL_TIME,
            ),
        ),
    )

    issues = validate_transaction_investigation_leakage(view)

    assert any(issue.code == "IL006" for issue in issues)


def test_ring_evidence_clean():
    issues = validate_ring_evidence_leakage(
        make_evidence(),
        target_timestamp=TARGET_TIME,
    )

    assert issues == ()


def test_ring_evidence_rejects_future_relationship():
    evidence = make_evidence()

    future_relationship = relationship(
        "customer_transaction",
        "customer",
        "c1",
        "transaction",
        "txn-1",
        FUTURE_TIME,
    )

    invalid = FraudRingEvidence(
        candidate_id=evidence.candidate_id,
        target_transaction_id=evidence.target_transaction_id,
        entities=evidence.entities,
        relationships=(
            future_relationship,
        ),
        network_score=evidence.network_score,
        evidence_items=evidence.evidence_items,
        temporal_context=evidence.temporal_context,
    )

    issues = validate_ring_evidence_leakage(
        invalid,
        target_timestamp=TARGET_TIME,
    )

    assert any(issue.code == "IL002" for issue in issues)


def test_ring_investigation_clean():
    view = assemble_fraud_ring_investigation_view(
        candidate=make_candidate(),
        network_score=make_score(),
        evidence=make_evidence(),
    )

    issues = validate_ring_investigation_leakage(
        view,
        target_timestamp=TARGET_TIME,
    )

    assert issues == ()


def test_ring_investigation_rejects_future_context():
    candidate = make_candidate()

    future_relationship = relationship(
        "transaction_device",
        "transaction",
        "txn-1",
        "device",
        "d1",
        FUTURE_TIME,
    )

    invalid_candidate = FraudRingCandidate(
        candidate_id=candidate.candidate_id,
        entities=candidate.entities,
        relationships=(
            future_relationship,
        ),
        connected_component_size=3,
        shared_relationship_count=1,
    )

    view = assemble_fraud_ring_investigation_view(
        candidate=invalid_candidate,
        network_score=make_score(),
    )

    issues = validate_ring_investigation_leakage(
        view,
        target_timestamp=TARGET_TIME,
    )

    assert any(issue.code == "IL002" for issue in issues)


def test_cross_component_consistency_is_clean():
    intelligence = make_transaction_intelligence()
    view = make_transaction_view()

    issues = validate_cross_component_consistency(
        transaction_intelligence=intelligence,
        transaction_view=view,
    )

    assert issues == ()


def test_cross_component_transaction_id_mismatch():
    intelligence = make_transaction_intelligence()
    view = make_transaction_view()

    altered = type(view)(
        transaction_id="txn-2",
        timestamp=view.timestamp,
        prediction_probability=view.prediction_probability,
        prediction_label=view.prediction_label,
        suspicion_signals=view.suspicion_signals,
        related_entities=view.related_entities,
        relationships=view.relationships,
        candidate_ring_ids=view.candidate_ring_ids,
        network_scores=view.network_scores,
        ring_evidence=view.ring_evidence,
        investigation_summary=view.investigation_summary,
        temporal_rule=view.temporal_rule,
    )

    issues = validate_cross_component_consistency(
        transaction_intelligence=intelligence,
        transaction_view=altered,
    )

    assert any(issue.code == "IL008" for issue in issues)


def test_cross_component_probability_mismatch():
    intelligence = make_transaction_intelligence()
    view = make_transaction_view()

    altered = type(view)(
        transaction_id=view.transaction_id,
        timestamp=view.timestamp,
        prediction_probability=0.91,
        prediction_label=view.prediction_label,
        suspicion_signals=view.suspicion_signals,
        related_entities=view.related_entities,
        relationships=view.relationships,
        candidate_ring_ids=view.candidate_ring_ids,
        network_scores=view.network_scores,
        ring_evidence=view.ring_evidence,
        investigation_summary=view.investigation_summary,
        temporal_rule=view.temporal_rule,
    )

    issues = validate_cross_component_consistency(
        transaction_intelligence=intelligence,
        transaction_view=altered,
    )

    assert any(issue.code == "IL010" for issue in issues)


def test_cross_component_label_mismatch():
    intelligence = make_transaction_intelligence()
    view = make_transaction_view()

    altered = type(view)(
        transaction_id=view.transaction_id,
        timestamp=view.timestamp,
        prediction_probability=view.prediction_probability,
        prediction_label=0,
        suspicion_signals=view.suspicion_signals,
        related_entities=view.related_entities,
        relationships=view.relationships,
        candidate_ring_ids=view.candidate_ring_ids,
        network_scores=view.network_scores,
        ring_evidence=view.ring_evidence,
        investigation_summary=view.investigation_summary,
        temporal_rule=view.temporal_rule,
    )

    issues = validate_cross_component_consistency(
        transaction_intelligence=intelligence,
        transaction_view=altered,
    )

    assert any(issue.code == "IL011" for issue in issues)


def test_complete_validation_passes():
    intelligence = make_transaction_intelligence()
    transaction_view = make_transaction_view()

    ring_view = assemble_fraud_ring_investigation_view(
        candidate=make_candidate(),
        network_score=make_score(),
        evidence=make_evidence(),
    )

    report = validate_fraud_intelligence(
        transaction_intelligence=intelligence,
        transaction_view=transaction_view,
        ring_evidence=(make_evidence(),),
        ring_views=(ring_view,),
        target_timestamp=TARGET_TIME,
    )

    assert report.passed is True
    assert report.issue_count == 0
    assert "transaction_investigation" in (
        report.checked_components
    )


def test_complete_validation_detects_future_context():
    intelligence = make_transaction_intelligence()

    transaction_view = make_transaction_view(
        relationships=(
            relationship(
                "customer_transaction",
                "customer",
                "c1",
                "transaction",
                "txn-1",
                FUTURE_TIME,
            ),
        )
    )

    report = validate_fraud_intelligence(
        transaction_intelligence=intelligence,
        transaction_view=transaction_view,
    )

    assert report.passed is False
    assert report.issue_count >= 1


def test_validation_report_issue_count():
    report = IntelligenceValidationReport(
        passed=False,
        issues=(
            IntelligenceLeakageIssue(
                code="IL001",
                message="Test issue.",
                component="test",
            ),
        ),
        checked_components=("test",),
    )

    assert report.issue_count == 1


def test_assert_valid_does_not_raise():
    report = IntelligenceValidationReport(
        passed=True,
        issues=(),
        checked_components=("test",),
    )

    assert_fraud_intelligence_valid(report)


def test_assert_invalid_raises():
    report = IntelligenceValidationReport(
        passed=False,
        issues=(
            IntelligenceLeakageIssue(
                code="IL001",
                message="Test failure.",
                component="test",
            ),
        ),
        checked_components=("test",),
    )

    with pytest.raises(FraudIntelligenceContractError):
        assert_fraud_intelligence_valid(report)