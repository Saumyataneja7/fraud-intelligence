from __future__ import annotations

from datetime import datetime, timezone

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
from fraud_intelligence.intelligence.ring_network_scoring import (
    RingNetworkScore,
)
from fraud_intelligence.intelligence.suspicious_transaction import (
    SuspicionSignal,
    SuspiciousTransactionIntelligence,
)
from fraud_intelligence.intelligence.transaction_investigation import (
    TransactionInvestigationView,
    assemble_transaction_investigation_view,
    build_transaction_investigation_summary,
    extract_transaction_entities,
    extract_transaction_network_scores,
    extract_transaction_ring_evidence,
    extract_transaction_ring_ids,
    validate_transaction_investigation_view,
)


TIMESTAMP = datetime(
    2025,
    6,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)


def rel(
    relationship_type: str,
    source_type: str,
    source_id: str,
    target_type: str,
    target_id: str,
) -> EntityRelationship:
    return EntityRelationship(
        relationship_type=relationship_type,
        source=EntityReference(source_type, source_id),
        target=EntityReference(target_type, target_id),
        timestamp=None,
    )


def make_suspicious_transaction() -> SuspiciousTransactionIntelligence:
    return SuspiciousTransactionIntelligence(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        prediction_probability=0.82,
        prediction_label=1,
        signals=(
            SuspicionSignal(
                name="amount_vs_customer_mean",
                value=3.2,
                description="Elevated historical amount ratio.",
                source="engineered_transaction_feature",
            ),
            SuspicionSignal(
                name="is_new_device",
                value=True,
                description="New device for customer.",
                source="engineered_transaction_feature",
            ),
        ),
        entity_counts=(
            ("device", 1),
            ("customer", 1),
        ),
    )


def make_candidate() -> FraudRingCandidate:
    relationships = (
        rel(
            "customer_transaction",
            "customer",
            "c1",
            "transaction",
            "txn-1",
        ),
        rel(
            "transaction_device",
            "transaction",
            "txn-1",
            "device",
            "d1",
        ),
        rel(
            "customer_device",
            "customer",
            "c1",
            "device",
            "d1",
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
        shared_relationship_count=3,
    )


def make_score() -> RingNetworkScore:
    return RingNetworkScore(
        candidate_id="ring-1",
        entity_count=3,
        relationship_count=3,
        relationship_density=1.0,
        entity_type_count=3,
        relationship_type_count=3,
        non_transaction_entity_count=2,
        non_transaction_connectivity=1.0,
        structural_score=70.0,
    )


def make_evidence() -> FraudRingEvidence:
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
                description="Multiple entities are connected.",
                entity_types=("customer", "device", "transaction"),
                relationship_types=(
                    "customer_transaction",
                    "customer_device",
                    "transaction_device",
                ),
                strength="strong",
            ),
        ),
        temporal_context="Historical investigation context only.",
    )


def test_extract_transaction_entities_includes_target():
    relationships = (
        rel(
            "customer_transaction",
            "customer",
            "c1",
            "transaction",
            "txn-1",
        ),
        rel(
            "transaction_device",
            "transaction",
            "txn-1",
            "device",
            "d1",
        ),
    )

    entities = extract_transaction_entities(
        transaction_id="txn-1",
        relationships=relationships,
    )

    assert entities == (
        EntityReference("customer", "c1"),
        EntityReference("device", "d1"),
        EntityReference("transaction", "txn-1"),
    )


def test_extract_transaction_entities_ignores_unrelated_entities():
    relationships = (
        rel(
            "customer_transaction",
            "customer",
            "c1",
            "transaction",
            "txn-1",
        ),
        rel(
            "customer_device",
            "customer",
            "c2",
            "device",
            "d2",
        ),
    )

    entities = extract_transaction_entities(
        transaction_id="txn-1",
        relationships=relationships,
    )

    assert entities == (
        EntityReference("customer", "c1"),
        EntityReference("transaction", "txn-1"),
    )


def test_extract_transaction_ring_ids():
    result = extract_transaction_ring_ids(
        transaction_id="txn-1",
        candidates=(make_candidate(),),
    )

    assert result == ("ring-1",)


def test_transaction_without_ring_has_no_ring_ids():
    result = extract_transaction_ring_ids(
        transaction_id="txn-other",
        candidates=(make_candidate(),),
    )

    assert result == ()


def test_extract_transaction_network_scores():
    score = make_score()

    result = extract_transaction_network_scores(
        candidate_ring_ids=("ring-1",),
        network_scores=(score,),
    )

    assert result == (score,)


def test_extract_transaction_network_scores_ignores_other_rings():
    score = make_score()

    other_score = RingNetworkScore(
        candidate_id="ring-2",
        entity_count=4,
        relationship_count=4,
        relationship_density=0.5,
        entity_type_count=3,
        relationship_type_count=3,
        non_transaction_entity_count=3,
        non_transaction_connectivity=1.0,
        structural_score=60.0,
    )

    result = extract_transaction_network_scores(
        candidate_ring_ids=("ring-1",),
        network_scores=(other_score, score),
    )

    assert result == (score,)


def test_extract_transaction_ring_evidence():
    evidence = make_evidence()

    result = extract_transaction_ring_evidence(
        candidate_ring_ids=("ring-1",),
        ring_evidence=(evidence,),
    )

    assert result == (evidence,)


def test_summary_contains_prediction_context():
    summary = build_transaction_investigation_summary(
        transaction_id="txn-1",
        prediction_label=1,
        prediction_probability=0.82,
        suspicion_signal_count=2,
        related_entity_count=3,
        candidate_ring_count=1,
    )

    assert "txn-1" in summary
    assert "0.820000" in summary
    assert "positive" in summary
    assert "2 suspiciousness signal" in summary
    assert "3 directly related entity" in summary
    assert "1 candidate network group" in summary


def test_assemble_transaction_investigation_view():
    relationships = make_candidate().relationships
    candidate = make_candidate()
    score = make_score()
    evidence = make_evidence()

    view = assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        suspicious_transaction=make_suspicious_transaction(),
        relationships=relationships,
        candidates=(candidate,),
        network_scores=(score,),
        ring_evidence=(evidence,),
    )

    assert isinstance(
        view,
        TransactionInvestigationView,
    )

    assert view.transaction_id == "txn-1"
    assert view.prediction_probability == 0.82
    assert view.prediction_label == 1
    assert view.suspicion_signal_count == 2
    assert view.related_entity_count == 3
    assert view.relationship_count == 3
    assert view.candidate_ring_ids == ("ring-1",)
    assert view.network_scores == (score,)
    assert view.ring_evidence == (evidence,)


def test_transaction_id_mismatch_is_rejected():
    with pytest.raises(FraudIntelligenceContractError):
        assemble_transaction_investigation_view(
            transaction_id="txn-other",
            timestamp=TIMESTAMP,
            suspicious_transaction=make_suspicious_transaction(),
            relationships=(),
        )


def test_timestamp_mismatch_is_rejected():
    other_timestamp = datetime(
        2025,
        6,
        1,
        12,
        1,
        tzinfo=timezone.utc,
    )

    with pytest.raises(FraudIntelligenceContractError):
        assemble_transaction_investigation_view(
            transaction_id="txn-1",
            timestamp=other_timestamp,
            suspicious_transaction=make_suspicious_transaction(),
            relationships=(),
        )


def test_empty_relationships_are_allowed():
    view = assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        suspicious_transaction=make_suspicious_transaction(),
        relationships=(),
    )

    assert view.related_entity_count == 1
    assert view.candidate_ring_count == 0


def test_view_is_deterministic():
    candidate = make_candidate()

    relationships = candidate.relationships

    view_a = assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        suspicious_transaction=make_suspicious_transaction(),
        relationships=relationships,
        candidates=(candidate,),
        network_scores=(make_score(),),
        ring_evidence=(make_evidence(),),
    )

    view_b = assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        suspicious_transaction=make_suspicious_transaction(),
        relationships=tuple(reversed(relationships)),
        candidates=(candidate,),
        network_scores=(make_score(),),
        ring_evidence=(make_evidence(),),
    )

    # Relationship ordering itself is preserved by the view, so compare
    # the derived investigation context rather than raw relationship order.
    assert view_a.related_entities == view_b.related_entities
    assert view_a.candidate_ring_ids == view_b.candidate_ring_ids
    assert view_a.network_scores == view_b.network_scores
    assert view_a.ring_evidence == view_b.ring_evidence


def test_validation_accepts_valid_view():
    candidate = make_candidate()

    view = assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        suspicious_transaction=make_suspicious_transaction(),
        relationships=candidate.relationships,
        candidates=(candidate,),
        network_scores=(make_score(),),
        ring_evidence=(make_evidence(),),
    )

    validate_transaction_investigation_view(view)


def test_validation_rejects_invalid_probability():
    view = TransactionInvestigationView(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        prediction_probability=1.5,
        prediction_label=1,
        suspicion_signals=(),
        related_entities=(),
        relationships=(),
        candidate_ring_ids=(),
        network_scores=(),
        ring_evidence=(),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_transaction_investigation_view(view)


def test_validation_rejects_invalid_prediction_label():
    view = TransactionInvestigationView(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        prediction_probability=0.5,
        prediction_label=2,
        suspicion_signals=(),
        related_entities=(),
        relationships=(),
        candidate_ring_ids=(),
        network_scores=(),
        ring_evidence=(),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_transaction_investigation_view(view)


def test_validation_rejects_unsorted_ring_ids():
    view = TransactionInvestigationView(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        prediction_probability=0.5,
        prediction_label=1,
        suspicion_signals=(),
        related_entities=(),
        relationships=(),
        candidate_ring_ids=("ring-2", "ring-1"),
        network_scores=(),
        ring_evidence=(),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_transaction_investigation_view(view)


def test_validation_rejects_score_without_ring():
    view = TransactionInvestigationView(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        prediction_probability=0.5,
        prediction_label=1,
        suspicion_signals=(),
        related_entities=(),
        relationships=(),
        candidate_ring_ids=(),
        network_scores=(make_score(),),
        ring_evidence=(),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_transaction_investigation_view(view)


def test_validation_rejects_evidence_without_ring():
    view = TransactionInvestigationView(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        prediction_probability=0.5,
        prediction_label=1,
        suspicion_signals=(),
        related_entities=(),
        relationships=(),
        candidate_ring_ids=(),
        network_scores=(),
        ring_evidence=(make_evidence(),),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_transaction_investigation_view(view)


def test_validation_rejects_forbidden_summary_text():
    view = TransactionInvestigationView(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        prediction_probability=0.5,
        prediction_label=1,
        suspicion_signals=(),
        related_entities=(),
        relationships=(),
        candidate_ring_ids=(),
        network_scores=(),
        ring_evidence=(),
        investigation_summary="is_fraud is true.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_transaction_investigation_view(view)


def test_prediction_is_preserved_not_recomputed():
    suspicious = make_suspicious_transaction()

    view = assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        suspicious_transaction=suspicious,
        relationships=(),
    )

    assert view.prediction_probability == (
        suspicious.prediction_probability
    )

    assert view.prediction_label == (
        suspicious.prediction_label
    )


def test_ring_evidence_count():
    candidate = make_candidate()

    view = assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        suspicious_transaction=make_suspicious_transaction(),
        relationships=candidate.relationships,
        candidates=(candidate,),
        network_scores=(make_score(),),
        ring_evidence=(make_evidence(),),
    )

    assert view.evidence_count == 1


def test_temporal_rule_is_present():
    view = assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        suspicious_transaction=make_suspicious_transaction(),
        relationships=(),
    )

    assert "Future" in view.temporal_rule


def test_transaction_view_does_not_create_new_prediction():
    view = assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TIMESTAMP,
        suspicious_transaction=make_suspicious_transaction(),
        relationships=(),
    )

    # The view contains the existing prediction fields only.
    assert view.prediction_probability == 0.82
    assert view.prediction_label == 1