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
from fraud_intelligence.intelligence.ring_investigation import (
    FraudRingInvestigationView,
    assemble_fraud_ring_investigation_view,
    assemble_fraud_ring_investigation_views,
    build_ring_investigation_summary,
    count_entity_types,
    count_relationship_types,
    extract_ring_transactions,
    select_ring_evidence,
    validate_fraud_ring_investigation_view,
)
from fraud_intelligence.intelligence.ring_network_scoring import (
    RingNetworkScore,
)


TIMESTAMP = datetime(
    2025,
    6,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)


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


def make_candidate(
    candidate_id: str = "ring-1",
) -> FraudRingCandidate:
    relationships = (
        relationship(
            "customer_transaction",
            "customer",
            "c1",
            "transaction",
            "txn-1",
            TIMESTAMP,
        ),
        relationship(
            "transaction_device",
            "transaction",
            "txn-1",
            "device",
            "d1",
            TIMESTAMP,
        ),
        relationship(
            "customer_device",
            "customer",
            "c1",
            "device",
            "d1",
            TIMESTAMP,
        ),
    )

    return FraudRingCandidate(
        candidate_id=candidate_id,
        entities=(
            EntityReference("customer", "c1"),
            EntityReference("device", "d1"),
            EntityReference("transaction", "txn-1"),
        ),
        relationships=relationships,
        connected_component_size=3,
        shared_relationship_count=3,
    )


def make_score(
    candidate_id: str = "ring-1",
) -> RingNetworkScore:
    return RingNetworkScore(
        candidate_id=candidate_id,
        entity_count=3,
        relationship_count=3,
        relationship_density=1.0,
        entity_type_count=3,
        relationship_type_count=3,
        non_transaction_entity_count=2,
        non_transaction_connectivity=1.0,
        structural_score=70.0,
    )


def make_evidence(
    candidate_id: str = "ring-1",
) -> FraudRingEvidence:
    candidate = make_candidate(candidate_id)
    score = make_score(candidate_id)

    return FraudRingEvidence(
        candidate_id=candidate_id,
        target_transaction_id="txn-1",
        entities=candidate.entities,
        relationships=candidate.relationships,
        network_score=score,
        evidence_items=(
            RingEvidenceItem(
                evidence_type="entity_connectivity",
                description="Connected entities share relationships.",
                entity_types=(
                    "customer",
                    "device",
                    "transaction",
                ),
                relationship_types=(
                    "customer_transaction",
                    "transaction_device",
                    "customer_device",
                ),
                strength="strong",
            ),
        ),
        temporal_context="Historical investigation context only.",
    )


def test_count_entity_types():
    entities = (
        EntityReference("customer", "c1"),
        EntityReference("device", "d1"),
        EntityReference("transaction", "txn-1"),
        EntityReference("customer", "c2"),
    )

    assert count_entity_types(entities) == (
        ("customer", 2),
        ("device", 1),
        ("transaction", 1),
    )


def test_count_relationship_types():
    relationships = (
        relationship(
            "customer_transaction",
            "customer",
            "c1",
            "transaction",
            "txn-1",
        ),
        relationship(
            "customer_transaction",
            "customer",
            "c2",
            "transaction",
            "txn-1",
        ),
        relationship(
            "transaction_device",
            "transaction",
            "txn-1",
            "device",
            "d1",
        ),
    )

    assert count_relationship_types(relationships) == (
        ("customer_transaction", 2),
        ("transaction_device", 1),
    )


def test_extract_ring_transactions():
    candidate = make_candidate()

    result = extract_ring_transactions(
        candidate.entities,
        candidate.relationships,
    )

    assert result[0].transaction_id == "txn-1"
    assert result[0].timestamp == TIMESTAMP


def test_extract_ring_transactions_without_timestamp():
    entities = (
        EntityReference("transaction", "txn-1"),
    )

    result = extract_ring_transactions(
        entities,
        (),
    )

    assert result == (
        (
            # Dataclass comparison through fields.
            result[0]
        ),
    )

    assert result[0].timestamp is None


def test_select_ring_evidence():
    evidence = make_evidence()

    result = select_ring_evidence(
        candidate_id="ring-1",
        evidence=(evidence,),
    )

    assert result == evidence


def test_select_missing_ring_evidence_returns_none():
    assert (
        select_ring_evidence(
            candidate_id="ring-1",
            evidence=(),
        )
        is None
    )


def test_duplicate_ring_evidence_is_rejected():
    evidence = make_evidence()

    with pytest.raises(FraudIntelligenceContractError):
        select_ring_evidence(
            candidate_id="ring-1",
            evidence=(evidence, evidence),
        )


def test_build_ring_summary():
    summary = build_ring_investigation_summary(
        candidate_id="ring-1",
        entity_count=3,
        relationship_count=3,
        transaction_count=1,
        entity_type_count=3,
        relationship_type_count=3,
        structural_score=70.0,
        evidence_count=1,
    )

    assert "ring-1" in summary
    assert "3 entities" in summary
    assert "3 relationship" in summary
    assert "1 transaction" in summary
    assert "70.000000" in summary
    assert "not a fraud probability" in summary


def test_assemble_ring_investigation_view():
    candidate = make_candidate()
    score = make_score()
    evidence = make_evidence()

    view = assemble_fraud_ring_investigation_view(
        candidate=candidate,
        network_score=score,
        evidence=evidence,
    )

    assert isinstance(
        view,
        FraudRingInvestigationView,
    )

    assert view.candidate_id == "ring-1"
    assert view.entity_count == 3
    assert view.relationship_count == 3
    assert view.transaction_count == 1
    assert view.network_score == score
    assert view.evidence == evidence
    assert view.evidence_count == 1


def test_assemble_without_evidence():
    candidate = make_candidate()
    score = make_score()

    view = assemble_fraud_ring_investigation_view(
        candidate=candidate,
        network_score=score,
    )

    assert view.evidence is None
    assert view.evidence_count == 0


def test_candidate_score_mismatch_is_rejected():
    candidate = make_candidate("ring-1")
    score = make_score("ring-2")

    with pytest.raises(FraudIntelligenceContractError):
        assemble_fraud_ring_investigation_view(
            candidate=candidate,
            network_score=score,
        )


def test_candidate_evidence_mismatch_is_rejected():
    candidate = make_candidate("ring-1")
    score = make_score("ring-1")
    evidence = make_evidence("ring-2")

    with pytest.raises(FraudIntelligenceContractError):
        assemble_fraud_ring_investigation_view(
            candidate=candidate,
            network_score=score,
            evidence=evidence,
        )


def test_batch_assembly():
    candidate_1 = make_candidate("ring-1")
    candidate_2 = make_candidate("ring-2")

    score_1 = make_score("ring-1")
    score_2 = make_score("ring-2")

    views = assemble_fraud_ring_investigation_views(
        candidates=(candidate_2, candidate_1),
        network_scores=(score_2, score_1),
        evidence=(
            make_evidence("ring-2"),
            make_evidence("ring-1"),
        ),
    )

    assert tuple(
        view.candidate_id
        for view in views
    ) == (
        "ring-1",
        "ring-2",
    )


def test_batch_requires_score_for_every_candidate():
    candidate = make_candidate("ring-1")

    with pytest.raises(FraudIntelligenceContractError):
        assemble_fraud_ring_investigation_views(
            candidates=(candidate,),
            network_scores=(),
        )


def test_batch_rejects_duplicate_scores():
    candidate = make_candidate("ring-1")
    score = make_score("ring-1")

    with pytest.raises(FraudIntelligenceContractError):
        assemble_fraud_ring_investigation_views(
            candidates=(candidate,),
            network_scores=(score, score),
        )


def test_view_validation_accepts_valid_view():
    view = assemble_fraud_ring_investigation_view(
        candidate=make_candidate(),
        network_score=make_score(),
        evidence=make_evidence(),
    )

    validate_fraud_ring_investigation_view(view)


def test_validation_rejects_score_candidate_mismatch():
    view = FraudRingInvestigationView(
        candidate_id="ring-1",
        entities=(),
        relationships=(),
        transactions=(),
        network_score=make_score("ring-2"),
        evidence=None,
        entity_type_counts=(),
        relationship_type_counts=(),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_fraud_ring_investigation_view(view)


def test_validation_rejects_relationship_outside_entity_set():
    candidate = make_candidate()

    bad_relationship = relationship(
        "customer_device",
        "customer",
        "not-in-ring",
        "device",
        "d1",
    )

    view = FraudRingInvestigationView(
        candidate_id="ring-1",
        entities=candidate.entities,
        relationships=(
            bad_relationship,
        ),
        transactions=(
            # Transaction entity still matches.
            extract_ring_transactions(
                candidate.entities,
                candidate.relationships,
            )[0],
        ),
        network_score=make_score(),
        evidence=None,
        entity_type_counts=count_entity_types(
            candidate.entities,
        ),
        relationship_type_counts=count_relationship_types(
            (bad_relationship,),
        ),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_fraud_ring_investigation_view(view)


def test_validation_rejects_transaction_reference_mismatch():
    candidate = make_candidate()

    view = FraudRingInvestigationView(
        candidate_id="ring-1",
        entities=candidate.entities,
        relationships=candidate.relationships,
        transactions=(),
        network_score=make_score(),
        evidence=None,
        entity_type_counts=count_entity_types(
            candidate.entities,
        ),
        relationship_type_counts=count_relationship_types(
            candidate.relationships,
        ),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_fraud_ring_investigation_view(view)


def test_validation_rejects_forbidden_text():
    candidate = make_candidate()

    view = FraudRingInvestigationView(
        candidate_id="ring-1",
        entities=candidate.entities,
        relationships=candidate.relationships,
        transactions=extract_ring_transactions(
            candidate.entities,
            candidate.relationships,
        ),
        network_score=make_score(),
        evidence=None,
        entity_type_counts=count_entity_types(
            candidate.entities,
        ),
        relationship_type_counts=count_relationship_types(
            candidate.relationships,
        ),
        investigation_summary="is_fraud appears here.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_fraud_ring_investigation_view(view)


def test_structural_score_is_preserved():
    view = assemble_fraud_ring_investigation_view(
        candidate=make_candidate(),
        network_score=make_score(),
    )

    assert view.network_score.structural_score == 70.0


def test_ring_view_does_not_create_fraud_probability():
    view = assemble_fraud_ring_investigation_view(
        candidate=make_candidate(),
        network_score=make_score(),
        evidence=make_evidence(),
    )

    assert not hasattr(view, "fraud_probability")
    assert not hasattr(view, "fraud_score")


def test_temporal_rule_is_present():
    view = assemble_fraud_ring_investigation_view(
        candidate=make_candidate(),
        network_score=make_score(),
    )

    assert "Future" in view.temporal_rule


def test_entity_and_relationship_counts_are_consistent():
    view = assemble_fraud_ring_investigation_view(
        candidate=make_candidate(),
        network_score=make_score(),
    )

    assert view.entity_type_counts == (
        ("customer", 1),
        ("device", 1),
        ("transaction", 1),
    )

    assert view.relationship_type_counts == (
        ("customer_device", 1),
        ("customer_transaction", 1),
        ("transaction_device", 1),
    )


def test_multiple_transactions_are_sorted():
    entities = (
        EntityReference("transaction", "txn-2"),
        EntityReference("transaction", "txn-1"),
    )

    relationships = (
        relationship(
            "transaction_device",
            "transaction",
            "txn-2",
            "device",
            "d2",
            TIMESTAMP,
        ),
        relationship(
            "transaction_device",
            "transaction",
            "txn-1",
            "device",
            "d1",
            TIMESTAMP,
        ),
    )

    result = extract_ring_transactions(
        entities,
        relationships,
    )

    assert tuple(
        item.transaction_id
        for item in result
    ) == (
        "txn-1",
        "txn-2",
    )