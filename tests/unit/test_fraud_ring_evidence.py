from __future__ import annotations

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
    assemble_fraud_ring_evidence,
    build_ring_evidence_items,
    validate_fraud_ring_evidence,
    validate_ring_evidence_item,
)
from fraud_intelligence.intelligence.ring_network_scoring import (
    RingNetworkScore,
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
        source=EntityReference(
            source_type,
            source_id,
        ),
        target=EntityReference(
            target_type,
            target_id,
        ),
        timestamp=None,
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
        rel(
            "customer_ip",
            "customer",
            "c1",
            "ip",
            "ip1",
        ),
    )

    entities = (
        EntityReference("customer", "c1"),
        EntityReference("device", "d1"),
        EntityReference("ip", "ip1"),
        EntityReference("transaction", "txn-1"),
    )

    return FraudRingCandidate(
        candidate_id="ring-test",
        entities=entities,
        relationships=relationships,
        connected_component_size=4,
        shared_relationship_count=4,
    )


def make_score() -> RingNetworkScore:
    return RingNetworkScore(
        candidate_id="ring-test",
        entity_count=4,
        relationship_count=4,
        relationship_density=0.666667,
        entity_type_count=4,
        relationship_type_count=4,
        non_transaction_entity_count=3,
        non_transaction_connectivity=1.0,
        structural_score=72.5,
    )


def test_build_ring_evidence_items():
    items = build_ring_evidence_items(
        candidate=make_candidate(),
        network_score=make_score(),
    )

    assert len(items) > 0

    evidence_types = {
        item.evidence_type
        for item in items
    }

    assert "entity_connectivity" in evidence_types
    assert "relationship_richness" in evidence_types
    assert "entity_type_diversity" in evidence_types
    assert "relationship_type_diversity" in evidence_types


def test_evidence_items_are_deterministically_ordered():
    candidate = make_candidate()
    score = make_score()

    items_a = build_ring_evidence_items(
        candidate=candidate,
        network_score=score,
    )

    items_b = build_ring_evidence_items(
        candidate=candidate,
        network_score=score,
    )

    assert items_a == items_b


def test_evidence_descriptions_are_present():
    items = build_ring_evidence_items(
        candidate=make_candidate(),
        network_score=make_score(),
    )

    assert all(
        item.description
        for item in items
    )


def test_evidence_strength_values_are_valid():
    items = build_ring_evidence_items(
        candidate=make_candidate(),
        network_score=make_score(),
    )

    assert all(
        item.strength in {
            "limited",
            "moderate",
            "strong",
        }
        for item in items
    )


def test_assemble_fraud_ring_evidence():
    evidence = assemble_fraud_ring_evidence(
        candidate=make_candidate(),
        network_score=make_score(),
        target_transaction_id="txn-1",
    )

    assert isinstance(
        evidence,
        FraudRingEvidence,
    )

    assert evidence.candidate_id == "ring-test"
    assert evidence.target_transaction_id == "txn-1"
    assert evidence.entity_count == 4
    assert evidence.relationship_count == 4
    assert evidence.evidence_count > 0


def test_target_transaction_must_belong_to_candidate():
    with pytest.raises(FraudIntelligenceContractError):
        assemble_fraud_ring_evidence(
            candidate=make_candidate(),
            network_score=make_score(),
            target_transaction_id="txn-other",
        )


def test_candidate_and_score_ids_must_match():
    score = RingNetworkScore(
        candidate_id="different-ring",
        entity_count=4,
        relationship_count=4,
        relationship_density=0.5,
        entity_type_count=4,
        relationship_type_count=4,
        non_transaction_entity_count=3,
        non_transaction_connectivity=1.0,
        structural_score=70.0,
    )

    with pytest.raises(FraudIntelligenceContractError):
        assemble_fraud_ring_evidence(
            candidate=make_candidate(),
            network_score=score,
            target_transaction_id="txn-1",
        )


def test_empty_target_transaction_id_is_rejected():
    with pytest.raises(FraudIntelligenceContractError):
        assemble_fraud_ring_evidence(
            candidate=make_candidate(),
            network_score=make_score(),
            target_transaction_id="",
        )


def test_temporal_context_is_present():
    evidence = assemble_fraud_ring_evidence(
        candidate=make_candidate(),
        network_score=make_score(),
        target_transaction_id="txn-1",
    )

    assert "Future" in evidence.temporal_context


def test_evidence_does_not_use_fraud_labels():
    evidence = assemble_fraud_ring_evidence(
        candidate=make_candidate(),
        network_score=make_score(),
        target_transaction_id="txn-1",
    )

    for item in evidence.evidence_items:
        assert "is_fraud" not in item.description
        assert "fraud_scenario" not in item.description


def test_validation_accepts_valid_evidence():
    evidence = assemble_fraud_ring_evidence(
        candidate=make_candidate(),
        network_score=make_score(),
        target_transaction_id="txn-1",
    )

    validate_fraud_ring_evidence(evidence)


def test_validation_rejects_duplicate_evidence_types():
    item = RingEvidenceItem(
        evidence_type="entity_connectivity",
        description="Evidence",
        entity_types=("customer",),
        relationship_types=("customer_device",),
        strength="moderate",
    )

    evidence = FraudRingEvidence(
        candidate_id="ring-test",
        target_transaction_id="txn-1",
        entities=make_candidate().entities,
        relationships=make_candidate().relationships,
        network_score=make_score(),
        evidence_items=(item, item),
        temporal_context="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_fraud_ring_evidence(evidence)


def test_validation_rejects_forbidden_evidence_text():
    item = RingEvidenceItem(
        evidence_type="test",
        description="Evidence contains is_fraud",
        entity_types=("customer",),
        relationship_types=("customer_device",),
        strength="moderate",
    )

    evidence = FraudRingEvidence(
        candidate_id="ring-test",
        target_transaction_id="txn-1",
        entities=make_candidate().entities,
        relationships=make_candidate().relationships,
        network_score=make_score(),
        evidence_items=(item,),
        temporal_context="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_fraud_ring_evidence(evidence)


def test_evidence_item_validation_accepts_valid_item():
    item = RingEvidenceItem(
        evidence_type="entity_connectivity",
        description="Multiple entities are connected.",
        entity_types=("customer", "device"),
        relationship_types=("customer_device",),
        strength="moderate",
    )

    validate_ring_evidence_item(item)


def test_invalid_evidence_strength_is_rejected():
    item = RingEvidenceItem(
        evidence_type="entity_connectivity",
        description="Multiple entities are connected.",
        entity_types=("customer", "device"),
        relationship_types=("customer_device",),
        strength="extreme",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_ring_evidence_item(item)


def test_empty_evidence_type_is_rejected():
    item = RingEvidenceItem(
        evidence_type="",
        description="Multiple entities are connected.",
        entity_types=("customer",),
        relationship_types=("customer_device",),
        strength="moderate",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_ring_evidence_item(item)


def test_empty_description_is_rejected():
    item = RingEvidenceItem(
        evidence_type="entity_connectivity",
        description="",
        entity_types=("customer",),
        relationship_types=("customer_device",),
        strength="moderate",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_ring_evidence_item(item)


def test_forbidden_entity_type_text_is_rejected():
    item = RingEvidenceItem(
        evidence_type="test",
        description="Valid evidence",
        entity_types=("is_fraud",),
        relationship_types=(),
        strength="limited",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_ring_evidence_item(item)


def test_forbidden_relationship_type_text_is_rejected():
    item = RingEvidenceItem(
        evidence_type="test",
        description="Valid evidence",
        entity_types=(),
        relationship_types=("fraud_scenario",),
        strength="limited",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_ring_evidence_item(item)


def test_network_score_is_preserved():
    score = make_score()

    evidence = assemble_fraud_ring_evidence(
        candidate=make_candidate(),
        network_score=score,
        target_transaction_id="txn-1",
    )

    assert evidence.network_score == score


def test_candidate_entities_are_preserved():
    candidate = make_candidate()

    evidence = assemble_fraud_ring_evidence(
        candidate=candidate,
        network_score=make_score(),
        target_transaction_id="txn-1",
    )

    assert evidence.entities == candidate.entities


def test_candidate_relationships_are_preserved():
    candidate = make_candidate()

    evidence = assemble_fraud_ring_evidence(
        candidate=candidate,
        network_score=make_score(),
        target_transaction_id="txn-1",
    )

    assert evidence.relationships == candidate.relationships