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
    build_investigation_graph,
    detect_connected_candidates,
    detect_fraud_ring_candidates,
    validate_candidate_detection_result,
    validate_fraud_ring_candidate,
)


TIMESTAMP = datetime(
    2025,
    1,
    1,
    10,
    0,
    tzinfo=timezone.utc,
)


def rel(
    relationship_type: str,
    source_type: str,
    source_id: str,
    target_type: str,
    target_id: str,
):
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


def test_build_investigation_graph():
    relationships = (
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

    graph = build_investigation_graph(relationships)

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


def test_connected_candidate_is_detected():
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

    result = detect_fraud_ring_candidates(
        target_transaction_id="txn-1",
        relationships=relationships,
        min_entities=3,
        min_shared_relationships=2,
    )

    assert result.candidate_count == 1

    candidate = result.candidates[0]

    assert candidate.entity_count == 3
    assert candidate.relationship_count == 3
    assert candidate.connected_component_size == 3


def test_target_transaction_is_required():
    relationships = (
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

    result = detect_fraud_ring_candidates(
        target_transaction_id="txn-missing",
        relationships=relationships,
        min_entities=2,
        min_shared_relationships=1,
    )

    assert result.candidate_count == 0


def test_small_component_is_filtered():
    relationships = (
        rel(
            "customer_transaction",
            "customer",
            "c1",
            "transaction",
            "txn-1",
        ),
    )

    result = detect_fraud_ring_candidates(
        target_transaction_id="txn-1",
        relationships=relationships,
        min_entities=3,
        min_shared_relationships=1,
    )

    assert result.candidate_count == 0


def test_insufficient_relationships_are_filtered():
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

    result = detect_fraud_ring_candidates(
        target_transaction_id="txn-1",
        relationships=relationships,
        min_entities=3,
        min_shared_relationships=3,
    )

    assert result.candidate_count == 0


def test_disconnected_component_is_not_returned_for_target():
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
            "c1",
            "device",
            "d1",
        ),
        rel(
            "customer_transaction",
            "customer",
            "c2",
            "transaction",
            "txn-2",
        ),
        rel(
            "customer_device",
            "customer",
            "c2",
            "device",
            "d2",
        ),
    )

    result = detect_fraud_ring_candidates(
        target_transaction_id="txn-1",
        relationships=relationships,
        min_entities=3,
        min_shared_relationships=2,
    )

    assert result.candidate_count == 1

    entity_ids = {
        entity.entity_id
        for entity in result.candidates[0].entities
    }

    assert "txn-1" in entity_ids
    assert "txn-2" not in entity_ids


def test_general_connected_candidate_detection():
    relationships = (
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
        rel(
            "customer_device",
            "customer",
            "c2",
            "device",
            "d1",
        ),
    )

    candidates = detect_connected_candidates(
        relationships=relationships,
        min_entities=3,
        min_relationships=2,
    )

    assert len(candidates) == 1
    assert candidates[0].entity_count == 4


def test_candidate_id_is_deterministic():
    relationships_a = (
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

    relationships_b = tuple(reversed(relationships_a))

    result_a = detect_connected_candidates(
        relationships=relationships_a,
        min_entities=3,
        min_relationships=2,
    )

    result_b = detect_connected_candidates(
        relationships=relationships_b,
        min_entities=3,
        min_relationships=2,
    )

    assert result_a[0].candidate_id == result_b[0].candidate_id


def test_candidate_contains_only_relationships_inside_component():
    relationships = (
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

    candidates = detect_connected_candidates(
        relationships=relationships,
        min_entities=3,
        min_relationships=2,
    )

    candidate = candidates[0]

    candidate_entity_ids = {
        entity.entity_id
        for entity in candidate.entities
    }

    for relationship in candidate.relationships:
        assert relationship.source.entity_id in candidate_entity_ids
        assert relationship.target.entity_id in candidate_entity_ids


def test_duplicate_relationships_do_not_create_duplicate_graph_edges():
    relationship = rel(
        "customer_device",
        "customer",
        "c1",
        "device",
        "d1",
    )

    graph = build_investigation_graph(
        (
            relationship,
            relationship,
        )
    )

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1

    edge_data = next(iter(graph.edges(data=True)))[2]

    assert len(edge_data["relationships"]) == 2


def test_candidate_validation_accepts_valid_candidate():
    relationships = (
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

    candidates = detect_connected_candidates(
        relationships=relationships,
        min_entities=3,
        min_relationships=2,
    )

    validate_fraud_ring_candidate(candidates[0])


def test_candidate_validation_rejects_wrong_component_size():
    relationships = (
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

    candidates = detect_connected_candidates(
        relationships=relationships,
        min_entities=3,
        min_relationships=2,
    )

    candidate = candidates[0]

    from fraud_intelligence.intelligence.fraud_ring_candidates import (
        FraudRingCandidate,
    )

    invalid = FraudRingCandidate(
        candidate_id=candidate.candidate_id,
        entities=candidate.entities,
        relationships=candidate.relationships,
        connected_component_size=999,
        shared_relationship_count=candidate.relationship_count,
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_fraud_ring_candidate(invalid)


def test_result_validation():
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

    result = detect_fraud_ring_candidates(
        target_transaction_id="txn-1",
        relationships=relationships,
        min_entities=3,
        min_shared_relationships=2,
    )

    validate_candidate_detection_result(result)


def test_empty_relationships_return_no_candidates():
    result = detect_fraud_ring_candidates(
        target_transaction_id="txn-1",
        relationships=(),
    )

    assert result.candidate_count == 0


def test_target_transaction_id_required():
    with pytest.raises(FraudIntelligenceContractError):
        detect_fraud_ring_candidates(
            target_transaction_id="",
            relationships=(),
        )


def test_min_entities_must_be_at_least_two():
    with pytest.raises(FraudIntelligenceContractError):
        detect_fraud_ring_candidates(
            target_transaction_id="txn-1",
            relationships=(),
            min_entities=1,
        )


def test_min_relationships_must_be_positive():
    with pytest.raises(FraudIntelligenceContractError):
        detect_fraud_ring_candidates(
            target_transaction_id="txn-1",
            relationships=(),
            min_shared_relationships=0,
        )


def test_candidate_order_is_deterministic():
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

    result_a = detect_fraud_ring_candidates(
        target_transaction_id="txn-1",
        relationships=relationships,
    )

    result_b = detect_fraud_ring_candidates(
        target_transaction_id="txn-1",
        relationships=tuple(reversed(relationships)),
    )

    assert tuple(
        candidate.candidate_id
        for candidate in result_a.candidates
    ) == tuple(
        candidate.candidate_id
        for candidate in result_b.candidates
    )