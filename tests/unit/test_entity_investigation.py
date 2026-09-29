from __future__ import annotations

import pytest

from fraud_intelligence.intelligence.contracts import (
    FraudIntelligenceContractError,
)
from fraud_intelligence.intelligence.entity_relationships import (
    EntityReference,
    EntityRelationship,
)
from fraud_intelligence.intelligence.entity_investigation import (
    EntityInvestigationView,
    RelatedEntity,
    assemble_entity_investigation_view,
    build_investigation_summary,
    extract_candidate_ring_ids,
    extract_network_scores,
    extract_related_entities,
    extract_related_transactions,
    validate_entity_investigation_view,
)
from fraud_intelligence.intelligence.fraud_ring_candidates import (
    FraudRingCandidate,
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
        source=EntityReference(source_type, source_id),
        target=EntityReference(target_type, target_id),
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


def test_extract_related_entities_incoming():
    entity = EntityReference("customer", "c1")

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
    )

    related = extract_related_entities(
        entity=entity,
        relationships=relationships,
    )

    assert len(related) == 2

    assert related[0].direction == "outgoing"


def test_extract_related_entities_outgoing():
    entity = EntityReference("transaction", "txn-1")

    relationships = (
        rel(
            "transaction_device",
            "transaction",
            "txn-1",
            "device",
            "d1",
        ),
    )

    related = extract_related_entities(
        entity=entity,
        relationships=relationships,
    )

    assert related == (
        RelatedEntity(
            entity=EntityReference("device", "d1"),
            relationship_type="transaction_device",
            direction="outgoing",
            timestamp=None,
        ),
    )


def test_unrelated_relationships_are_ignored():
    entity = EntityReference("customer", "c1")

    relationships = (
        rel(
            "customer_device",
            "customer",
            "c1",
            "device",
            "d1",
        ),
        rel(
            "customer_device",
            "customer",
            "c2",
            "device",
            "d2",
        ),
    )

    related = extract_related_entities(
        entity=entity,
        relationships=relationships,
    )

    assert len(related) == 1
    assert related[0].entity.entity_id == "d1"


def test_related_entities_are_deterministically_sorted():
    entity = EntityReference("customer", "c1")

    relationships = (
        rel(
            "customer_ip",
            "customer",
            "c1",
            "ip",
            "ip2",
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

    related = extract_related_entities(
        entity=entity,
        relationships=relationships,
    )

    assert tuple(
        item.entity.entity_id
        for item in related
    ) == (
        "d1",
        "ip1",
        "ip2",
    )


def test_extract_related_transactions():
    related_entities = (
        RelatedEntity(
            entity=EntityReference("transaction", "txn-2"),
            relationship_type="customer_transaction",
            direction="outgoing",
            timestamp=None,
        ),
        RelatedEntity(
            entity=EntityReference("device", "d1"),
            relationship_type="customer_device",
            direction="outgoing",
            timestamp=None,
        ),
        RelatedEntity(
            entity=EntityReference("transaction", "txn-1"),
            relationship_type="customer_transaction",
            direction="outgoing",
            timestamp=None,
        ),
    )

    transactions = extract_related_transactions(
        related_entities
    )

    assert transactions == (
        "txn-1",
        "txn-2",
    )


def test_extract_candidate_ring_ids():
    entity = EntityReference("customer", "c1")

    candidate = make_candidate()

    result = extract_candidate_ring_ids(
        entity=entity,
        candidates=(candidate,),
    )

    assert result == ("ring-1",)


def test_entity_not_in_candidate_has_no_ring():
    entity = EntityReference("customer", "c9")

    result = extract_candidate_ring_ids(
        entity=entity,
        candidates=(make_candidate(),),
    )

    assert result == ()


def test_extract_network_scores():
    scores = (
        RingNetworkScore(
            candidate_id="ring-2",
            entity_count=3,
            relationship_count=3,
            relationship_density=1.0,
            entity_type_count=3,
            relationship_type_count=3,
            non_transaction_entity_count=2,
            non_transaction_connectivity=1.0,
            structural_score=60.0,
        ),
        make_score(),
    )

    result = extract_network_scores(
        candidate_ring_ids=("ring-1",),
        network_scores=scores,
    )

    assert result == (make_score(),)


def test_network_scores_are_sorted():
    score_a = make_score()

    score_b = RingNetworkScore(
        candidate_id="ring-0",
        entity_count=3,
        relationship_count=2,
        relationship_density=0.5,
        entity_type_count=2,
        relationship_type_count=2,
        non_transaction_entity_count=2,
        non_transaction_connectivity=1.0,
        structural_score=50.0,
    )

    result = extract_network_scores(
        candidate_ring_ids=("ring-1", "ring-0"),
        network_scores=(score_a, score_b),
    )

    assert tuple(
        score.candidate_id
        for score in result
    ) == (
        "ring-0",
        "ring-1",
    )


def test_build_investigation_summary():
    entity = EntityReference("customer", "c1")

    summary = build_investigation_summary(
        entity=entity,
        related_entities=(
            RelatedEntity(
                entity=EntityReference("device", "d1"),
                relationship_type="customer_device",
                direction="outgoing",
                timestamp=None,
            ),
        ),
        related_transactions=("txn-1",),
        candidate_ring_ids=("ring-1",),
    )

    assert "customer c1" in summary
    assert "1 directly related entities" in summary
    assert "1 directly related transactions" in summary
    assert "1 candidate network group" in summary


def test_assemble_entity_investigation_view():
    entity = EntityReference("customer", "c1")

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
    )

    view = assemble_entity_investigation_view(
        entity=entity,
        relationships=relationships,
        candidates=(make_candidate(),),
        network_scores=(make_score(),),
    )

    assert isinstance(
        view,
        EntityInvestigationView,
    )

    assert view.entity == entity
    assert view.related_entity_count == 2
    assert view.related_transaction_count == 1
    assert view.candidate_ring_count == 1
    assert view.network_scores == (make_score(),)


def test_empty_relationships_create_valid_view():
    entity = EntityReference("merchant", "m1")

    view = assemble_entity_investigation_view(
        entity=entity,
        relationships=(),
    )

    assert view.related_entity_count == 0
    assert view.related_transaction_count == 0
    assert view.candidate_ring_count == 0


def test_view_is_deterministic():
    entity = EntityReference("customer", "c1")

    relationships = (
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
            "c1",
            "device",
            "d1",
        ),
    )

    view_a = assemble_entity_investigation_view(
        entity=entity,
        relationships=relationships,
    )

    view_b = assemble_entity_investigation_view(
        entity=entity,
        relationships=tuple(reversed(relationships)),
    )

    assert view_a == view_b


def test_view_validation_accepts_valid_view():
    entity = EntityReference("customer", "c1")

    view = assemble_entity_investigation_view(
        entity=entity,
        relationships=(
            rel(
                "customer_device",
                "customer",
                "c1",
                "device",
                "d1",
            ),
        ),
    )

    validate_entity_investigation_view(view)


def test_view_rejects_duplicate_related_entities():
    entity = EntityReference("customer", "c1")

    duplicate_related = RelatedEntity(
        entity=EntityReference("device", "d1"),
        relationship_type="customer_device",
        direction="outgoing",
        timestamp=None,
    )

    view = EntityInvestigationView(
        entity=entity,
        related_entities=(
            duplicate_related,
            duplicate_related,
        ),
        related_transactions=(),
        candidate_ring_ids=(),
        network_scores=(),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_entity_investigation_view(view)


def test_view_rejects_self_reference():
    entity = EntityReference("customer", "c1")

    view = EntityInvestigationView(
        entity=entity,
        related_entities=(
            RelatedEntity(
                entity=entity,
                relationship_type="customer_device",
                direction="outgoing",
                timestamp=None,
            ),
        ),
        related_transactions=(),
        candidate_ring_ids=(),
        network_scores=(),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_entity_investigation_view(view)


def test_view_rejects_invalid_direction():
    entity = EntityReference("customer", "c1")

    view = EntityInvestigationView(
        entity=entity,
        related_entities=(
            RelatedEntity(
                entity=EntityReference("device", "d1"),
                relationship_type="customer_device",
                direction="sideways",
                timestamp=None,
            ),
        ),
        related_transactions=(),
        candidate_ring_ids=(),
        network_scores=(),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_entity_investigation_view(view)


def test_view_rejects_unsorted_transactions():
    entity = EntityReference("customer", "c1")

    view = EntityInvestigationView(
        entity=entity,
        related_entities=(),
        related_transactions=("txn-2", "txn-1"),
        candidate_ring_ids=(),
        network_scores=(),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_entity_investigation_view(view)


def test_view_rejects_duplicate_transaction_ids():
    entity = EntityReference("customer", "c1")

    view = EntityInvestigationView(
        entity=entity,
        related_entities=(),
        related_transactions=("txn-1", "txn-1"),
        candidate_ring_ids=(),
        network_scores=(),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_entity_investigation_view(view)


def test_view_rejects_forbidden_summary_text():
    entity = EntityReference("customer", "c1")

    view = EntityInvestigationView(
        entity=entity,
        related_entities=(),
        related_transactions=(),
        candidate_ring_ids=(),
        network_scores=(),
        investigation_summary="is_fraud was detected.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_entity_investigation_view(view)


def test_view_rejects_score_without_candidate():
    entity = EntityReference("customer", "c1")

    view = EntityInvestigationView(
        entity=entity,
        related_entities=(),
        related_transactions=(),
        candidate_ring_ids=(),
        network_scores=(make_score(),),
        investigation_summary="Summary.",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_entity_investigation_view(view)


def test_view_requires_temporal_rule():
    entity = EntityReference("customer", "c1")

    view = EntityInvestigationView(
        entity=entity,
        related_entities=(),
        related_transactions=(),
        candidate_ring_ids=(),
        network_scores=(),
        investigation_summary="Summary.",
        temporal_rule="",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_entity_investigation_view(view)


def test_view_requires_summary():
    entity = EntityReference("customer", "c1")

    view = EntityInvestigationView(
        entity=entity,
        related_entities=(),
        related_transactions=(),
        candidate_ring_ids=(),
        network_scores=(),
        investigation_summary="",
        temporal_rule="Historical context only.",
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_entity_investigation_view(view)