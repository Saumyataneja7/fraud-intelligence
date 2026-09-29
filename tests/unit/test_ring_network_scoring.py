from __future__ import annotations

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
from fraud_intelligence.intelligence.ring_network_scoring import (
    RingNetworkScore,
    score_fraud_ring_candidate,
    score_fraud_ring_candidates,
    validate_ring_network_score,
    validate_ring_network_scores,
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


def test_score_candidate_returns_valid_score():
    candidate = make_candidate()

    score = score_fraud_ring_candidate(candidate)

    assert isinstance(score, RingNetworkScore)
    assert score.candidate_id == "ring-test"
    assert score.entity_count == 4
    assert score.relationship_count == 4
    assert 0.0 <= score.relationship_density <= 1.0
    assert 0.0 <= score.non_transaction_connectivity <= 1.0
    assert 0.0 <= score.structural_score <= 100.0


def test_relationship_density_is_calculated():
    candidate = make_candidate()

    score = score_fraud_ring_candidate(candidate)

    # Four entities -> C(4, 2) = 6 possible undirected pairs.
    assert score.relationship_density == round(4 / 6, 6)


def test_entity_type_count():
    candidate = make_candidate()

    score = score_fraud_ring_candidate(candidate)

    assert score.entity_type_count == 4


def test_relationship_type_count():
    candidate = make_candidate()

    score = score_fraud_ring_candidate(candidate)

    assert score.relationship_type_count == 4


def test_non_transaction_entity_count():
    candidate = make_candidate()

    score = score_fraud_ring_candidate(candidate)

    assert score.non_transaction_entity_count == 3


def test_non_transaction_connectivity():
    candidate = make_candidate()

    score = score_fraud_ring_candidate(candidate)

    # Every relationship touches at least one non-transaction entity.
    assert score.non_transaction_connectivity == 1.0


def test_score_is_deterministic():
    candidate = make_candidate()

    score_a = score_fraud_ring_candidate(candidate)
    score_b = score_fraud_ring_candidate(candidate)

    assert score_a == score_b


def test_batch_scoring_is_sorted_by_candidate_id():
    candidate_a = make_candidate()

    candidate_b = FraudRingCandidate(
        candidate_id="ring-a",
        entities=candidate_a.entities,
        relationships=candidate_a.relationships,
        connected_component_size=4,
        shared_relationship_count=4,
    )

    candidate_c = FraudRingCandidate(
        candidate_id="ring-z",
        entities=candidate_a.entities,
        relationships=candidate_a.relationships,
        connected_component_size=4,
        shared_relationship_count=4,
    )

    scores = score_fraud_ring_candidates(
        (
            candidate_c,
            candidate_a,
            candidate_b,
        )
    )

    assert tuple(
        score.candidate_id
        for score in scores
    ) == (
        "ring-a",
        "ring-test",
        "ring-z",
    )


def test_batch_scoring_empty_input():
    assert score_fraud_ring_candidates(()) == ()


def test_structural_score_is_not_a_probability():
    candidate = make_candidate()

    score = score_fraud_ring_candidate(candidate)

    assert isinstance(score.structural_score, float)
    assert 0.0 <= score.structural_score <= 100.0


def test_score_contains_no_fraud_label_fields():
    score = score_fraud_ring_candidate(
        make_candidate()
    )

    assert not hasattr(score, "is_fraud")
    assert not hasattr(score, "fraud_scenario")


def test_validation_accepts_valid_score():
    score = score_fraud_ring_candidate(
        make_candidate()
    )

    validate_ring_network_score(score)


def test_validation_accepts_valid_collection():
    scores = score_fraud_ring_candidates(
        (
            make_candidate(),
        )
    )

    validate_ring_network_scores(scores)


def test_validation_rejects_negative_entity_count():
    score = RingNetworkScore(
        candidate_id="ring-1",
        entity_count=-1,
        relationship_count=2,
        relationship_density=0.5,
        entity_type_count=2,
        relationship_type_count=2,
        non_transaction_entity_count=2,
        non_transaction_connectivity=1.0,
        structural_score=50.0,
    )

    try:
        validate_ring_network_score(score)
    except FraudIntelligenceContractError:
        return

    raise AssertionError(
        "Expected validation to reject negative entity count."
    )


def test_validation_rejects_density_above_one():
    score = RingNetworkScore(
        candidate_id="ring-1",
        entity_count=4,
        relationship_count=2,
        relationship_density=1.1,
        entity_type_count=2,
        relationship_type_count=2,
        non_transaction_entity_count=2,
        non_transaction_connectivity=1.0,
        structural_score=50.0,
    )

    try:
        validate_ring_network_score(score)
    except FraudIntelligenceContractError:
        return

    raise AssertionError(
        "Expected validation to reject invalid density."
    )


def test_validation_rejects_score_above_100():
    score = RingNetworkScore(
        candidate_id="ring-1",
        entity_count=4,
        relationship_count=2,
        relationship_density=0.5,
        entity_type_count=2,
        relationship_type_count=2,
        non_transaction_entity_count=2,
        non_transaction_connectivity=1.0,
        structural_score=101.0,
    )

    try:
        validate_ring_network_score(score)
    except FraudIntelligenceContractError:
        return

    raise AssertionError(
        "Expected validation to reject structural score above 100."
    )


def test_validation_rejects_duplicate_candidate_ids():
    score = score_fraud_ring_candidate(
        make_candidate()
    )

    try:
        validate_ring_network_scores(
            (
                score,
                score,
            )
        )
    except FraudIntelligenceContractError:
        return

    raise AssertionError(
        "Expected duplicate candidate IDs to be rejected."
    )


def test_larger_entity_count_increases_entity_component():
    candidate_small = FraudRingCandidate(
        candidate_id="small",
        entities=(
            EntityReference("customer", "c1"),
            EntityReference("device", "d1"),
            EntityReference("transaction", "t1"),
        ),
        relationships=(
            rel(
                "customer_transaction",
                "customer",
                "c1",
                "transaction",
                "t1",
            ),
            rel(
                "transaction_device",
                "transaction",
                "t1",
                "device",
                "d1",
            ),
        ),
        connected_component_size=3,
        shared_relationship_count=2,
    )

    candidate_large = FraudRingCandidate(
        candidate_id="large",
        entities=(
            EntityReference("customer", "c1"),
            EntityReference("device", "d1"),
            EntityReference("ip", "ip1"),
            EntityReference("transaction", "t1"),
        ),
        relationships=(
            rel(
                "customer_transaction",
                "customer",
                "c1",
                "transaction",
                "t1",
            ),
            rel(
                "transaction_device",
                "transaction",
                "t1",
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
        ),
        connected_component_size=4,
        shared_relationship_count=4,
    )

    small_score = score_fraud_ring_candidate(
        candidate_small
    )
    large_score = score_fraud_ring_candidate(
        candidate_large
    )

    assert large_score.entity_count > small_score.entity_count


def test_zero_relationship_candidate_has_zero_connectivity():
    candidate = FraudRingCandidate(
        candidate_id="ring-empty",
        entities=(
            EntityReference("customer", "c1"),
            EntityReference("device", "d1"),
        ),
        relationships=(),
        connected_component_size=2,
        shared_relationship_count=0,
    )

    score = score_fraud_ring_candidate(candidate)

    assert score.relationship_count == 0
    assert score.non_transaction_connectivity == 0.0