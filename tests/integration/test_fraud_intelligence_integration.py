from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fraud_intelligence.intelligence.entity_relationships import (
    EntityReference,
    EntityRelationship,
    assemble_entity_relationship_intelligence,
)
from fraud_intelligence.intelligence.entity_investigation import (
    assemble_entity_investigation_view,
)
from fraud_intelligence.intelligence.fraud_ring_candidates import (
    detect_fraud_ring_candidates,
)
from fraud_intelligence.intelligence.fraud_ring_evidence import (
    assemble_fraud_ring_evidence,
)
from fraud_intelligence.intelligence.ring_investigation import (
    assemble_fraud_ring_investigation_view,
)
from fraud_intelligence.intelligence.ring_network_scoring import (
    score_fraud_ring_candidate,
)
from fraud_intelligence.intelligence.suspicious_transaction import (
    SuspicionSignal,
    SuspiciousTransactionIntelligence,
)
from fraud_intelligence.intelligence.transaction_investigation import (
    assemble_transaction_investigation_view,
)
from fraud_intelligence.intelligence.validation import (
    validate_fraud_intelligence,
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


def relationship(
    relationship_type: str,
    source_type: str,
    source_id: str,
    target_type: str,
    target_id: str,
    timestamp=HISTORICAL_TIME,
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
        timestamp=timestamp,
    )


def make_relationships() -> tuple[EntityRelationship, ...]:
    """
    Small deterministic historical graph representing:

        customer
          ├── account
          ├── device
          └── transaction
                  └── merchant

    plus a shared device relationship with another customer.
    """

    return (
        relationship(
            "customer_account",
            "customer",
            "c1",
            "account",
            "a1",
        ),
        relationship(
            "account_card",
            "account",
            "a1",
            "card",
            "card1",
        ),
        relationship(
            "customer_transaction",
            "customer",
            "c1",
            "transaction",
            "txn-1",
        ),
        relationship(
            "account_transaction",
            "account",
            "a1",
            "transaction",
            "txn-1",
        ),
        relationship(
            "card_transaction",
            "card",
            "card1",
            "transaction",
            "txn-1",
        ),
        relationship(
            "customer_device",
            "customer",
            "c1",
            "device",
            "d1",
        ),
        relationship(
            "customer_ip",
            "customer",
            "c1",
            "ip",
            "ip1",
        ),
        relationship(
            "customer_merchant",
            "customer",
            "c1",
            "merchant",
            "m1",
        ),
        relationship(
            "transaction_merchant",
            "transaction",
            "txn-1",
            "merchant",
            "m1",
        ),
        relationship(
            "transaction_device",
            "transaction",
            "txn-1",
            "device",
            "d1",
        ),
        relationship(
            "transaction_ip",
            "transaction",
            "txn-1",
            "ip",
            "ip1",
        ),
        relationship(
            "customer_device",
            "customer",
            "c2",
            "device",
            "d1",
        ),
    )


def make_suspicious_transaction() -> (
    SuspiciousTransactionIntelligence
):
    return SuspiciousTransactionIntelligence(
        transaction_id="txn-1",
        timestamp=TARGET_TIME,
        prediction_probability=0.91,
        prediction_label=1,
        signals=(
            SuspicionSignal(
                name="velocity",
                value=4,
                description="Historical transaction velocity.",
                source="engineered_feature",
            ),
            SuspicionSignal(
                name="is_new_device",
                value=True,
                description="Device is new for the customer.",
                source="engineered_feature",
            ),
        ),
        entity_counts=(
            ("customer", 1),
            ("account", 1),
            ("card", 1),
            ("device", 1),
            ("ip", 1),
            ("merchant", 1),
        ),
    )


def build_candidate(
    relationships: tuple[EntityRelationship, ...],
):
    result = detect_fraud_ring_candidates(
        target_transaction_id="txn-1",
        relationships=relationships,
    )

    assert result.candidates

    return result.candidates[0]


def test_end_to_end_transaction_intelligence_flow():
    relationships = make_relationships()

    suspicious = make_suspicious_transaction()

    relationship_intelligence = (
        assemble_entity_relationship_intelligence(
            transaction_id="txn-1",
            transaction_timestamp=TARGET_TIME,
            relationships=relationships,
        )
    )

    assert relationship_intelligence.transaction_id == "txn-1"

    candidate = build_candidate(
        relationship_intelligence.relationships
    )

    network_score = score_fraud_ring_candidate(
        candidate
    )

    ring_evidence = assemble_fraud_ring_evidence(
        candidate=candidate,
        network_score=network_score,
        target_transaction_id="txn-1",
    )

    transaction_view = (
        assemble_transaction_investigation_view(
            transaction_id="txn-1",
            timestamp=TARGET_TIME,
            suspicious_transaction=suspicious,
            relationships=relationship_intelligence.relationships,
            candidates=(candidate,),
            network_scores=(network_score,),
            ring_evidence=(ring_evidence,),
        )
    )

    assert transaction_view.transaction_id == "txn-1"
    assert transaction_view.prediction_probability == 0.91
    assert transaction_view.candidate_ring_ids

    ring_view = assemble_fraud_ring_investigation_view(
        candidate=candidate,
        network_score=network_score,
        evidence=ring_evidence,
    )

    assert ring_view.candidate_id == candidate.candidate_id
    assert ring_view.entity_count == len(
        candidate.entities
    )
    assert ring_view.relationship_count == len(
        candidate.relationships
    )

    validation = validate_fraud_intelligence(
        transaction_intelligence=suspicious,
        transaction_view=transaction_view,
        ring_evidence=(ring_evidence,),
        ring_views=(ring_view,),
        target_timestamp=TARGET_TIME,
    )

    assert validation.passed is True
    assert validation.issue_count == 0


def test_transaction_and_ring_share_same_candidate():
    relationships = make_relationships()

    suspicious = make_suspicious_transaction()

    relationship_intelligence = (
        assemble_entity_relationship_intelligence(
            transaction_id="txn-1",
            transaction_timestamp=TARGET_TIME,
            relationships=relationships,
        )
    )

    candidate = build_candidate(
        relationship_intelligence.relationships
    )

    network_score = score_fraud_ring_candidate(candidate)

    evidence = assemble_fraud_ring_evidence(
        candidate=candidate,
        network_score=network_score,
        target_transaction_id="txn-1",
    )

    transaction_view = (
        assemble_transaction_investigation_view(
            transaction_id="txn-1",
            timestamp=TARGET_TIME,
            suspicious_transaction=suspicious,
            relationships=relationship_intelligence.relationships,
            candidates=(candidate,),
            network_scores=(network_score,),
            ring_evidence=(evidence,),
        )
    )

    ring_view = assemble_fraud_ring_investigation_view(
        candidate=candidate,
        network_score=network_score,
        evidence=evidence,
    )

    assert transaction_view.candidate_ring_ids == (
        ring_view.candidate_id,
    )

    assert transaction_view.network_scores == (
        ring_view.network_score,
    )


def test_ring_evidence_matches_ring_investigation():
    relationships = make_relationships()

    relationship_intelligence = (
        assemble_entity_relationship_intelligence(
            transaction_id="txn-1",
            transaction_timestamp=TARGET_TIME,
            relationships=relationships,
        )
    )

    candidate = build_candidate(
        relationship_intelligence.relationships
    )

    network_score = score_fraud_ring_candidate(candidate)

    evidence = assemble_fraud_ring_evidence(
        candidate=candidate,
        network_score=network_score,
        target_transaction_id="txn-1",
    )

    ring_view = assemble_fraud_ring_investigation_view(
        candidate=candidate,
        network_score=network_score,
        evidence=evidence,
    )

    assert ring_view.evidence is evidence
    assert ring_view.evidence.candidate_id == (
        ring_view.candidate_id
    )
    assert ring_view.evidence.network_score == (
        ring_view.network_score
    )


def test_historical_relationships_flow_through_entire_pipeline():
    relationships = make_relationships()

    relationship_intelligence = (
        assemble_entity_relationship_intelligence(
            transaction_id="txn-1",
            transaction_timestamp=TARGET_TIME,
            relationships=relationships,
        )
    )

    assert relationship_intelligence.relationships

    for item in relationship_intelligence.relationships:
        if item.timestamp is not None:
            assert item.timestamp < TARGET_TIME

    candidate = build_candidate(
        relationship_intelligence.relationships
    )

    for item in candidate.relationships:
        if item.timestamp is not None:
            assert item.timestamp < TARGET_TIME


def test_no_prediction_is_recomputed_during_investigation():
    relationships = make_relationships()

    suspicious = make_suspicious_transaction()

    relationship_intelligence = (
        assemble_entity_relationship_intelligence(
            transaction_id="txn-1",
            transaction_timestamp=TARGET_TIME,
            relationships=relationships,
        )
    )

    candidate = build_candidate(
        relationship_intelligence.relationships
    )

    network_score = score_fraud_ring_candidate(candidate)

    evidence = assemble_fraud_ring_evidence(
        candidate=candidate,
        network_score=network_score,
        target_transaction_id="txn-1",
    )

    view = assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TARGET_TIME,
        suspicious_transaction=suspicious,
        relationships=relationship_intelligence.relationships,
        candidates=(candidate,),
        network_scores=(network_score,),
        ring_evidence=(evidence,),
    )

    assert view.prediction_probability == (
        suspicious.prediction_probability
    )
    assert view.prediction_label == (
        suspicious.prediction_label
    )


def test_entity_investigation_can_consume_same_relationship_context():
    relationships = make_relationships()

    relationship_intelligence = (
        assemble_entity_relationship_intelligence(
            transaction_id="txn-1",
            transaction_timestamp=TARGET_TIME,
            relationships=relationships,
        )
    )

    candidate = build_candidate(
        relationship_intelligence.relationships
    )

    network_score = score_fraud_ring_candidate(candidate)

    entity_view = assemble_entity_investigation_view(
        entity=EntityReference(
            "customer",
            "c1",
        ),
        relationships=relationship_intelligence.relationships,
        candidates=(candidate,),
        network_scores=(network_score,),
    )

    assert entity_view.entity == EntityReference(
        "customer",
        "c1",
    )

    assert entity_view.related_entities
    assert entity_view.related_transactions


def test_investigation_pipeline_is_deterministic():
    relationships = make_relationships()

    suspicious = make_suspicious_transaction()

    first_relationships = (
        assemble_entity_relationship_intelligence(
            transaction_id="txn-1",
            transaction_timestamp=TARGET_TIME,
            relationships=relationships,
        )
    )

    second_relationships = (
        assemble_entity_relationship_intelligence(
            transaction_id="txn-1",
            transaction_timestamp=TARGET_TIME,
            relationships=tuple(reversed(relationships)),
        )
    )

    assert first_relationships == second_relationships

    first_candidate = build_candidate(
        first_relationships.relationships
    )

    second_candidate = build_candidate(
        second_relationships.relationships
    )

    assert first_candidate == second_candidate

    first_score = score_fraud_ring_candidate(first_candidate)
    second_score = score_fraud_ring_candidate(second_candidate)

    assert first_score == second_score

    first_evidence = assemble_fraud_ring_evidence(
        candidate=first_candidate,
        network_score=first_score,
        target_transaction_id="txn-1",
    )

    second_evidence = assemble_fraud_ring_evidence(
        candidate=second_candidate,
        network_score=second_score,
        target_transaction_id="txn-1",
    )

    assert first_evidence == second_evidence

    first_view = assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TARGET_TIME,
        suspicious_transaction=suspicious,
        relationships=first_relationships.relationships,
        candidates=(first_candidate,),
        network_scores=(first_score,),
        ring_evidence=(first_evidence,),
    )

    second_view = assemble_transaction_investigation_view(
        transaction_id="txn-1",
        timestamp=TARGET_TIME,
        suspicious_transaction=suspicious,
        relationships=second_relationships.relationships,
        candidates=(second_candidate,),
        network_scores=(second_score,),
        ring_evidence=(second_evidence,),
    )

    assert first_view.related_entities == (
        second_view.related_entities
    )

    assert first_view.candidate_ring_ids == (
        second_view.candidate_ring_ids
    )

    assert first_view.network_scores == (
        second_view.network_scores
    )


def test_complete_pipeline_has_no_validation_issues():
    relationships = make_relationships()

    suspicious = make_suspicious_transaction()

    relationship_intelligence = (
        assemble_entity_relationship_intelligence(
            transaction_id="txn-1",
            transaction_timestamp=TARGET_TIME,
            relationships=relationships,
        )
    )

    candidate = build_candidate(
        relationship_intelligence.relationships
    )

    network_score = score_fraud_ring_candidate(candidate)

    evidence = assemble_fraud_ring_evidence(
        candidate=candidate,
        network_score=network_score,
        target_transaction_id="txn-1",
    )

    transaction_view = (
        assemble_transaction_investigation_view(
            transaction_id="txn-1",
            timestamp=TARGET_TIME,
            suspicious_transaction=suspicious,
            relationships=relationship_intelligence.relationships,
            candidates=(candidate,),
            network_scores=(network_score,),
            ring_evidence=(evidence,),
        )
    )

    ring_view = assemble_fraud_ring_investigation_view(
        candidate=candidate,
        network_score=network_score,
        evidence=evidence,
    )

    report = validate_fraud_intelligence(
        transaction_intelligence=suspicious,
        transaction_view=transaction_view,
        ring_evidence=(evidence,),
        ring_views=(ring_view,),
        target_timestamp=TARGET_TIME,
    )

    assert report.passed
    assert report.issues == ()