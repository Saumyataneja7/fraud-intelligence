from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fraud_intelligence.intelligence.contracts import (
    FraudIntelligenceContractError,
)
from fraud_intelligence.intelligence.entity_relationships import (
    EntityReference,
    EntityRelationship,
    EntityRelationshipIntelligence,
    assemble_entity_relationship_intelligence,
    build_direct_transaction_relationships,
    extract_entities_from_relationships,
    filter_temporal_relationships,
    validate_entity_relationship_intelligence,
    validate_relationship,
)


TARGET_TIMESTAMP = datetime(
    2025,
    6,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)

HISTORICAL_TIMESTAMP = TARGET_TIMESTAMP - timedelta(minutes=5)


def test_build_direct_transaction_relationships():
    relationships = build_direct_transaction_relationships(
        transaction_id="txn-1",
        transaction_timestamp=TARGET_TIMESTAMP,
        transaction={
            "customer_id": "customer-1",
            "account_id": "account-1",
            "card_id": "card-1",
            "merchant_id": "merchant-1",
            "device_id": "device-1",
            "ip_id": "ip-1",
        },
    )

    assert len(relationships) == 6

    relationship_types = {
        relationship.relationship_type
        for relationship in relationships
    }

    assert relationship_types == {
        "customer_transaction",
        "account_transaction",
        "card_transaction",
        "transaction_merchant",
        "transaction_device",
        "transaction_ip",
    }


def test_direct_relationship_endpoints_match_topology():
    relationships = build_direct_transaction_relationships(
        transaction_id="txn-1",
        transaction_timestamp=TARGET_TIMESTAMP,
        transaction={
            "customer_id": "customer-1",
            "account_id": "account-1",
            "card_id": "card-1",
            "merchant_id": "merchant-1",
            "device_id": "device-1",
            "ip_id": "ip-1",
        },
    )

    for relationship in relationships:
        validate_relationship(relationship)


def test_missing_optional_entity_does_not_create_relationship():
    relationships = build_direct_transaction_relationships(
        transaction_id="txn-1",
        transaction_timestamp=TARGET_TIMESTAMP,
        transaction={
            "customer_id": "customer-1",
            "account_id": "account-1",
            "merchant_id": "merchant-1",
        },
    )

    assert len(relationships) == 3

    types = {
        relationship.relationship_type
        for relationship in relationships
    }

    assert types == {
        "customer_transaction",
        "account_transaction",
        "transaction_merchant",
    }


def test_temporal_relationship_filter_allows_historical_context():
    relationship = EntityRelationship(
        relationship_type="customer_device",
        source=EntityReference("customer", "customer-1"),
        target=EntityReference("device", "device-1"),
        timestamp=HISTORICAL_TIMESTAMP,
    )

    filtered = filter_temporal_relationships(
        transaction_timestamp=TARGET_TIMESTAMP,
        relationships=[relationship],
    )

    assert filtered == (relationship,)


def test_same_timestamp_relationship_is_rejected():
    relationship = EntityRelationship(
        relationship_type="customer_device",
        source=EntityReference("customer", "customer-1"),
        target=EntityReference("device", "device-1"),
        timestamp=TARGET_TIMESTAMP,
    )

    with pytest.raises(FraudIntelligenceContractError):
        filter_temporal_relationships(
            transaction_timestamp=TARGET_TIMESTAMP,
            relationships=[relationship],
        )


def test_future_relationship_is_rejected():
    relationship = EntityRelationship(
        relationship_type="customer_device",
        source=EntityReference("customer", "customer-1"),
        target=EntityReference("device", "device-1"),
        timestamp=TARGET_TIMESTAMP + timedelta(seconds=1),
    )

    with pytest.raises(FraudIntelligenceContractError):
        filter_temporal_relationships(
            transaction_timestamp=TARGET_TIMESTAMP,
            relationships=[relationship],
        )


def test_untimestamped_static_relationship_is_allowed():
    relationship = EntityRelationship(
        relationship_type="customer_account",
        source=EntityReference("customer", "customer-1"),
        target=EntityReference("account", "account-1"),
        timestamp=None,
    )

    filtered = filter_temporal_relationships(
        transaction_timestamp=TARGET_TIMESTAMP,
        relationships=[relationship],
    )

    assert filtered == (relationship,)


def test_extract_entities_is_unique_and_deterministic():
    relationships = (
        EntityRelationship(
            relationship_type="customer_account",
            source=EntityReference("customer", "customer-2"),
            target=EntityReference("account", "account-2"),
        ),
        EntityRelationship(
            relationship_type="account_card",
            source=EntityReference("account", "account-2"),
            target=EntityReference("card", "card-1"),
        ),
        EntityRelationship(
            relationship_type="customer_device",
            source=EntityReference("customer", "customer-2"),
            target=EntityReference("device", "device-1"),
        ),
    )

    entities = extract_entities_from_relationships(relationships)

    assert entities == (
        EntityReference("account", "account-2"),
        EntityReference("card", "card-1"),
        EntityReference("customer", "customer-2"),
        EntityReference("device", "device-1"),
    )


def test_assembly_is_deterministic():
    relationships = (
        EntityRelationship(
            relationship_type="transaction_ip",
            source=EntityReference("transaction", "txn-1"),
            target=EntityReference("ip", "ip-1"),
            timestamp=HISTORICAL_TIMESTAMP,
        ),
        EntityRelationship(
            relationship_type="transaction_merchant",
            source=EntityReference("transaction", "txn-1"),
            target=EntityReference("merchant", "merchant-1"),
            timestamp=HISTORICAL_TIMESTAMP,
        ),
    )

    intelligence = assemble_entity_relationship_intelligence(
        transaction_id="txn-1",
        transaction_timestamp=TARGET_TIMESTAMP,
        relationships=relationships,
    )

    assert intelligence.transaction_id == "txn-1"
    assert intelligence.relationship_count == 2
    assert intelligence.entity_count == 3

    assert intelligence.relationships[0].relationship_type == (
        "transaction_ip"
    )
    assert intelligence.relationships[1].relationship_type == (
        "transaction_merchant"
    )


def test_entities_of_type():
    intelligence = assemble_entity_relationship_intelligence(
        transaction_id="txn-1",
        transaction_timestamp=TARGET_TIMESTAMP,
        relationships=(
            EntityRelationship(
                relationship_type="customer_transaction",
                source=EntityReference("customer", "customer-1"),
                target=EntityReference("transaction", "txn-1"),
                timestamp=HISTORICAL_TIMESTAMP,
            ),
            EntityRelationship(
                relationship_type="transaction_device",
                source=EntityReference("transaction", "txn-1"),
                target=EntityReference("device", "device-1"),
                timestamp=HISTORICAL_TIMESTAMP,
            ),
        ),
    )

    assert intelligence.entities_of_type("customer") == (
        EntityReference("customer", "customer-1"),
    )

    assert intelligence.entities_of_type("device") == (
        EntityReference("device", "device-1"),
    )


def test_unknown_entity_type_is_rejected():
    relationship = EntityRelationship(
        relationship_type="customer_device",
        source=EntityReference("alien", "alien-1"),
        target=EntityReference("device", "device-1"),
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_relationship(relationship)


def test_unknown_relationship_type_is_rejected():
    relationship = EntityRelationship(
        relationship_type="customer_alien",
        source=EntityReference("customer", "customer-1"),
        target=EntityReference("device", "device-1"),
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_relationship(relationship)


def test_invalid_relationship_source_type_is_rejected():
    relationship = EntityRelationship(
        relationship_type="customer_device",
        source=EntityReference("account", "account-1"),
        target=EntityReference("device", "device-1"),
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_relationship(relationship)


def test_invalid_relationship_target_type_is_rejected():
    relationship = EntityRelationship(
        relationship_type="customer_device",
        source=EntityReference("customer", "customer-1"),
        target=EntityReference("merchant", "merchant-1"),
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_relationship(relationship)


def test_duplicate_relationships_are_rejected():
    relationship = EntityRelationship(
        relationship_type="customer_device",
        source=EntityReference("customer", "customer-1"),
        target=EntityReference("device", "device-1"),
        timestamp=HISTORICAL_TIMESTAMP,
    )

    with pytest.raises(FraudIntelligenceContractError):
        assemble_entity_relationship_intelligence(
            transaction_id="txn-1",
            transaction_timestamp=TARGET_TIMESTAMP,
            relationships=(
                relationship,
                relationship,
            ),
        )


def test_completed_intelligence_validates():
    relationship = EntityRelationship(
        relationship_type="customer_device",
        source=EntityReference("customer", "customer-1"),
        target=EntityReference("device", "device-1"),
        timestamp=HISTORICAL_TIMESTAMP,
    )

    intelligence = assemble_entity_relationship_intelligence(
        transaction_id="txn-1",
        transaction_timestamp=TARGET_TIMESTAMP,
        relationships=(relationship,),
    )

    validate_entity_relationship_intelligence(intelligence)


def test_completed_intelligence_rejects_missing_entity():
    relationship = EntityRelationship(
        relationship_type="customer_device",
        source=EntityReference("customer", "customer-1"),
        target=EntityReference("device", "device-1"),
        timestamp=HISTORICAL_TIMESTAMP,
    )

    intelligence = EntityRelationshipIntelligence(
        transaction_id="txn-1",
        transaction_timestamp=TARGET_TIMESTAMP,
        entities=(
            EntityReference("customer", "customer-1"),
        ),
        relationships=(relationship,),
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_entity_relationship_intelligence(intelligence)


def test_empty_relationship_collection_is_valid():
    intelligence = assemble_entity_relationship_intelligence(
        transaction_id="txn-1",
        transaction_timestamp=TARGET_TIMESTAMP,
        relationships=(),
    )

    assert intelligence.entity_count == 0
    assert intelligence.relationship_count == 0


def test_non_empty_transaction_id_required():
    with pytest.raises(FraudIntelligenceContractError):
        assemble_entity_relationship_intelligence(
            transaction_id="",
            transaction_timestamp=TARGET_TIMESTAMP,
            relationships=(),
        )