from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .contracts import FraudIntelligenceContractError


# ---------------------------------------------------------------------------
# Canonical entity / relationship types
# ---------------------------------------------------------------------------

ENTITY_TYPES: tuple[str, ...] = (
    "customer",
    "account",
    "card",
    "transaction",
    "merchant",
    "device",
    "ip",
)

RELATIONSHIP_TYPES: tuple[str, ...] = (
    "customer_account",
    "account_card",
    "customer_transaction",
    "account_transaction",
    "card_transaction",
    "customer_device",
    "customer_ip",
    "customer_merchant",
    "transaction_merchant",
    "transaction_device",
    "transaction_ip",
)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityReference:
    """
    A single entity connected to the target transaction.
    """

    entity_type: str
    entity_id: str


@dataclass(frozen=True)
class EntityRelationship:
    """
    A graph relationship relevant to the target transaction.

    source and target are entity references from the frozen Phase 6
    topology.
    """

    relationship_type: str
    source: EntityReference
    target: EntityReference
    timestamp: object | None = None


@dataclass(frozen=True)
class EntityRelationshipIntelligence:
    """
    Complete relationship context for one target transaction.
    """

    transaction_id: str
    transaction_timestamp: object
    entities: tuple[EntityReference, ...]
    relationships: tuple[EntityRelationship, ...]

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)

    def entities_of_type(
        self,
        entity_type: str,
    ) -> tuple[EntityReference, ...]:
        return tuple(
            entity
            for entity in self.entities
            if entity.entity_type == entity_type
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_entity_type(entity_type: str) -> None:
    if entity_type not in ENTITY_TYPES:
        raise FraudIntelligenceContractError(
            f"Unknown entity type: {entity_type!r}"
        )


def validate_relationship_type(relationship_type: str) -> None:
    if relationship_type not in RELATIONSHIP_TYPES:
        raise FraudIntelligenceContractError(
            f"Unknown relationship type: {relationship_type!r}"
        )


def validate_entity_reference(
    entity: EntityReference,
) -> None:
    validate_entity_type(entity.entity_type)

    if not entity.entity_id:
        raise FraudIntelligenceContractError(
            "Entity ID must be non-empty."
        )


def validate_relationship(
    relationship: EntityRelationship,
) -> None:
    validate_relationship_type(
        relationship.relationship_type
    )

    validate_entity_reference(relationship.source)
    validate_entity_reference(relationship.target)

    # Validate that the relationship's endpoint types match
    # the canonical Phase 6 relationship topology.
    expected = {
        "customer_account": ("customer", "account"),
        "account_card": ("account", "card"),
        "customer_transaction": ("customer", "transaction"),
        "account_transaction": ("account", "transaction"),
        "card_transaction": ("card", "transaction"),
        "customer_device": ("customer", "device"),
        "customer_ip": ("customer", "ip"),
        "customer_merchant": ("customer", "merchant"),
        "transaction_merchant": ("transaction", "merchant"),
        "transaction_device": ("transaction", "device"),
        "transaction_ip": ("transaction", "ip"),
    }

    expected_source, expected_target = expected[
        relationship.relationship_type
    ]

    if relationship.source.entity_type != expected_source:
        raise FraudIntelligenceContractError(
            f"Relationship {relationship.relationship_type!r} "
            f"requires source type {expected_source!r}, got "
            f"{relationship.source.entity_type!r}."
        )

    if relationship.target.entity_type != expected_target:
        raise FraudIntelligenceContractError(
            f"Relationship {relationship.relationship_type!r} "
            f"requires target type {expected_target!r}, got "
            f"{relationship.target.entity_type!r}."
        )


def validate_temporal_relationship(
    *,
    transaction_timestamp,
    relationship_timestamp,
) -> None:
    """
    Relationship evidence must strictly precede the target transaction.

    Same-timestamp context is rejected because the project does not
    define an event ordering for identical timestamps.
    """

    if relationship_timestamp is None:
        return

    if relationship_timestamp >= transaction_timestamp:
        raise FraudIntelligenceContractError(
            "Relationship context must be strictly earlier than "
            "the target transaction timestamp."
        )


# ---------------------------------------------------------------------------
# Relationship construction
# ---------------------------------------------------------------------------

def build_direct_transaction_relationships(
    *,
    transaction_id: str,
    transaction_timestamp,
    transaction: Mapping[str, object],
) -> tuple[EntityRelationship, ...]:
    """
    Construct the canonical direct relationships for a transaction.

    These relationships come directly from the transaction's entity
    identifiers and mirror the frozen Phase 6 graph topology.

    Expected fields:

    customer_id
    account_id
    card_id
    merchant_id
    device_id
    ip_id

    The transaction itself is represented as the target endpoint.

    Direct transaction relationships are structural identity links.
    They do not introduce any fraud label or predictive information.
    """

    if not transaction_id:
        raise FraudIntelligenceContractError(
            "transaction_id must be non-empty."
        )

    relationships: list[EntityRelationship] = []

    transaction_ref = EntityReference(
        entity_type="transaction",
        entity_id=str(transaction_id),
    )

    relationship_fields = (
        (
            "customer_id",
            "customer_transaction",
            "customer",
        ),
        (
            "account_id",
            "account_transaction",
            "account",
        ),
        (
            "card_id",
            "card_transaction",
            "card",
        ),
    )

    for field, relationship_type, source_type in relationship_fields:
        entity_id = transaction.get(field)

        if entity_id is None:
            continue

        relationships.append(
            EntityRelationship(
                relationship_type=relationship_type,
                source=EntityReference(
                    entity_type=source_type,
                    entity_id=str(entity_id),
                ),
                target=transaction_ref,
                timestamp=transaction_timestamp,
            )
        )

    # Transaction -> merchant/device/IP relationships.
    for field, relationship_type, target_type in (
        (
            "merchant_id",
            "transaction_merchant",
            "merchant",
        ),
        (
            "device_id",
            "transaction_device",
            "device",
        ),
        (
            "ip_id",
            "transaction_ip",
            "ip",
        ),
    ):
        entity_id = transaction.get(field)

        if entity_id is None:
            continue

        relationships.append(
            EntityRelationship(
                relationship_type=relationship_type,
                source=transaction_ref,
                target=EntityReference(
                    entity_type=target_type,
                    entity_id=str(entity_id),
                ),
                timestamp=transaction_timestamp,
            )
        )

    for relationship in relationships:
        validate_relationship(relationship)

    return tuple(relationships)


# ---------------------------------------------------------------------------
# Historical relationship filtering
# ---------------------------------------------------------------------------

def filter_temporal_relationships(
    *,
    transaction_timestamp,
    relationships: Iterable[EntityRelationship],
) -> tuple[EntityRelationship, ...]:
    """
    Keep only relationships whose evidence timestamp is strictly
    earlier than the target transaction.

    Relationships without timestamps are retained because the
    timestamp may describe a static structural relationship rather
    than an event.

    This function never modifies the original relationship objects.
    """

    filtered: list[EntityRelationship] = []

    for relationship in relationships:
        validate_relationship(relationship)

        validate_temporal_relationship(
            transaction_timestamp=transaction_timestamp,
            relationship_timestamp=relationship.timestamp,
        )

        filtered.append(relationship)

    return tuple(filtered)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def extract_entities_from_relationships(
    relationships: Iterable[EntityRelationship],
) -> tuple[EntityReference, ...]:
    """
    Extract unique entities from relationship endpoints.

    Ordering is deterministic by entity type followed by entity ID.
    """

    entities: dict[tuple[str, str], EntityReference] = {}

    for relationship in relationships:
        validate_relationship(relationship)

        for entity in (
            relationship.source,
            relationship.target,
        ):
            key = (
                entity.entity_type,
                entity.entity_id,
            )
            entities[key] = entity

    return tuple(
        sorted(
            entities.values(),
            key=lambda entity: (
                entity.entity_type,
                entity.entity_id,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Intelligence assembly
# ---------------------------------------------------------------------------

def assemble_entity_relationship_intelligence(
    *,
    transaction_id: str,
    transaction_timestamp,
    relationships: Sequence[EntityRelationship],
) -> EntityRelationshipIntelligence:
    """
    Assemble deterministic entity relationship intelligence.

    All supplied relationships are validated and must satisfy the
    project's strict temporal rule.
    """

    if not transaction_id:
        raise FraudIntelligenceContractError(
            "transaction_id must be non-empty."
        )

    validated_relationships = filter_temporal_relationships(
        transaction_timestamp=transaction_timestamp,
        relationships=relationships,
    )

    # Prevent duplicate graph evidence.
    relationship_keys: set[
        tuple[
            str,
            str,
            str,
            str,
            str,
        ]
    ] = set()

    for relationship in validated_relationships:
        key = (
            relationship.relationship_type,
            relationship.source.entity_type,
            relationship.source.entity_id,
            relationship.target.entity_type,
            relationship.target.entity_id,
        )

        if key in relationship_keys:
            raise FraudIntelligenceContractError(
                "Duplicate relationship detected: "
                f"{key!r}"
            )

        relationship_keys.add(key)

    ordered_relationships = tuple(
        sorted(
            validated_relationships,
            key=lambda relationship: (
                relationship.relationship_type,
                relationship.source.entity_type,
                relationship.source.entity_id,
                relationship.target.entity_type,
                relationship.target.entity_id,
            ),
        )
    )

    entities = extract_entities_from_relationships(
        ordered_relationships
    )

    return EntityRelationshipIntelligence(
        transaction_id=str(transaction_id),
        transaction_timestamp=transaction_timestamp,
        entities=entities,
        relationships=ordered_relationships,
    )


# ---------------------------------------------------------------------------
# Completed-object validation
# ---------------------------------------------------------------------------

def validate_entity_relationship_intelligence(
    intelligence: EntityRelationshipIntelligence,
) -> None:
    """
    Validate a completed relationship intelligence object.
    """

    if not intelligence.transaction_id:
        raise FraudIntelligenceContractError(
            "transaction_id must be non-empty."
        )

    entity_keys: set[tuple[str, str]] = set()

    for entity in intelligence.entities:
        validate_entity_reference(entity)

        key = (
            entity.entity_type,
            entity.entity_id,
        )

        if key in entity_keys:
            raise FraudIntelligenceContractError(
                f"Duplicate entity detected: {key!r}"
            )

        entity_keys.add(key)

    relationship_keys: set[
        tuple[
            str,
            str,
            str,
            str,
            str,
        ]
    ] = set()

    for relationship in intelligence.relationships:
        validate_relationship(relationship)

        validate_temporal_relationship(
            transaction_timestamp=intelligence.transaction_timestamp,
            relationship_timestamp=relationship.timestamp,
        )

        key = (
            relationship.relationship_type,
            relationship.source.entity_type,
            relationship.source.entity_id,
            relationship.target.entity_type,
            relationship.target.entity_id,
        )

        if key in relationship_keys:
            raise FraudIntelligenceContractError(
                f"Duplicate relationship detected: {key!r}"
            )

        relationship_keys.add(key)

        if relationship.source not in intelligence.entities:
            raise FraudIntelligenceContractError(
                "Relationship source is missing from entity collection."
            )

        if relationship.target not in intelligence.entities:
            raise FraudIntelligenceContractError(
                "Relationship target is missing from entity collection."
            )