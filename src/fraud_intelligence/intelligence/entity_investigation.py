from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import FraudIntelligenceContractError
from .entity_relationships import (
    EntityReference,
    EntityRelationship,
    validate_entity_reference,
    validate_relationship,
)
from .fraud_ring_candidates import FraudRingCandidate
from .ring_network_scoring import RingNetworkScore


ENTITY_TYPES: tuple[str, ...] = (
    "customer",
    "account",
    "card",
    "transaction",
    "merchant",
    "device",
    "ip",
)


@dataclass(frozen=True)
class RelatedEntity:
    """
    One entity connected to the investigated entity.
    """

    entity: EntityReference
    relationship_type: str
    direction: str
    timestamp: object | None


@dataclass(frozen=True)
class EntityInvestigationView:
    """
    Backend investigation view for one entity.

    This is a descriptive representation of the entity's network
    context. It does not produce a fraud prediction.
    """

    entity: EntityReference
    related_entities: tuple[RelatedEntity, ...]
    related_transactions: tuple[str, ...]
    candidate_ring_ids: tuple[str, ...]
    network_scores: tuple[RingNetworkScore, ...]
    investigation_summary: str
    temporal_rule: str

    @property
    def related_entity_count(self) -> int:
        return len(self.related_entities)

    @property
    def related_transaction_count(self) -> int:
        return len(self.related_transactions)

    @property
    def candidate_ring_count(self) -> int:
        return len(self.candidate_ring_ids)


# ---------------------------------------------------------------------------
# Relationship extraction
# ---------------------------------------------------------------------------


def _related_entity_from_relationship(
    *,
    entity: EntityReference,
    relationship: EntityRelationship,
) -> RelatedEntity:
    """
    Convert a graph relationship into an investigator-facing relation.
    """

    if relationship.source == entity:
        return RelatedEntity(
            entity=relationship.target,
            relationship_type=relationship.relationship_type,
            direction="outgoing",
            timestamp=relationship.timestamp,
        )

    if relationship.target == entity:
        return RelatedEntity(
            entity=relationship.source,
            relationship_type=relationship.relationship_type,
            direction="incoming",
            timestamp=relationship.timestamp,
        )

    raise FraudIntelligenceContractError(
        "Relationship does not contain the investigated entity."
    )


def extract_related_entities(
    *,
    entity: EntityReference,
    relationships: Sequence[EntityRelationship],
) -> tuple[RelatedEntity, ...]:
    """
    Extract all entities directly related to the investigated entity.

    Duplicate relationships are rejected so the view remains a faithful
    representation of the validated investigation graph.
    """

    validate_entity_reference(entity)

    related: list[RelatedEntity] = []
    seen: set[
        tuple[
            str,
            str,
            str,
            str,
        ]
    ] = set()

    for relationship in relationships:
        validate_relationship(relationship)

        if (
            relationship.source != entity
            and relationship.target != entity
        ):
            continue

        item = _related_entity_from_relationship(
            entity=entity,
            relationship=relationship,
        )

        key = (
            item.relationship_type,
            item.entity.entity_type,
            item.entity.entity_id,
            item.direction,
        )

        if key in seen:
            raise FraudIntelligenceContractError(
                f"Duplicate relationship context: {key!r}"
            )

        seen.add(key)
        related.append(item)

    return tuple(
        sorted(
            related,
            key=lambda item: (
                item.relationship_type,
                item.direction,
                item.entity.entity_type,
                item.entity.entity_id,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Transaction extraction
# ---------------------------------------------------------------------------


def extract_related_transactions(
    related_entities: Sequence[RelatedEntity],
) -> tuple[str, ...]:
    """
    Extract directly related transaction IDs.

    Transaction identity is derived from graph relationships rather than
    fraud labels.
    """

    transaction_ids = {
        item.entity.entity_id
        for item in related_entities
        if item.entity.entity_type == "transaction"
    }

    return tuple(sorted(transaction_ids))


# ---------------------------------------------------------------------------
# Ring extraction
# ---------------------------------------------------------------------------


def extract_candidate_ring_ids(
    *,
    entity: EntityReference,
    candidates: Sequence[FraudRingCandidate],
) -> tuple[str, ...]:
    """
    Return candidate ring IDs containing the investigated entity.
    """

    entity_key = (
        entity.entity_type,
        entity.entity_id,
    )

    ring_ids: list[str] = []

    for candidate in candidates:
        candidate_keys = {
            (
                candidate_entity.entity_type,
                candidate_entity.entity_id,
            )
            for candidate_entity in candidate.entities
        }

        if entity_key in candidate_keys:
            ring_ids.append(candidate.candidate_id)

    return tuple(sorted(set(ring_ids)))


def extract_network_scores(
    *,
    candidate_ring_ids: Sequence[str],
    network_scores: Sequence[RingNetworkScore],
) -> tuple[RingNetworkScore, ...]:
    """
    Return scores corresponding to the candidate rings present in the view.
    """

    candidate_ids = set(candidate_ring_ids)

    return tuple(
        sorted(
            (
                score
                for score in network_scores
                if score.candidate_id in candidate_ids
            ),
            key=lambda score: score.candidate_id,
        )
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_investigation_summary(
    *,
    entity: EntityReference,
    related_entities: Sequence[RelatedEntity],
    related_transactions: Sequence[str],
    candidate_ring_ids: Sequence[str],
) -> str:
    """
    Build a concise descriptive summary.

    This text deliberately avoids claims that the entity is fraudulent.
    """

    entity_count = len(related_entities)
    transaction_count = len(related_transactions)
    ring_count = len(candidate_ring_ids)

    return (
        f"{entity.entity_type} {entity.entity_id} has "
        f"{entity_count} directly related entities, "
        f"{transaction_count} directly related transactions, "
        f"and appears in {ring_count} candidate network group(s). "
        "The view contains descriptive relationship context and "
        "does not establish fraud by itself."
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_entity_investigation_view(
    *,
    entity: EntityReference,
    relationships: Sequence[EntityRelationship],
    candidates: Sequence[FraudRingCandidate] = (),
    network_scores: Sequence[RingNetworkScore] = (),
) -> EntityInvestigationView:
    """
    Assemble the investigation view for an entity.

    Relationships supplied to this function are expected to have already
    passed the Phase 9 temporal controls.
    """

    validate_entity_reference(entity)

    related_entities = extract_related_entities(
        entity=entity,
        relationships=relationships,
    )

    related_transactions = extract_related_transactions(
        related_entities,
    )

    candidate_ring_ids = extract_candidate_ring_ids(
        entity=entity,
        candidates=candidates,
    )

    selected_network_scores = extract_network_scores(
        candidate_ring_ids=candidate_ring_ids,
        network_scores=network_scores,
    )

    summary = build_investigation_summary(
        entity=entity,
        related_entities=related_entities,
        related_transactions=related_transactions,
        candidate_ring_ids=candidate_ring_ids,
    )

    return EntityInvestigationView(
        entity=entity,
        related_entities=related_entities,
        related_transactions=related_transactions,
        candidate_ring_ids=candidate_ring_ids,
        network_scores=selected_network_scores,
        investigation_summary=summary,
        temporal_rule=(
            "Only validated investigation relationships may be used. "
            "Future context must not be introduced."
        ),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_entity_investigation_view(
    view: EntityInvestigationView,
) -> None:
    """
    Validate a completed entity investigation view.
    """

    validate_entity_reference(view.entity)

    if not view.temporal_rule:
        raise FraudIntelligenceContractError(
            "temporal_rule must be non-empty."
        )

    if not view.investigation_summary:
        raise FraudIntelligenceContractError(
            "investigation_summary must be non-empty."
        )

    related_keys: set[
        tuple[
            str,
            str,
            str,
            str,
        ]
    ] = set()

    for item in view.related_entities:
        validate_entity_reference(item.entity)

        if item.direction not in {
            "incoming",
            "outgoing",
        }:
            raise FraudIntelligenceContractError(
                "Relationship direction must be incoming or outgoing."
            )

        key = (
            item.relationship_type,
            item.entity.entity_type,
            item.entity.entity_id,
            item.direction,
        )

        if key in related_keys:
            raise FraudIntelligenceContractError(
                f"Duplicate related entity: {key!r}"
            )

        related_keys.add(key)

        if item.entity == view.entity:
            raise FraudIntelligenceContractError(
                "Entity investigation view cannot contain itself "
                "as a related entity."
            )

    if tuple(sorted(view.related_transactions)) != (
        view.related_transactions
    ):
        raise FraudIntelligenceContractError(
            "related_transactions must be sorted."
        )

    if len(set(view.related_transactions)) != (
        len(view.related_transactions)
    ):
        raise FraudIntelligenceContractError(
            "related_transactions must be unique."
        )

    if tuple(sorted(view.candidate_ring_ids)) != (
        view.candidate_ring_ids
    ):
        raise FraudIntelligenceContractError(
            "candidate_ring_ids must be sorted."
        )

    if len(set(view.candidate_ring_ids)) != (
        len(view.candidate_ring_ids)
    ):
        raise FraudIntelligenceContractError(
            "candidate_ring_ids must be unique."
        )

    score_ids = tuple(
        score.candidate_id
        for score in view.network_scores
    )

    if score_ids != tuple(sorted(score_ids)):
        raise FraudIntelligenceContractError(
            "network_scores must be sorted by candidate ID."
        )

    if len(set(score_ids)) != len(score_ids):
        raise FraudIntelligenceContractError(
            "network_scores must contain unique candidate IDs."
        )

    if not set(score_ids).issubset(
        set(view.candidate_ring_ids)
    ):
        raise FraudIntelligenceContractError(
            "Every network score must correspond to a candidate ring "
            "present in the view."
        )

    forbidden_terms = (
        "is_fraud",
        "fraud_scenario",
    )

    combined_text = (
        view.investigation_summary
        + " "
        + view.temporal_rule
    )

    for forbidden_term in forbidden_terms:
        if forbidden_term in combined_text:
            raise FraudIntelligenceContractError(
                "Investigation view contains forbidden predictive "
                f"field: {forbidden_term!r}"
            )