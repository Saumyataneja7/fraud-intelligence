from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import FraudIntelligenceContractError
from .entity_relationships import (
    EntityReference,
    EntityRelationship,
    validate_relationship,
)
from .fraud_ring_candidates import FraudRingCandidate
from .fraud_ring_evidence import (
    FraudRingEvidence,
)
from .ring_network_scoring import RingNetworkScore


@dataclass(frozen=True)
class RingTransactionReference:
    """Transaction contained in an investigation ring."""

    transaction_id: str
    timestamp: object | None


@dataclass(frozen=True)
class FraudRingInvestigationView:
    """
    Investigator-facing view of one fraud-ring candidate.

    This is a descriptive aggregation of existing structural,
    network, and evidence outputs. It does not create a new
    fraud prediction.
    """

    candidate_id: str

    entities: tuple[EntityReference, ...]
    relationships: tuple[EntityRelationship, ...]
    transactions: tuple[RingTransactionReference, ...]

    network_score: RingNetworkScore
    evidence: FraudRingEvidence | None

    entity_type_counts: tuple[tuple[str, int], ...]
    relationship_type_counts: tuple[tuple[str, int], ...]

    investigation_summary: str
    temporal_rule: str

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)

    @property
    def transaction_count(self) -> int:
        return len(self.transactions)

    @property
    def evidence_count(self) -> int:
        if self.evidence is None:
            return 0

        return self.evidence.evidence_count


# ---------------------------------------------------------------------------
# Entity / relationship aggregation
# ---------------------------------------------------------------------------


def count_entity_types(
    entities: Sequence[EntityReference],
) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}

    for entity in entities:
        counts[entity.entity_type] = (
            counts.get(entity.entity_type, 0) + 1
        )

    return tuple(sorted(counts.items()))


def count_relationship_types(
    relationships: Sequence[EntityRelationship],
) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}

    for relationship in relationships:
        validate_relationship(relationship)

        counts[relationship.relationship_type] = (
            counts.get(relationship.relationship_type, 0) + 1
        )

    return tuple(sorted(counts.items()))


def extract_ring_transactions(
    entities: Sequence[EntityReference],
    relationships: Sequence[EntityRelationship],
) -> tuple[RingTransactionReference, ...]:
    """
    Extract transaction references from the candidate.

    Timestamp is taken from transaction-involving relationships where
    available. No timestamp is invented when the source relationship
    has no timestamp.
    """

    transaction_ids = {
        entity.entity_id
        for entity in entities
        if entity.entity_type == "transaction"
    }

    timestamps: dict[str, object | None] = {
        transaction_id: None
        for transaction_id in transaction_ids
    }

    for relationship in relationships:
        validate_relationship(relationship)

        for entity in (
            relationship.source,
            relationship.target,
        ):
            if (
                entity.entity_type == "transaction"
                and entity.entity_id in timestamps
                and relationship.timestamp is not None
            ):
                existing = timestamps[entity.entity_id]

                if existing is None:
                    timestamps[entity.entity_id] = (
                        relationship.timestamp
                    )

                elif relationship.timestamp < existing:
                    timestamps[entity.entity_id] = (
                        relationship.timestamp
                    )

    return tuple(
        RingTransactionReference(
            transaction_id=transaction_id,
            timestamp=timestamps[transaction_id],
        )
        for transaction_id in sorted(transaction_ids)
    )


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def select_ring_evidence(
    *,
    candidate_id: str,
    evidence: Sequence[FraudRingEvidence],
) -> FraudRingEvidence | None:
    """
    Select the evidence package belonging to the candidate.

    At most one evidence package is accepted for a candidate.
    """

    matches = [
        item
        for item in evidence
        if item.candidate_id == candidate_id
    ]

    if len(matches) > 1:
        raise FraudIntelligenceContractError(
            "Multiple evidence packages found for candidate."
        )

    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_ring_investigation_summary(
    *,
    candidate_id: str,
    entity_count: int,
    relationship_count: int,
    transaction_count: int,
    entity_type_count: int,
    relationship_type_count: int,
    structural_score: float,
    evidence_count: int,
) -> str:
    """
    Build a descriptive ring-level investigation summary.

    The structural score is reported as an existing network score.
    It is not converted into a fraud probability.
    """

    entity_word = "entity" if entity_count == 1 else "entities"

    evidence_description = (
        f"{evidence_count} evidence item(s)"
        if evidence_count
        else "no assembled evidence items"
    )

    return (
        f"Candidate ring {candidate_id} contains "
        f"{entity_count} {entity_word}, "
        f"{relationship_count} relationship(s), and "
        f"{transaction_count} transaction(s) across "
        f"{entity_type_count} entity type(s) and "
        f"{relationship_type_count} relationship type(s). "
        f"Its existing structural network score is "
        f"{structural_score:.6f}. The investigation package contains "
        f"{evidence_description}. This is structural investigation "
        "context and is not a fraud probability."
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_fraud_ring_investigation_view(
    *,
    candidate: FraudRingCandidate,
    network_score: RingNetworkScore,
    evidence: FraudRingEvidence | None = None,
) -> FraudRingInvestigationView:
    """
    Assemble a complete investigation view for one candidate ring.
    """

    if candidate.candidate_id != network_score.candidate_id:
        raise FraudIntelligenceContractError(
            "Candidate ID and network score ID do not match."
        )

    if evidence is not None:
        if evidence.candidate_id != candidate.candidate_id:
            raise FraudIntelligenceContractError(
                "Candidate ID and evidence ID do not match."
            )

    for relationship in candidate.relationships:
        validate_relationship(relationship)

    entities = tuple(
        sorted(
            candidate.entities,
            key=lambda entity: (
                entity.entity_type,
                entity.entity_id,
            ),
        )
    )

    relationships = tuple(
        sorted(
            candidate.relationships,
            key=lambda relationship: (
                relationship.relationship_type,
                relationship.source.entity_type,
                relationship.source.entity_id,
                relationship.target.entity_type,
                relationship.target.entity_id,
            ),
        )
    )

    transactions = extract_ring_transactions(
        entities,
        relationships,
    )

    entity_type_counts = count_entity_types(entities)
    relationship_type_counts = count_relationship_types(
        relationships
    )

    evidence_count = (
        evidence.evidence_count
        if evidence is not None
        else 0
    )

    summary = build_ring_investigation_summary(
        candidate_id=candidate.candidate_id,
        entity_count=len(entities),
        relationship_count=len(relationships),
        transaction_count=len(transactions),
        entity_type_count=len(entity_type_counts),
        relationship_type_count=len(
            relationship_type_counts
        ),
        structural_score=network_score.structural_score,
        evidence_count=evidence_count,
    )

    return FraudRingInvestigationView(
        candidate_id=candidate.candidate_id,
        entities=entities,
        relationships=relationships,
        transactions=transactions,
        network_score=network_score,
        evidence=evidence,
        entity_type_counts=entity_type_counts,
        relationship_type_counts=relationship_type_counts,
        investigation_summary=summary,
        temporal_rule=(
            "Only validated investigation context associated with "
            "the candidate ring may be used. Future predictive "
            "context must not be introduced."
        ),
    )


# ---------------------------------------------------------------------------
# Batch assembly
# ---------------------------------------------------------------------------


def assemble_fraud_ring_investigation_views(
    *,
    candidates: Sequence[FraudRingCandidate],
    network_scores: Sequence[RingNetworkScore],
    evidence: Sequence[FraudRingEvidence] = (),
) -> tuple[FraudRingInvestigationView, ...]:
    """
    Assemble investigation views for multiple candidate rings.

    Output ordering is deterministic by candidate ID.
    """

    score_by_id = {
        score.candidate_id: score
        for score in network_scores
    }

    if len(score_by_id) != len(network_scores):
        raise FraudIntelligenceContractError(
            "Duplicate network score candidate IDs."
        )

    views = []

    for candidate in candidates:
        score = score_by_id.get(candidate.candidate_id)

        if score is None:
            raise FraudIntelligenceContractError(
                "Missing network score for candidate "
                f"{candidate.candidate_id!r}."
            )

        candidate_evidence = select_ring_evidence(
            candidate_id=candidate.candidate_id,
            evidence=evidence,
        )

        views.append(
            assemble_fraud_ring_investigation_view(
                candidate=candidate,
                network_score=score,
                evidence=candidate_evidence,
            )
        )

    return tuple(
        sorted(
            views,
            key=lambda view: view.candidate_id,
        )
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_fraud_ring_investigation_view(
    view: FraudRingInvestigationView,
) -> None:
    """
    Validate the complete ring investigation view.
    """

    if not view.candidate_id:
        raise FraudIntelligenceContractError(
            "candidate_id must be non-empty."
        )

    if view.network_score.candidate_id != view.candidate_id:
        raise FraudIntelligenceContractError(
            "Network score candidate ID does not match view."
        )

    if view.evidence is not None:
        if view.evidence.candidate_id != view.candidate_id:
            raise FraudIntelligenceContractError(
                "Evidence candidate ID does not match view."
            )

    if not 0.0 <= view.network_score.structural_score <= 100.0:
        raise FraudIntelligenceContractError(
            "Structural score must be between 0 and 100."
        )

    entity_keys = {
        (
            entity.entity_type,
            entity.entity_id,
        )
        for entity in view.entities
    }

    relationship_keys: set[
        tuple[str, str, str, str, str]
    ] = set()

    for relationship in view.relationships:
        validate_relationship(relationship)

        source_key = (
            relationship.source.entity_type,
            relationship.source.entity_id,
        )

        target_key = (
            relationship.target.entity_type,
            relationship.target.entity_id,
        )

        if source_key not in entity_keys:
            raise FraudIntelligenceContractError(
                "Relationship source is outside the ring entity set."
            )

        if target_key not in entity_keys:
            raise FraudIntelligenceContractError(
                "Relationship target is outside the ring entity set."
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
                "Duplicate relationship in ring investigation view."
            )

        relationship_keys.add(key)

    transaction_ids = {
        entity.entity_id
        for entity in view.entities
        if entity.entity_type == "transaction"
    }

    view_transaction_ids = {
        transaction.transaction_id
        for transaction in view.transactions
    }

    if transaction_ids != view_transaction_ids:
        raise FraudIntelligenceContractError(
            "Transaction references do not match transaction entities."
        )

    if tuple(
        transaction.transaction_id
        for transaction in view.transactions
    ) != tuple(
        sorted(transaction.transaction_id for transaction in view.transactions)
    ):
        raise FraudIntelligenceContractError(
            "Transaction references must be sorted."
        )

    expected_entity_counts = count_entity_types(
        view.entities
    )

    if view.entity_type_counts != expected_entity_counts:
        raise FraudIntelligenceContractError(
            "entity_type_counts do not match entities."
        )

    expected_relationship_counts = count_relationship_types(
        view.relationships
    )

    if (
        view.relationship_type_counts
        != expected_relationship_counts
    ):
        raise FraudIntelligenceContractError(
            "relationship_type_counts do not match relationships."
        )

    if not view.investigation_summary:
        raise FraudIntelligenceContractError(
            "investigation_summary must be non-empty."
        )

    if not view.temporal_rule:
        raise FraudIntelligenceContractError(
            "temporal_rule must be non-empty."
        )

    forbidden_terms = (
        "is_fraud",
        "fraud_scenario",
    )

    text = (
        view.investigation_summary
        + " "
        + view.temporal_rule
    ).lower()

    for term in forbidden_terms:
        if term in text:
            raise FraudIntelligenceContractError(
                f"Forbidden predictive field {term!r} "
                "appears in investigation text."
            )