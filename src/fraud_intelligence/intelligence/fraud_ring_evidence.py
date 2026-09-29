from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import FraudIntelligenceContractError
from .entity_relationships import (
    EntityReference,
    EntityRelationship,
    validate_relationship,
)
from .fraud_ring_candidates import (
    FraudRingCandidate,
    validate_fraud_ring_candidate,
)
from .ring_network_scoring import (
    RingNetworkScore,
    validate_ring_network_score,
)


FORBIDDEN_EVIDENCE_FIELDS: frozenset[str] = frozenset(
    {
        "is_fraud",
        "fraud_scenario",
    }
)


@dataclass(frozen=True)
class RingEvidenceItem:
    """
    One investigator-readable piece of structural evidence.

    Evidence is descriptive and does not represent a fraud label.
    """

    evidence_type: str
    description: str
    entity_types: tuple[str, ...]
    relationship_types: tuple[str, ...]
    strength: str


@dataclass(frozen=True)
class FraudRingEvidence:
    """
    Complete evidence package for one candidate ring.
    """

    candidate_id: str
    target_transaction_id: str

    entities: tuple[EntityReference, ...]
    relationships: tuple[EntityRelationship, ...]

    network_score: RingNetworkScore
    evidence_items: tuple[RingEvidenceItem, ...]

    temporal_context: str

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_items)

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------


def _entity_type_set(
    candidate: FraudRingCandidate,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                entity.entity_type
                for entity in candidate.entities
            }
        )
    )


def _relationship_type_set(
    candidate: FraudRingCandidate,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                relationship.relationship_type
                for relationship in candidate.relationships
            }
        )
    )


def _strength_from_score(
    structural_score: float,
) -> str:
    """
    Convert the structural score into a descriptive evidence strength.

    This is NOT a fraud classification.

    The categories describe the amount of structural evidence only.
    """

    if structural_score >= 75.0:
        return "strong"

    if structural_score >= 50.0:
        return "moderate"

    return "limited"


# ---------------------------------------------------------------------------
# Evidence generation
# ---------------------------------------------------------------------------


def build_ring_evidence_items(
    *,
    candidate: FraudRingCandidate,
    network_score: RingNetworkScore,
) -> tuple[RingEvidenceItem, ...]:
    """
    Build deterministic investigator-readable evidence items.

    No fraud labels or model outputs are used.
    """

    validate_fraud_ring_candidate(candidate)
    validate_ring_network_score(network_score)

    if candidate.candidate_id != network_score.candidate_id:
        raise FraudIntelligenceContractError(
            "Candidate ID and network score ID must match."
        )

    evidence: list[RingEvidenceItem] = []

    entity_types = _entity_type_set(candidate)
    relationship_types = _relationship_type_set(candidate)

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    if candidate.entity_count >= 3:
        evidence.append(
            RingEvidenceItem(
                evidence_type="entity_connectivity",
                description=(
                    f"The candidate contains {candidate.entity_count} "
                    "connected entities."
                ),
                entity_types=entity_types,
                relationship_types=relationship_types,
                strength=_strength_from_score(
                    network_score.structural_score
                ),
            )
        )

    # ------------------------------------------------------------------
    # Relationship richness
    # ------------------------------------------------------------------

    if candidate.relationship_count >= 2:
        evidence.append(
            RingEvidenceItem(
                evidence_type="relationship_richness",
                description=(
                    f"The candidate contains "
                    f"{candidate.relationship_count} relationships "
                    "connecting the participating entities."
                ),
                entity_types=entity_types,
                relationship_types=relationship_types,
                strength=_strength_from_score(
                    network_score.structural_score
                ),
            )
        )

    # ------------------------------------------------------------------
    # Entity diversity
    # ------------------------------------------------------------------

    if network_score.entity_type_count >= 3:
        evidence.append(
            RingEvidenceItem(
                evidence_type="entity_type_diversity",
                description=(
                    f"The candidate spans "
                    f"{network_score.entity_type_count} entity types."
                ),
                entity_types=entity_types,
                relationship_types=relationship_types,
                strength=(
                    "strong"
                    if network_score.entity_type_count >= 5
                    else "moderate"
                ),
            )
        )

    # ------------------------------------------------------------------
    # Relationship diversity
    # ------------------------------------------------------------------

    if network_score.relationship_type_count >= 2:
        evidence.append(
            RingEvidenceItem(
                evidence_type="relationship_type_diversity",
                description=(
                    f"The candidate uses "
                    f"{network_score.relationship_type_count} "
                    "relationship types."
                ),
                entity_types=entity_types,
                relationship_types=relationship_types,
                strength=(
                    "strong"
                    if network_score.relationship_type_count >= 4
                    else "moderate"
                ),
            )
        )

    # ------------------------------------------------------------------
    # Density
    # ------------------------------------------------------------------

    if network_score.relationship_density > 0.5:
        evidence.append(
            RingEvidenceItem(
                evidence_type="network_density",
                description=(
                    "The candidate has relatively dense connectivity "
                    "among its participating entities."
                ),
                entity_types=entity_types,
                relationship_types=relationship_types,
                strength="strong",
            )
        )
    elif network_score.relationship_density > 0.25:
        evidence.append(
            RingEvidenceItem(
                evidence_type="network_density",
                description=(
                    "The candidate has multiple connections among "
                    "its participating entities."
                ),
                entity_types=entity_types,
                relationship_types=relationship_types,
                strength="moderate",
            )
        )

    # ------------------------------------------------------------------
    # Non-transaction connectivity
    # ------------------------------------------------------------------

    if network_score.non_transaction_connectivity > 0.5:
        evidence.append(
            RingEvidenceItem(
                evidence_type="non_transaction_connectivity",
                description=(
                    "Most candidate relationships connect "
                    "non-transaction entities, providing reusable "
                    "network context beyond a single transaction."
                ),
                entity_types=entity_types,
                relationship_types=relationship_types,
                strength="strong",
            )
        )

    # ------------------------------------------------------------------
    # Stable ordering
    # ------------------------------------------------------------------

    return tuple(
        sorted(
            evidence,
            key=lambda item: (
                item.evidence_type,
                item.description,
            ),
        )
    )


# ---------------------------------------------------------------------------
# Evidence assembly
# ---------------------------------------------------------------------------


def assemble_fraud_ring_evidence(
    *,
    candidate: FraudRingCandidate,
    network_score: RingNetworkScore,
    target_transaction_id: str,
) -> FraudRingEvidence:
    """
    Assemble the complete evidence package for a candidate ring.

    This function combines Phase 9.4 candidate detection with the
    Phase 9.5 structural network score.

    No fraud labels are accepted or generated.
    """

    if not target_transaction_id:
        raise FraudIntelligenceContractError(
            "target_transaction_id must be non-empty."
        )

    validate_fraud_ring_candidate(candidate)
    validate_ring_network_score(network_score)

    if candidate.candidate_id != network_score.candidate_id:
        raise FraudIntelligenceContractError(
            "Candidate and network score IDs do not match."
        )

    target_key = (
        "transaction",
        str(target_transaction_id),
    )

    candidate_entity_keys = {
        (
            entity.entity_type,
            entity.entity_id,
        )
        for entity in candidate.entities
    }

    if target_key not in candidate_entity_keys:
        raise FraudIntelligenceContractError(
            "Target transaction must belong to the candidate ring."
        )

    for relationship in candidate.relationships:
        validate_relationship(relationship)

    evidence_items = build_ring_evidence_items(
        candidate=candidate,
        network_score=network_score,
    )

    temporal_context = (
        "Evidence is based on the validated relationship context "
        "available to the investigation. Future transaction context "
        "must not be introduced."
    )

    return FraudRingEvidence(
        candidate_id=candidate.candidate_id,
        target_transaction_id=str(target_transaction_id),
        entities=candidate.entities,
        relationships=candidate.relationships,
        network_score=network_score,
        evidence_items=evidence_items,
        temporal_context=temporal_context,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ring_evidence_item(
    item: RingEvidenceItem,
) -> None:
    """
    Validate one evidence item.
    """

    if not item.evidence_type:
        raise FraudIntelligenceContractError(
            "evidence_type must be non-empty."
        )

    if not item.description:
        raise FraudIntelligenceContractError(
            "Evidence description must be non-empty."
        )

    if item.strength not in {
        "limited",
        "moderate",
        "strong",
    }:
        raise FraudIntelligenceContractError(
            "Evidence strength must be limited, moderate, or strong."
        )

    forbidden = (
        set(item.entity_types)
        | set(item.relationship_types)
    ).intersection(FORBIDDEN_EVIDENCE_FIELDS)

    if forbidden:
        raise FraudIntelligenceContractError(
            "Evidence contains forbidden fields: "
            f"{sorted(forbidden)}"
        )


def validate_fraud_ring_evidence(
    evidence: FraudRingEvidence,
) -> None:
    """
    Validate a complete fraud-ring evidence package.
    """

    if not evidence.candidate_id:
        raise FraudIntelligenceContractError(
            "candidate_id must be non-empty."
        )

    if not evidence.target_transaction_id:
        raise FraudIntelligenceContractError(
            "target_transaction_id must be non-empty."
        )

    if not evidence.temporal_context:
        raise FraudIntelligenceContractError(
            "temporal_context must be non-empty."
        )

    if evidence.network_score.candidate_id != (
        evidence.candidate_id
    ):
        raise FraudIntelligenceContractError(
            "Network score candidate ID does not match evidence."
        )

    entity_keys = {
        (
            entity.entity_type,
            entity.entity_id,
        )
        for entity in evidence.entities
    }

    if len(entity_keys) != len(evidence.entities):
        raise FraudIntelligenceContractError(
            "Evidence contains duplicate entities."
        )

    relationship_keys: set[
        tuple[
            str,
            str,
            str,
            str,
            str,
        ]
    ] = set()

    for relationship in evidence.relationships:
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
                "Relationship source is outside the evidence entity set."
            )

        if target_key not in entity_keys:
            raise FraudIntelligenceContractError(
                "Relationship target is outside the evidence entity set."
            )

        relationship_key = (
            relationship.relationship_type,
            relationship.source.entity_type,
            relationship.source.entity_id,
            relationship.target.entity_type,
            relationship.target.entity_id,
        )

        if relationship_key in relationship_keys:
            raise FraudIntelligenceContractError(
                "Evidence contains duplicate relationships."
            )

        relationship_keys.add(relationship_key)

    evidence_types: set[str] = set()

    for item in evidence.evidence_items:
        validate_ring_evidence_item(item)

        if item.evidence_type in evidence_types:
            raise FraudIntelligenceContractError(
                "Evidence item types must be unique."
            )

        evidence_types.add(item.evidence_type)

    target_key = (
        "transaction",
        evidence.target_transaction_id,
    )

    if target_key not in entity_keys:
        raise FraudIntelligenceContractError(
            "Target transaction is missing from evidence entities."
        )

    forbidden_text = (
        "is_fraud",
        "fraud_scenario",
    )

    combined_text = " ".join(
        [
            evidence.temporal_context,
            *(
                item.description
                for item in evidence.evidence_items
            ),
        ]
    )

    for forbidden_field in forbidden_text:
        if forbidden_field in combined_text:
            raise FraudIntelligenceContractError(
                "Evidence text must not contain forbidden "
                f"predictive field: {forbidden_field!r}"
            )