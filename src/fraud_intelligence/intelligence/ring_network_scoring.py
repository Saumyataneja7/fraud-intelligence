from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Sequence

from .contracts import FraudIntelligenceContractError
from .entity_relationships import EntityRelationship
from .fraud_ring_candidates import FraudRingCandidate


# ---------------------------------------------------------------------------
# Score weights
# ---------------------------------------------------------------------------

ENTITY_COUNT_WEIGHT = 0.20
RELATIONSHIP_COUNT_WEIGHT = 0.20
DENSITY_WEIGHT = 0.25
ENTITY_DIVERSITY_WEIGHT = 0.15
RELATIONSHIP_DIVERSITY_WEIGHT = 0.10
NON_TRANSACTION_CONNECTIVITY_WEIGHT = 0.10


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RingNetworkScore:
    """
    Structural network score for one fraud-ring candidate.

    This score represents network connectivity and structural richness.
    It is NOT a fraud probability and must not be interpreted as a
    classification output.
    """

    candidate_id: str

    entity_count: int
    relationship_count: int

    relationship_density: float

    entity_type_count: int
    relationship_type_count: int

    non_transaction_entity_count: int
    non_transaction_connectivity: float

    structural_score: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bounded_score(value: float) -> float:
    """
    Clamp a score to [0, 1].
    """

    return max(0.0, min(1.0, float(value)))


def _normalize_count(
    count: int,
    *,
    reference: int,
) -> float:
    """
    Normalize a count to [0, 1].

    The reference value is intentionally fixed and deterministic.
    It is a structural normalization, not a learned threshold.
    """

    if count <= 0:
        return 0.0

    if reference <= 0:
        raise FraudIntelligenceContractError(
            "reference must be positive."
        )

    return _bounded_score(count / reference)


def _entity_type_diversity(
    candidate: FraudRingCandidate,
) -> float:
    """
    Fraction of allowed entity types represented in the candidate.
    """

    allowed_entity_types = 7

    entity_types = {
        entity.entity_type
        for entity in candidate.entities
    }

    return _bounded_score(
        len(entity_types) / allowed_entity_types
    )


def _relationship_type_diversity(
    candidate: FraudRingCandidate,
) -> float:
    """
    Fraction of canonical relationship types represented.
    """

    allowed_relationship_types = 11

    relationship_types = {
        relationship.relationship_type
        for relationship in candidate.relationships
    }

    return _bounded_score(
        len(relationship_types) / allowed_relationship_types
    )


def _relationship_density(
    *,
    entity_count: int,
    relationship_count: int,
) -> float:
    """
    Calculate undirected graph density.

    density = E / C(N, 2)

    Multiple canonical relationships between the same pair are counted
    as relationships, so this metric captures relationship richness rather
    than strictly simple-graph density.
    """

    if entity_count < 2:
        return 0.0

    possible_pairs = comb(entity_count, 2)

    if possible_pairs == 0:
        return 0.0

    return _bounded_score(
        relationship_count / possible_pairs
    )


def _non_transaction_connectivity(
    candidate: FraudRingCandidate,
) -> float:
    """
    Measure how strongly the candidate's non-transaction entities are
    connected through relationships.

    This is based on the ratio of relationships touching at least one
    non-transaction entity.
    """

    if not candidate.relationships:
        return 0.0

    non_transaction_entities = {
        (
            entity.entity_type,
            entity.entity_id,
        )
        for entity in candidate.entities
        if entity.entity_type != "transaction"
    }

    if not non_transaction_entities:
        return 0.0

    relevant_relationships = 0

    for relationship in candidate.relationships:
        source = (
            relationship.source.entity_type,
            relationship.source.entity_id,
        )

        target = (
            relationship.target.entity_type,
            relationship.target.entity_id,
        )

        if (
            source in non_transaction_entities
            or target in non_transaction_entities
        ):
            relevant_relationships += 1

    return _bounded_score(
        relevant_relationships / len(candidate.relationships)
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_fraud_ring_candidate(
    candidate: FraudRingCandidate,
) -> RingNetworkScore:
    """
    Calculate a deterministic structural network score.

    The score is deliberately independent of fraud labels and model
    predictions.

    All components are normalized to [0, 1], then combined into a
    weighted score and scaled to [0, 100].
    """

    entity_count = candidate.entity_count
    relationship_count = candidate.relationship_count

    relationship_density = _relationship_density(
        entity_count=entity_count,
        relationship_count=relationship_count,
    )

    entity_type_count = len(
        {
            entity.entity_type
            for entity in candidate.entities
        }
    )

    relationship_type_count = len(
        {
            relationship.relationship_type
            for relationship in candidate.relationships
        }
    )

    non_transaction_entity_count = sum(
        entity.entity_type != "transaction"
        for entity in candidate.entities
    )

    non_transaction_connectivity = (
        _non_transaction_connectivity(candidate)
    )

    # Fixed structural normalizations.
    entity_count_component = _normalize_count(
        entity_count,
        reference=10,
    )

    relationship_count_component = _normalize_count(
        relationship_count,
        reference=15,
    )

    entity_diversity_component = _entity_type_diversity(
        candidate
    )

    relationship_diversity_component = (
        _relationship_type_diversity(candidate)
    )

    structural_score = 100.0 * (
        ENTITY_COUNT_WEIGHT * entity_count_component
        + RELATIONSHIP_COUNT_WEIGHT * relationship_count_component
        + DENSITY_WEIGHT * relationship_density
        + ENTITY_DIVERSITY_WEIGHT * entity_diversity_component
        + RELATIONSHIP_DIVERSITY_WEIGHT
        * relationship_diversity_component
        + NON_TRANSACTION_CONNECTIVITY_WEIGHT
        * non_transaction_connectivity
    )

    structural_score = round(
        _bounded_score(structural_score / 100.0) * 100.0,
        6,
    )

    return RingNetworkScore(
        candidate_id=candidate.candidate_id,
        entity_count=entity_count,
        relationship_count=relationship_count,
        relationship_density=round(
            relationship_density,
            6,
        ),
        entity_type_count=entity_type_count,
        relationship_type_count=relationship_type_count,
        non_transaction_entity_count=non_transaction_entity_count,
        non_transaction_connectivity=round(
            non_transaction_connectivity,
            6,
        ),
        structural_score=structural_score,
    )


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------


def score_fraud_ring_candidates(
    candidates: Sequence[FraudRingCandidate],
) -> tuple[RingNetworkScore, ...]:
    """
    Score multiple candidates deterministically.

    Results are ordered by candidate ID, not by score. This avoids turning
    the module into a ranking or selection mechanism.
    """

    scores = tuple(
        score_fraud_ring_candidate(candidate)
        for candidate in candidates
    )

    return tuple(
        sorted(
            scores,
            key=lambda score: score.candidate_id,
        )
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ring_network_score(
    score: RingNetworkScore,
) -> None:
    """
    Validate one network score.
    """

    if not score.candidate_id:
        raise FraudIntelligenceContractError(
            "candidate_id must be non-empty."
        )

    if score.entity_count < 0:
        raise FraudIntelligenceContractError(
            "entity_count cannot be negative."
        )

    if score.relationship_count < 0:
        raise FraudIntelligenceContractError(
            "relationship_count cannot be negative."
        )

    if score.entity_type_count < 0:
        raise FraudIntelligenceContractError(
            "entity_type_count cannot be negative."
        )

    if score.relationship_type_count < 0:
        raise FraudIntelligenceContractError(
            "relationship_type_count cannot be negative."
        )

    if score.non_transaction_entity_count < 0:
        raise FraudIntelligenceContractError(
            "non_transaction_entity_count cannot be negative."
        )

    bounded_fields = {
        "relationship_density": score.relationship_density,
        "non_transaction_connectivity": (
            score.non_transaction_connectivity
        ),
    }

    for field_name, value in bounded_fields.items():
        if not 0.0 <= float(value) <= 1.0:
            raise FraudIntelligenceContractError(
                f"{field_name} must be between 0 and 1."
            )

    if not 0.0 <= float(score.structural_score) <= 100.0:
        raise FraudIntelligenceContractError(
            "structural_score must be between 0 and 100."
        )


def validate_ring_network_scores(
    scores: Sequence[RingNetworkScore],
) -> None:
    """
    Validate a collection of network scores.
    """

    candidate_ids: set[str] = set()

    for score in scores:
        validate_ring_network_score(score)

        if score.candidate_id in candidate_ids:
            raise FraudIntelligenceContractError(
                f"Duplicate candidate ID: {score.candidate_id!r}"
            )

        candidate_ids.add(score.candidate_id)