from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import networkx as nx

from .contracts import FraudIntelligenceContractError
from .entity_relationships import (
    EntityReference,
    EntityRelationship,
    validate_relationship,
)


# ---------------------------------------------------------------------------
# Candidate-ring configuration
# ---------------------------------------------------------------------------

DEFAULT_MIN_ENTITIES = 3
DEFAULT_MIN_SHARED_RELATIONSHIPS = 2


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FraudRingCandidate:
    """
    A candidate coordinated entity group.

    This is an investigation candidate, not a fraud label.
    """

    candidate_id: str
    entities: tuple[EntityReference, ...]
    relationships: tuple[EntityRelationship, ...]
    connected_component_size: int
    shared_relationship_count: int

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)


@dataclass(frozen=True)
class FraudRingCandidateDetectionResult:
    """
    Candidate-ring detection output for one investigation context.
    """

    target_transaction_id: str
    candidates: tuple[FraudRingCandidate, ...]

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_investigation_graph(
    relationships: Iterable[EntityRelationship],
) -> nx.Graph:
    """
    Build an undirected investigation graph from validated relationships.

    Direction is intentionally removed for candidate discovery because
    coordinated activity can be expressed through multiple relationship
    directions.

    Relationship metadata is retained on graph edges.
    """

    graph = nx.Graph()

    for relationship in relationships:
        validate_relationship(relationship)

        source_key = (
            relationship.source.entity_type,
            relationship.source.entity_id,
        )

        target_key = (
            relationship.target.entity_type,
            relationship.target.entity_id,
        )

        graph.add_node(
            source_key,
            entity=relationship.source,
        )

        graph.add_node(
            target_key,
            entity=relationship.target,
        )

        if graph.has_edge(source_key, target_key):
            graph[source_key][target_key]["relationships"].append(
                relationship
            )
        else:
            graph.add_edge(
                source_key,
                target_key,
                relationships=[relationship],
            )

    return graph


# ---------------------------------------------------------------------------
# Candidate component discovery
# ---------------------------------------------------------------------------


def _component_relationships(
    graph: nx.Graph,
    component: set[tuple[str, str]],
) -> tuple[EntityRelationship, ...]:
    relationships: list[EntityRelationship] = []

    for source, target, data in graph.subgraph(component).edges(
        data=True
    ):
        del source, target

        relationships.extend(data["relationships"])

    # Stable ordering.
    return tuple(
        sorted(
            relationships,
            key=lambda relationship: (
                relationship.relationship_type,
                relationship.source.entity_type,
                relationship.source.entity_id,
                relationship.target.entity_type,
                relationship.target.entity_id,
            ),
        )
    )


def _component_entities(
    graph: nx.Graph,
    component: set[tuple[str, str]],
) -> tuple[EntityReference, ...]:
    entities = [
        graph.nodes[node]["entity"]
        for node in component
    ]

    return tuple(
        sorted(
            entities,
            key=lambda entity: (
                entity.entity_type,
                entity.entity_id,
            ),
        )
    )


def _candidate_id(
    entities: Sequence[EntityReference],
) -> str:
    """
    Generate a deterministic candidate identifier.

    The identifier is based only on entity identity and is therefore
    independent of NetworkX traversal order.
    """

    parts = [
        f"{entity.entity_type}:{entity.entity_id}"
        for entity in entities
    ]

    return "ring-" + "|".join(parts)


# ---------------------------------------------------------------------------
# Candidate detection
# ---------------------------------------------------------------------------


def detect_fraud_ring_candidates(
    *,
    target_transaction_id: str,
    relationships: Sequence[EntityRelationship],
    min_entities: int = DEFAULT_MIN_ENTITIES,
    min_shared_relationships: int = DEFAULT_MIN_SHARED_RELATIONSHIPS,
) -> FraudRingCandidateDetectionResult:
    """
    Detect connected entity groups that are large enough to investigate.

    Parameters
    ----------
    target_transaction_id:
        Transaction being investigated.

    relationships:
        Leakage-safe relationships supplied by Phase 9.3.

    min_entities:
        Minimum number of entities required for a candidate.

    min_shared_relationships:
        Minimum number of graph relationships required.

    Notes
    -----
    This function does NOT:
      - use is_fraud
      - use fraud_scenario
      - train a model
      - calculate a fraud probability
      - declare a group fraudulent

    It only identifies structurally connected candidate groups.
    """

    if not target_transaction_id:
        raise FraudIntelligenceContractError(
            "target_transaction_id must be non-empty."
        )

    if min_entities < 2:
        raise FraudIntelligenceContractError(
            "min_entities must be at least 2."
        )

    if min_shared_relationships < 1:
        raise FraudIntelligenceContractError(
            "min_shared_relationships must be at least 1."
        )

    graph = build_investigation_graph(relationships)

    candidates: list[FraudRingCandidate] = []

    for component in nx.connected_components(graph):
        component_relationships = _component_relationships(
            graph,
            component,
        )

        component_entities = _component_entities(
            graph,
            component,
        )

        if len(component_entities) < min_entities:
            continue

        if len(component_relationships) < min_shared_relationships:
            continue

        # A component qualifies as relevant only when the target
        # transaction itself belongs to it.
        target_node = (
            "transaction",
            str(target_transaction_id),
        )

        if target_node not in component:
            continue

        candidate_id = _candidate_id(component_entities)

        candidates.append(
            FraudRingCandidate(
                candidate_id=candidate_id,
                entities=component_entities,
                relationships=component_relationships,
                connected_component_size=len(component),
                shared_relationship_count=len(
                    component_relationships
                ),
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.entity_count,
            -candidate.relationship_count,
            candidate.candidate_id,
        )
    )

    return FraudRingCandidateDetectionResult(
        target_transaction_id=str(target_transaction_id),
        candidates=tuple(candidates),
    )


# ---------------------------------------------------------------------------
# General candidate discovery
# ---------------------------------------------------------------------------


def detect_connected_candidates(
    *,
    relationships: Sequence[EntityRelationship],
    min_entities: int = DEFAULT_MIN_ENTITIES,
    min_relationships: int = DEFAULT_MIN_SHARED_RELATIONSHIPS,
) -> tuple[FraudRingCandidate, ...]:
    """
    Discover all structurally connected candidate groups.

    Unlike detect_fraud_ring_candidates(), this function does not require
    a target transaction.

    It is useful for later network-wide investigation workflows.
    """

    if min_entities < 2:
        raise FraudIntelligenceContractError(
            "min_entities must be at least 2."
        )

    if min_relationships < 1:
        raise FraudIntelligenceContractError(
            "min_relationships must be at least 1."
        )

    graph = build_investigation_graph(relationships)

    candidates: list[FraudRingCandidate] = []

    for component in nx.connected_components(graph):
        component_relationships = _component_relationships(
            graph,
            component,
        )

        component_entities = _component_entities(
            graph,
            component,
        )

        if len(component_entities) < min_entities:
            continue

        if len(component_relationships) < min_relationships:
            continue

        candidates.append(
            FraudRingCandidate(
                candidate_id=_candidate_id(component_entities),
                entities=component_entities,
                relationships=component_relationships,
                connected_component_size=len(component),
                shared_relationship_count=len(
                    component_relationships
                ),
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.entity_count,
            -candidate.relationship_count,
            candidate.candidate_id,
        )
    )

    return tuple(candidates)


# ---------------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------------


def validate_fraud_ring_candidate(
    candidate: FraudRingCandidate,
) -> None:
    """
    Validate one candidate ring.
    """

    if not candidate.candidate_id:
        raise FraudIntelligenceContractError(
            "candidate_id must be non-empty."
        )

    if candidate.connected_component_size != candidate.entity_count:
        raise FraudIntelligenceContractError(
            "connected_component_size must match entity_count."
        )

    if candidate.shared_relationship_count != (
        candidate.relationship_count
    ):
        raise FraudIntelligenceContractError(
            "shared_relationship_count must match relationship_count."
        )

    entity_keys = {
        (
            entity.entity_type,
            entity.entity_id,
        )
        for entity in candidate.entities
    }

    if len(entity_keys) != len(candidate.entities):
        raise FraudIntelligenceContractError(
            "Candidate contains duplicate entities."
        )

    for relationship in candidate.relationships:
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
                "Candidate relationship source is outside "
                "the candidate entity set."
            )

        if target_key not in entity_keys:
            raise FraudIntelligenceContractError(
                "Candidate relationship target is outside "
                "the candidate entity set."
            )


def validate_candidate_detection_result(
    result: FraudRingCandidateDetectionResult,
) -> None:
    """
    Validate complete candidate detection output.
    """

    if not result.target_transaction_id:
        raise FraudIntelligenceContractError(
            "target_transaction_id must be non-empty."
        )

    candidate_ids: set[str] = set()

    for candidate in result.candidates:
        validate_fraud_ring_candidate(candidate)

        if candidate.candidate_id in candidate_ids:
            raise FraudIntelligenceContractError(
                f"Duplicate candidate ID: {candidate.candidate_id!r}"
            )

        candidate_ids.add(candidate.candidate_id)

        target_key = (
            "transaction",
            result.target_transaction_id,
        )

        entity_keys = {
            (
                entity.entity_type,
                entity.entity_id,
            )
            for entity in candidate.entities
        }

        if target_key not in entity_keys:
            raise FraudIntelligenceContractError(
                "Every target transaction candidate must contain "
                "the target transaction."
            )