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
from .fraud_ring_evidence import FraudRingEvidence
from .ring_network_scoring import RingNetworkScore
from .suspicious_transaction import (
    SuspiciousTransactionIntelligence,
)


FORBIDDEN_TERMS: frozenset[str] = frozenset(
    {
        "is_fraud",
        "fraud_scenario",
    }
)


@dataclass(frozen=True)
class TransactionInvestigationView:
    """
    Complete investigation view for one transaction.

    This combines existing predictive/explanation information with
    descriptive fraud-intelligence context.

    It does not modify the underlying model prediction.
    """

    transaction_id: str
    timestamp: object

    prediction_probability: float
    prediction_label: int

    suspicion_signals: tuple
    related_entities: tuple[EntityReference, ...]
    relationships: tuple[EntityRelationship, ...]

    candidate_ring_ids: tuple[str, ...]
    network_scores: tuple[RingNetworkScore, ...]

    ring_evidence: tuple[FraudRingEvidence, ...]

    investigation_summary: str
    temporal_rule: str

    @property
    def suspicion_signal_count(self) -> int:
        return len(self.suspicion_signals)

    @property
    def related_entity_count(self) -> int:
        return len(self.related_entities)

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)

    @property
    def candidate_ring_count(self) -> int:
        return len(self.candidate_ring_ids)

    @property
    def evidence_count(self) -> int:
        return sum(
            evidence.evidence_count
            for evidence in self.ring_evidence
        )


# ---------------------------------------------------------------------------
# Relationship helpers
# ---------------------------------------------------------------------------


def extract_transaction_entities(
    *,
    transaction_id: str,
    relationships: Sequence[EntityRelationship],
) -> tuple[EntityReference, ...]:
    """
    Extract all unique entities directly connected to the transaction.

    The target transaction itself is included.
    """

    if not transaction_id:
        raise FraudIntelligenceContractError(
            "transaction_id must be non-empty."
        )

    transaction_ref = EntityReference(
        "transaction",
        str(transaction_id),
    )

    entities: dict[tuple[str, str], EntityReference] = {
        (
            transaction_ref.entity_type,
            transaction_ref.entity_id,
        ): transaction_ref
    }

    for relationship in relationships:
        validate_relationship(relationship)

        if (
            relationship.source != transaction_ref
            and relationship.target != transaction_ref
        ):
            continue

        for entity in (
            relationship.source,
            relationship.target,
        ):
            entities[
                (
                    entity.entity_type,
                    entity.entity_id,
                )
            ] = entity

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
# Ring helpers
# ---------------------------------------------------------------------------


def extract_transaction_ring_ids(
    *,
    transaction_id: str,
    candidates: Sequence[FraudRingCandidate],
) -> tuple[str, ...]:
    """
    Return candidate ring IDs containing the target transaction.
    """

    target_key = (
        "transaction",
        str(transaction_id),
    )

    ring_ids = []

    for candidate in candidates:
        candidate_keys = {
            (
                entity.entity_type,
                entity.entity_id,
            )
            for entity in candidate.entities
        }

        if target_key in candidate_keys:
            ring_ids.append(candidate.candidate_id)

    return tuple(sorted(set(ring_ids)))


def extract_transaction_network_scores(
    *,
    candidate_ring_ids: Sequence[str],
    network_scores: Sequence[RingNetworkScore],
) -> tuple[RingNetworkScore, ...]:
    """
    Select network scores belonging to this transaction's candidate rings.
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


def extract_transaction_ring_evidence(
    *,
    candidate_ring_ids: Sequence[str],
    ring_evidence: Sequence[FraudRingEvidence],
) -> tuple[FraudRingEvidence, ...]:
    """
    Select evidence packages belonging to this transaction's candidate
    rings.
    """

    candidate_ids = set(candidate_ring_ids)

    return tuple(
        sorted(
            (
                evidence
                for evidence in ring_evidence
                if evidence.candidate_id in candidate_ids
            ),
            key=lambda evidence: evidence.candidate_id,
        )
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_transaction_investigation_summary(
    *,
    transaction_id: str,
    prediction_label: int,
    prediction_probability: float,
    suspicion_signal_count: int,
    related_entity_count: int,
    candidate_ring_count: int,
) -> str:
    """
    Build a descriptive investigation summary.

    The model output is reported as an existing prediction; this function
    does not create a new prediction or threshold.
    """

    prediction_description = (
        "positive"
        if prediction_label == 1
        else "negative"
    )

    return (
        f"Transaction {transaction_id} has an existing model "
        f"prediction of {prediction_description} with probability "
        f"{prediction_probability:.6f}. The investigation context "
        f"contains {suspicion_signal_count} suspiciousness signal(s), "
        f"{related_entity_count} directly related entity(ies), and "
        f"{candidate_ring_count} candidate network group(s). "
        "These intelligence signals provide contextual evidence and "
        "do not replace or modify the model prediction."
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_transaction_investigation_view(
    *,
    transaction_id: str,
    timestamp,
    suspicious_transaction: SuspiciousTransactionIntelligence,
    relationships: Sequence[EntityRelationship],
    candidates: Sequence[FraudRingCandidate] = (),
    network_scores: Sequence[RingNetworkScore] = (),
    ring_evidence: Sequence[FraudRingEvidence] = (),
) -> TransactionInvestigationView:
    """
    Assemble the complete transaction investigation view.

    The supplied suspicious-transaction intelligence is treated as an
    existing prediction/intelligence result. This function does not
    retrain or re-score the model.
    """

    if not transaction_id:
        raise FraudIntelligenceContractError(
            "transaction_id must be non-empty."
        )

    if suspicious_transaction.transaction_id != str(
        transaction_id
    ):
        raise FraudIntelligenceContractError(
            "Transaction IDs do not match."
        )

    if suspicious_transaction.timestamp != timestamp:
        raise FraudIntelligenceContractError(
            "Transaction timestamps do not match."
        )

    related_entities = extract_transaction_entities(
        transaction_id=transaction_id,
        relationships=relationships,
    )

    candidate_ring_ids = extract_transaction_ring_ids(
        transaction_id=transaction_id,
        candidates=candidates,
    )

    selected_network_scores = extract_transaction_network_scores(
        candidate_ring_ids=candidate_ring_ids,
        network_scores=network_scores,
    )

    selected_ring_evidence = extract_transaction_ring_evidence(
        candidate_ring_ids=candidate_ring_ids,
        ring_evidence=ring_evidence,
    )

    summary = build_transaction_investigation_summary(
        transaction_id=str(transaction_id),
        prediction_label=(
            suspicious_transaction.prediction_label
        ),
        prediction_probability=(
            suspicious_transaction.prediction_probability
        ),
        suspicion_signal_count=(
            suspicious_transaction.signal_count
        ),
        related_entity_count=len(related_entities),
        candidate_ring_count=len(candidate_ring_ids),
    )

    return TransactionInvestigationView(
        transaction_id=str(transaction_id),
        timestamp=timestamp,
        prediction_probability=(
            suspicious_transaction.prediction_probability
        ),
        prediction_label=(
            suspicious_transaction.prediction_label
        ),
        suspicion_signals=suspicious_transaction.signals,
        related_entities=related_entities,
        relationships=tuple(relationships),
        candidate_ring_ids=candidate_ring_ids,
        network_scores=selected_network_scores,
        ring_evidence=selected_ring_evidence,
        investigation_summary=summary,
        temporal_rule=(
            "Only validated historical investigation context may "
            "be used. Future context must not be introduced."
        ),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_transaction_investigation_view(
    view: TransactionInvestigationView,
) -> None:
    """
    Validate a complete transaction investigation view.
    """

    if not view.transaction_id:
        raise FraudIntelligenceContractError(
            "transaction_id must be non-empty."
        )

    if not 0.0 <= view.prediction_probability <= 1.0:
        raise FraudIntelligenceContractError(
            "prediction_probability must be between 0 and 1."
        )

    if view.prediction_label not in (0, 1):
        raise FraudIntelligenceContractError(
            "prediction_label must be 0 or 1."
        )

    if not view.investigation_summary:
        raise FraudIntelligenceContractError(
            "investigation_summary must be non-empty."
        )

    if not view.temporal_rule:
        raise FraudIntelligenceContractError(
            "temporal_rule must be non-empty."
        )

    transaction_ref = EntityReference(
        "transaction",
        view.transaction_id,
    )

    entity_keys = {
        (
            entity.entity_type,
            entity.entity_id,
        )
        for entity in view.related_entities
    }

    entity_keys.add(
        (
            transaction_ref.entity_type,
            transaction_ref.entity_id,
        )
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
                "Relationship source is outside the transaction "
                "investigation entity set."
            )

        if target_key not in entity_keys:
            raise FraudIntelligenceContractError(
                "Relationship target is outside the transaction "
                "investigation entity set."
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
                "Duplicate transaction relationship."
            )

        relationship_keys.add(relationship_key)

    ring_ids = tuple(view.candidate_ring_ids)

    if ring_ids != tuple(sorted(ring_ids)):
        raise FraudIntelligenceContractError(
            "candidate_ring_ids must be sorted."
        )

    if len(set(ring_ids)) != len(ring_ids):
        raise FraudIntelligenceContractError(
            "candidate_ring_ids must be unique."
        )

    score_ids = tuple(
        score.candidate_id
        for score in view.network_scores
    )

    if score_ids != tuple(sorted(score_ids)):
        raise FraudIntelligenceContractError(
            "network_scores must be sorted."
        )

    if not set(score_ids).issubset(set(ring_ids)):
        raise FraudIntelligenceContractError(
            "Every network score must correspond to a candidate ring."
        )

    evidence_ids = tuple(
        evidence.candidate_id
        for evidence in view.ring_evidence
    )

    if evidence_ids != tuple(sorted(evidence_ids)):
        raise FraudIntelligenceContractError(
            "ring_evidence must be sorted."
        )

    if not set(evidence_ids).issubset(set(ring_ids)):
        raise FraudIntelligenceContractError(
            "Every ring evidence package must correspond to "
            "a candidate ring."
        )

    combined_text = (
        view.investigation_summary
        + " "
        + view.temporal_rule
    )

    for forbidden_term in FORBIDDEN_TERMS:
        if forbidden_term in combined_text:
            raise FraudIntelligenceContractError(
                "Transaction investigation view contains forbidden "
                f"predictive field: {forbidden_term!r}"
            )