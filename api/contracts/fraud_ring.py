from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .common import APIContractBase, EntityReferenceResponse


class RingTransactionResponse(APIContractBase):
    """Transaction belonging to a candidate network."""

    transaction_id: str = Field(
        ...,
        min_length=1,
    )
    timestamp: datetime | None = None


class RingEvidenceResponse(APIContractBase):
    """Structural evidence associated with a candidate network."""

    evidence_type: str = Field(
        ...,
        min_length=1,
    )
    description: str = Field(
        ...,
        min_length=1,
    )
    entity_types: list[str] = Field(
        default_factory=list,
    )
    relationship_types: list[str] = Field(
        default_factory=list,
    )
    strength: str = Field(
        ...,
        min_length=1,
    )


class RingNetworkScoreResponse(APIContractBase):
    """Structural score for a candidate fraud network."""

    candidate_id: str = Field(
        ...,
        min_length=1,
    )
    entity_count: int = Field(
        ...,
        ge=0,
    )
    relationship_count: int = Field(
        ...,
        ge=0,
    )
    relationship_density: float = Field(
        ...,
        ge=0.0,
    )
    entity_type_count: int = Field(
        ...,
        ge=0,
    )
    relationship_type_count: int = Field(
        ...,
        ge=0,
    )
    non_transaction_entity_count: int = Field(
        ...,
        ge=0,
    )
    non_transaction_connectivity: float = Field(
        ...,
        ge=0.0,
    )
    structural_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
    )


class FraudRingInvestigationResponse(APIContractBase):
    """Investigation view for a candidate fraud network."""

    candidate_id: str = Field(
        ...,
        min_length=1,
    )
    entities: list[EntityReferenceResponse] = Field(
        default_factory=list,
    )
    relationships: list[dict] = Field(
        default_factory=list,
    )
    transactions: list[RingTransactionResponse] = Field(
        default_factory=list,
    )
    network_score: RingNetworkScoreResponse
    evidence: list[RingEvidenceResponse] = Field(
        default_factory=list,
    )
    entity_type_counts: dict[str, int] = Field(
        default_factory=dict,
    )
    relationship_type_counts: dict[str, int] = Field(
        default_factory=dict,
    )
    investigation_summary: str = Field(
        ...,
        min_length=1,
    )
    temporal_rule: str = Field(
        ...,
        min_length=1,
    )