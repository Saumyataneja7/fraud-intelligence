from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .common import APIContractBase, EntityReferenceResponse


class RelatedEntityResponse(APIContractBase):
    """Entity related to an investigation target."""

    entity: EntityReferenceResponse
    relationship_type: str = Field(
        ...,
        min_length=1,
    )
    direction: str = Field(
        ...,
        min_length=1,
    )
    timestamp: datetime | None = None


class EntityNetworkScoreResponse(APIContractBase):
    """Structural network score associated with an entity."""

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


class EntityInvestigationResponse(APIContractBase):
    """Investigation view for a customer or other graph entity."""

    entity: EntityReferenceResponse
    related_entities: list[RelatedEntityResponse] = Field(
        default_factory=list,
    )
    related_transactions: list[str] = Field(
        default_factory=list,
    )
    candidate_ring_ids: list[str] = Field(
        default_factory=list,
    )
    network_scores: list[EntityNetworkScoreResponse] = Field(
        default_factory=list,
    )
    investigation_summary: str = Field(
        ...,
        min_length=1,
    )
    temporal_rule: str = Field(
        ...,
        min_length=1,
    )