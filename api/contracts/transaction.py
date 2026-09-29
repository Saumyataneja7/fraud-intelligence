from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .common import APIContractBase, EntityReferenceResponse


class SuspicionSignalResponse(APIContractBase):
    """API representation of a transaction suspicion signal."""

    name: str = Field(
        ...,
        min_length=1,
    )
    value: float | int | bool | str
    description: str = Field(
        ...,
        min_length=1,
    )
    source: str = Field(
        ...,
        min_length=1,
    )


class TransactionRelationshipResponse(APIContractBase):
    """API representation of a transaction relationship."""

    relationship_type: str = Field(
        ...,
        min_length=1,
    )
    source: EntityReferenceResponse
    target: EntityReferenceResponse
    timestamp: datetime | None = None


class TransactionInvestigationResponse(APIContractBase):
    """Investigation view for a transaction."""

    transaction_id: str = Field(
        ...,
        min_length=1,
    )
    timestamp: datetime
    prediction_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )
    prediction_label: int = Field(
        ...,
        ge=0,
        le=1,
    )
    suspicion_signals: list[SuspicionSignalResponse] = Field(
        default_factory=list,
    )
    related_entities: list[EntityReferenceResponse] = Field(
        default_factory=list,
    )
    relationships: list[TransactionRelationshipResponse] = Field(
        default_factory=list,
    )
    candidate_ring_ids: list[str] = Field(
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