from __future__ import annotations

from pydantic import Field

from .common import APIContractBase


class FeatureAttributionResponse(APIContractBase):
    """Feature attribution exposed through the API."""

    feature: str = Field(
        ...,
        min_length=1,
    )
    attribution: float
    absolute_attribution: float = Field(
        ...,
        ge=0.0,
    )
    rank: int = Field(
        ...,
        ge=1,
    )


class GraphFindingResponse(APIContractBase):
    """Human-readable graph finding."""

    finding_type: str = Field(
        ...,
        min_length=1,
    )
    description: str = Field(
        ...,
        min_length=1,
    )


class ExplanationResponse(APIContractBase):
    """Model explanation response."""

    transaction_id: str = Field(
        ...,
        min_length=1,
    )
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
    feature_attributions: list[FeatureAttributionResponse] = Field(
        default_factory=list,
    )
    graph_findings: list[GraphFindingResponse] = Field(
        default_factory=list,
    )
    summary: str = Field(
        ...,
        min_length=1,
    )