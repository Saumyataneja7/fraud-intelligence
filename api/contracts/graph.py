from __future__ import annotations

from pydantic import Field

from .common import APIContractBase


class GraphNodeResponse(APIContractBase):
    """Graph node exposed by the API."""

    node_type: str = Field(
        ...,
        min_length=1,
    )
    node_id: int = Field(
        ...,
        ge=0,
    )


class GraphEdgeResponse(APIContractBase):
    """Graph edge exposed by the API."""

    relationship_type: str = Field(
        ...,
        min_length=1,
    )
    source_node_type: str = Field(
        ...,
        min_length=1,
    )
    source_node_id: int = Field(
        ...,
        ge=0,
    )
    target_node_type: str = Field(
        ...,
        min_length=1,
    )
    target_node_id: int = Field(
        ...,
        ge=0,
    )


class GraphResponse(APIContractBase):
    """Graph representation returned by an investigation endpoint."""

    target_node_type: str = Field(
        ...,
        min_length=1,
    )
    target_node_id: int = Field(
        ...,
        ge=0,
    )
    nodes: list[GraphNodeResponse] = Field(
        default_factory=list,
    )
    edges: list[GraphEdgeResponse] = Field(
        default_factory=list,
    )