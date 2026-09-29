from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class APIContractBase(BaseModel):
    """Base configuration shared by API response/request models."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class HealthResponse(APIContractBase):
    """Health endpoint response."""

    status: str = Field(
        ...,
        description="API health status.",
    )
    service: str = Field(
        ...,
        description="Service name.",
    )
    version: str = Field(
        ...,
        description="API version.",
    )


class APIErrorResponse(APIContractBase):
    """Standard API error response."""

    error: str = Field(
        ...,
        min_length=1,
    )
    message: str = Field(
        ...,
        min_length=1,
    )
    detail: str | None = None


class EntityReferenceResponse(APIContractBase):
    """Reference to an investigation entity."""

    entity_type: str = Field(
        ...,
        min_length=1,
    )
    entity_id: str = Field(
        ...,
        min_length=1,
    )