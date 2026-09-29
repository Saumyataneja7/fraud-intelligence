from __future__ import annotations

from pydantic import Field

from .common import APIContractBase


class PredictionRequest(APIContractBase):
    """Request for transaction fraud prediction."""

    transaction_id: str = Field(
        ...,
        min_length=1,
        description="Transaction identifier.",
    )


class PredictionResponse(APIContractBase):
    """Prediction result returned by the API."""

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
    model_name: str = Field(
        ...,
        min_length=1,
    )
    model_family: str = Field(
        ...,
        min_length=1,
    )