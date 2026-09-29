from __future__ import annotations

from pydantic import Field

from api.contracts.common import APIContractBase


class ClassificationMetricsResponse(APIContractBase):
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    pr_auc: float = Field(..., ge=0.0, le=1.0)
    roc_auc: float = Field(..., ge=0.0, le=1.0)

    true_negatives: int = Field(..., ge=0)
    false_positives: int = Field(..., ge=0)
    false_negatives: int = Field(..., ge=0)
    true_positives: int = Field(..., ge=0)

    support: int = Field(..., ge=0)
    predicted_positive_count: int = Field(..., ge=0)
    actual_positive_count: int = Field(..., ge=0)


class ModelEvaluationResponse(APIContractBase):
    model_name: str
    split_name: str
    metrics: ClassificationMetricsResponse


class ModelMetricsResponse(APIContractBase):
    feature_dataset: str
    feature_count: int = Field(..., ge=0)
    models: list[str]
    splits: list[str]
    results: list[ModelEvaluationResponse]
