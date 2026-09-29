from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from api.contracts.model_metrics import (
    ClassificationMetricsResponse,
    ModelEvaluationResponse,
    ModelMetricsResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
    / "classical_model_metrics.json"
)


class ModelMetricsServiceError(RuntimeError):
    """Raised when model metrics cannot be loaded."""


@dataclass(frozen=True)
class ModelMetricsService:
    """Read-only adapter for frozen Phase 5 model metrics."""

    metrics_path: Path = MODEL_METRICS_PATH

    def _load_metrics(self) -> dict[str, Any]:
        if not self.metrics_path.exists():
            raise ModelMetricsServiceError(
                f"Model metrics artifact not found: "
                f"{self.metrics_path}"
            )

        try:
            with self.metrics_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)
        except Exception as exc:
            raise ModelMetricsServiceError(
                "Failed to load model metrics artifact."
            ) from exc

        if not isinstance(payload, dict):
            raise ModelMetricsServiceError(
                "Model metrics artifact must contain a JSON object."
            )

        return payload

    @staticmethod
    def _validate_payload(
        payload: dict[str, Any],
    ) -> None:
        required = {
            "feature_dataset",
            "feature_count",
            "models",
            "splits",
            "results",
        }

        missing = required - set(payload)

        if missing:
            raise ModelMetricsServiceError(
                "Model metrics artifact is missing required "
                "fields: "
                + ", ".join(sorted(missing))
            )

        if not isinstance(
            payload["results"],
            list,
        ):
            raise ModelMetricsServiceError(
                "Model metrics results must be a list."
            )

    def get_metrics(self) -> ModelMetricsResponse:
        payload = self._load_metrics()

        self._validate_payload(payload)

        try:
            results = []

            for result in payload["results"]:
                metrics = result["metrics"]

                results.append(
                    ModelEvaluationResponse(
                        model_name=result["model_name"],
                        split_name=result["split_name"],
                        metrics=ClassificationMetricsResponse(
                            precision=metrics["precision"],
                            recall=metrics["recall"],
                            f1=metrics["f1"],
                            pr_auc=metrics["pr_auc"],
                            roc_auc=metrics["roc_auc"],
                            true_negatives=(
                                metrics["true_negatives"]
                            ),
                            false_positives=(
                                metrics["false_positives"]
                            ),
                            false_negatives=(
                                metrics["false_negatives"]
                            ),
                            true_positives=(
                                metrics["true_positives"]
                            ),
                            support=(
                                metrics["true_negatives"]
                                + metrics["false_positives"]
                                + metrics["false_negatives"]
                                + metrics["true_positives"]
                            ),
                            predicted_positive_count=(
                                metrics["false_positives"]
                                + metrics["true_positives"]
                            ),
                            actual_positive_count=(
                                metrics["false_negatives"]
                                + metrics["true_positives"]
                            ),
                        ),
                    )
                )

            return ModelMetricsResponse(
                feature_dataset=payload["feature_dataset"],
                feature_count=payload["feature_count"],
                models=list(payload["models"]),
                splits=list(payload["splits"]),
                results=results,
            )

        except (KeyError, TypeError, ValueError) as exc:
            raise ModelMetricsServiceError(
                "Model metrics artifact has an invalid schema."
            ) from exc
