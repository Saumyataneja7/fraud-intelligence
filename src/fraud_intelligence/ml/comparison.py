from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fraud_intelligence.ml.evaluation import (
    EvaluationResult,
)
from fraud_intelligence.ml.ranking import (
    RankingMetrics,
)


class ModelComparisonError(ValueError):
    """Raised when model comparison inputs are invalid."""


@dataclass(frozen=True)
class ModelComparisonRow:
    model_name: str
    split_name: str
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


@dataclass(frozen=True)
class ModelComparisonResult:
    rows: tuple[ModelComparisonRow, ...]


def _validate_evaluation_results(
    results: list[EvaluationResult]
    | tuple[EvaluationResult, ...],
) -> None:
    if not results:
        raise ModelComparisonError(
            "At least one evaluation result is required."
        )

    for result in results:
        if not isinstance(
            result,
            EvaluationResult,
        ):
            raise ModelComparisonError(
                "All results must be EvaluationResult instances."
            )


def build_model_comparison(
    results: list[EvaluationResult]
    | tuple[EvaluationResult, ...],
) -> ModelComparisonResult:
    _validate_evaluation_results(results)

    rows = tuple(
        ModelComparisonRow(
            model_name=result.model_name,
            split_name=result.split_name,
            precision=result.metrics.precision,
            recall=result.metrics.recall,
            f1=result.metrics.f1,
            pr_auc=result.metrics.pr_auc,
            roc_auc=result.metrics.roc_auc,
            true_negatives=result.metrics.true_negatives,
            false_positives=result.metrics.false_positives,
            false_negatives=result.metrics.false_negatives,
            true_positives=result.metrics.true_positives,
        )
        for result in results
    )

    return ModelComparisonResult(rows=rows)


def comparison_to_dataframe(
    comparison: ModelComparisonResult,
) -> pd.DataFrame:
    if not isinstance(
        comparison,
        ModelComparisonResult,
    ):
        raise ModelComparisonError(
            "comparison must be a ModelComparisonResult."
        )

    if not comparison.rows:
        raise ModelComparisonError(
            "Comparison contains no rows."
        )

    return pd.DataFrame(
        {
            "model_name": row.model_name,
            "split_name": row.split_name,
            "precision": row.precision,
            "recall": row.recall,
            "f1": row.f1,
            "pr_auc": row.pr_auc,
            "roc_auc": row.roc_auc,
            "true_negatives": row.true_negatives,
            "false_positives": row.false_positives,
            "false_negatives": row.false_negatives,
            "true_positives": row.true_positives,
        }
        for row in comparison.rows
    )


def build_ranking_comparison(
    ranking_results: dict[
        str,
        list[RankingMetrics]
        | tuple[RankingMetrics, ...],
    ],
) -> pd.DataFrame:
    if not ranking_results:
        raise ModelComparisonError(
            "At least one model ranking result is required."
        )

    rows: list[dict[str, object]] = []

    for model_name, metrics in ranking_results.items():
        if not model_name.strip():
            raise ModelComparisonError(
                "Model name cannot be empty."
            )

        if not metrics:
            raise ModelComparisonError(
                f"No ranking metrics supplied for {model_name}."
            )

        for metric in metrics:
            if not isinstance(
                metric,
                RankingMetrics,
            ):
                raise ModelComparisonError(
                    "Ranking results must contain RankingMetrics."
                )

            rows.append(
                {
                    "model_name": model_name,
                    "k": metric.k,
                    "precision_at_k": (
                        metric.precision_at_k
                    ),
                    "recall_at_k": (
                        metric.recall_at_k
                    ),
                    "true_positives_at_k": (
                        metric.true_positives_at_k
                    ),
                    "actual_positive_count": (
                        metric.actual_positive_count
                    ),
                    "reviewed_count": metric.reviewed_count,
                }
            )

    return pd.DataFrame(rows)


def validate_comparison_schema(
    comparison: pd.DataFrame,
) -> None:
    required_columns = {
        "model_name",
        "split_name",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    }

    missing = required_columns - set(
        comparison.columns
    )

    if missing:
        raise ModelComparisonError(
            "Comparison is missing required columns: "
            + ", ".join(sorted(missing))
        )