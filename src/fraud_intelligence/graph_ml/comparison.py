from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fraud_intelligence.ml.evaluation import EvaluationResult
from fraud_intelligence.graph_ml.evaluation import GNNEvaluationResult
from fraud_intelligence.graph_ml.ranking import GNNRankingResult


class GNNComparisonError(ValueError):
    """Raised when model-comparison inputs are invalid."""


@dataclass(frozen=True)
class ModelComparisonResult:
    """Comparison table for classical ML and GNN models."""

    metrics: pd.DataFrame

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(self.metrics["model"].tolist())

    @property
    def row_count(self) -> int:
        return len(self.metrics)


COMPARISON_COLUMNS = (
    "model",
    "model_family",
    "split",
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "roc_auc",
    "predicted_positives",
    "actual_positives",
)


RANKING_COLUMNS = (
    "model",
    "model_family",
    "split",
    "k",
    "precision_at_k",
    "recall_at_k",
)


def compare_classical_and_gnn(
    classical_results: list[EvaluationResult],
    gnn_result: GNNEvaluationResult,
    split_name: str,
) -> ModelComparisonResult:
    """
    Build a metric comparison between classical ML and GNN.

    The function is descriptive only and does not rank models.
    """

    if not classical_results:
        raise GNNComparisonError(
            "classical_results must not be empty."
        )

    if not isinstance(
        gnn_result,
        GNNEvaluationResult,
    ):
        raise GNNComparisonError(
            "gnn_result must be a GNNEvaluationResult."
        )

    _validate_split_name(split_name)

    rows: list[dict] = []

    for result in classical_results:
        if not isinstance(
            result,
            EvaluationResult,
        ):
            raise GNNComparisonError(
                "All classical results must be EvaluationResult objects."
            )

        rows.append(
            {
                "model": result.model_name,
                "model_family": "classical_ml",
                "split": split_name,
                "precision": result.metrics.precision,
                "recall": result.metrics.recall,
                "f1": result.metrics.f1,
                "pr_auc": result.metrics.pr_auc,
                "roc_auc": result.metrics.roc_auc,
                "predicted_positives": (
                    result.metrics.predicted_positive_count
                ),
                "actual_positives": result.metrics.actual_positive_count,
            }
        )

    rows.append(
        {
            "model": "GraphSAGE",
            "model_family": "gnn",
            "split": split_name,
            "precision": gnn_result.precision,
            "recall": gnn_result.recall,
            "f1": gnn_result.f1,
            "pr_auc": gnn_result.pr_auc,
            "roc_auc": gnn_result.roc_auc,
            "predicted_positives": (
                gnn_result.predicted_positive
            ),
            "actual_positives": gnn_result.actual_positive,
        }
    )

    frame = pd.DataFrame(
        rows,
        columns=COMPARISON_COLUMNS,
    )

    validate_model_comparison(frame)

    return ModelComparisonResult(
        metrics=frame,
    )


def compare_ranking_results(
    classical_ranking: pd.DataFrame | None,
    gnn_ranking: GNNRankingResult,
    split_name: str,
) -> pd.DataFrame:
    """
    Compare ranking metrics across model families.

    Classical ranking may be omitted when it has not yet been
    materialized. The function never invents missing metrics.
    """

    if not isinstance(
        gnn_ranking,
        GNNRankingResult,
    ):
        raise GNNComparisonError(
            "gnn_ranking must be a GNNRankingResult."
        )

    _validate_split_name(split_name)

    rows: list[dict] = []

    if classical_ranking is not None:
        _validate_ranking_frame(
            classical_ranking,
        )

        for row in classical_ranking.itertuples(
            index=False,
        ):
            rows.append(
                {
                    "model": row.model,
                    "model_family": "classical_ml",
                    "split": split_name,
                    "k": int(row.k),
                    "precision_at_k": float(
                        row.precision_at_k
                    ),
                    "recall_at_k": float(
                        row.recall_at_k
                    ),
                }
            )

    for metric in gnn_ranking.metrics:
        rows.append(
            {
                "model": "GraphSAGE",
                "model_family": "gnn",
                "split": split_name,
                "k": metric.k,
                "precision_at_k": metric.precision_at_k,
                "recall_at_k": metric.recall_at_k,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=RANKING_COLUMNS,
    )

    validate_ranking_comparison(frame)

    return frame


def validate_model_comparison(
    frame: pd.DataFrame,
) -> None:
    """Validate the standard model-comparison schema."""

    if not isinstance(frame, pd.DataFrame):
        raise GNNComparisonError(
            "Comparison must be a pandas DataFrame."
        )

    missing = set(COMPARISON_COLUMNS) - set(
        frame.columns
    )

    if missing:
        raise GNNComparisonError(
            f"Missing comparison columns: {sorted(missing)}"
        )

    if frame.empty:
        raise GNNComparisonError(
            "Comparison cannot be empty."
        )

    if frame["model"].isna().any():
        raise GNNComparisonError(
            "model cannot contain null values."
        )

    if frame["split"].isna().any():
        raise GNNComparisonError(
            "split cannot contain null values."
        )

    for column in (
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
    ):
        if not frame[column].between(
            0,
            1,
        ).all():
            raise GNNComparisonError(
                f"{column} must be in [0, 1]."
            )

    for column in (
        "predicted_positives",
        "actual_positives",
    ):
        if (frame[column] < 0).any():
            raise GNNComparisonError(
                f"{column} cannot be negative."
            )


def validate_ranking_comparison(
    frame: pd.DataFrame,
) -> None:
    """Validate Precision@K / Recall@K comparison schema."""

    if not isinstance(frame, pd.DataFrame):
        raise GNNComparisonError(
            "Ranking comparison must be a pandas DataFrame."
        )

    missing = set(RANKING_COLUMNS) - set(
        frame.columns
    )

    if missing:
        raise GNNComparisonError(
            f"Missing ranking columns: {sorted(missing)}"
        )

    if frame.empty:
        raise GNNComparisonError(
            "Ranking comparison cannot be empty."
        )

    if (frame["k"] <= 0).any():
        raise GNNComparisonError(
            "k must be greater than zero."
        )

    for column in (
        "precision_at_k",
        "recall_at_k",
    ):
        if not frame[column].between(
            0,
            1,
        ).all():
            raise GNNComparisonError(
                f"{column} must be in [0, 1]."
            )


def _validate_split_name(
    split_name: str,
) -> None:
    if not isinstance(
        split_name,
        str,
    ) or not split_name:
        raise GNNComparisonError(
            "split_name must be a non-empty string."
        )


def _validate_ranking_frame(
    frame: pd.DataFrame,
) -> None:
    required = {
        "model",
        "k",
        "precision_at_k",
        "recall_at_k",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise GNNComparisonError(
            "Missing classical ranking columns: "
            f"{sorted(missing)}"
        )

    if frame.empty:
        raise GNNComparisonError(
            "Classical ranking frame cannot be empty."
        )