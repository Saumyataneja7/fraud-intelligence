from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


class GNNRankingError(ValueError):
    """Raised when ranking inputs are invalid."""


@dataclass(frozen=True)
class GNNRankingMetrics:
    """Ranking metrics for one K value."""

    split_name: str
    k: int
    precision_at_k: float
    recall_at_k: float
    fraud_count_at_k: int
    total_fraud: int
    support: int


@dataclass(frozen=True)
class GNNRankingResult:
    """Ranking metrics across multiple K values."""

    split_name: str
    metrics: tuple[GNNRankingMetrics, ...]

    @property
    def k_values(self) -> tuple[int, ...]:
        return tuple(
            metric.k
            for metric in self.metrics
        )


def calculate_precision_at_k(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    k: int,
) -> float:
    """Calculate Precision@K using descending fraud probability."""

    metrics = calculate_ranking_metrics(
        labels=labels,
        probabilities=probabilities,
        k=k,
    )

    return metrics.precision_at_k


def calculate_recall_at_k(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    k: int,
) -> float:
    """Calculate Recall@K using descending fraud probability."""

    metrics = calculate_ranking_metrics(
        labels=labels,
        probabilities=probabilities,
        k=k,
    )

    return metrics.recall_at_k


def calculate_ranking_metrics(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    k: int,
    split_name: str = "test",
) -> GNNRankingMetrics:
    """
    Calculate Precision@K and Recall@K.

    Transactions are ranked by descending predicted fraud
    probability. Stable sorting is used so equal probabilities
    have deterministic ordering.
    """

    _validate_inputs(
        labels=labels,
        probabilities=probabilities,
        split_name=split_name,
    )

    if not isinstance(k, int) or isinstance(k, bool):
        raise GNNRankingError(
            "k must be an integer."
        )

    if k <= 0:
        raise GNNRankingError(
            "k must be greater than zero."
        )

    support = labels.shape[0]

    if k > support:
        raise GNNRankingError(
            f"k cannot exceed support={support}."
        )

    labels_np = labels.cpu().numpy()
    probabilities_np = probabilities.cpu().numpy()

    # Stable descending ranking.
    ranking = np.argsort(
        -probabilities_np,
        kind="mergesort",
    )

    top_k_indices = ranking[:k]

    top_k_labels = labels_np[
        top_k_indices
    ]

    fraud_count_at_k = int(
        top_k_labels.sum()
    )

    total_fraud = int(
        labels_np.sum()
    )

    precision_at_k = fraud_count_at_k / k

    if total_fraud == 0:
        raise GNNRankingError(
            "Recall@K is undefined when total fraud is zero."
        )

    recall_at_k = fraud_count_at_k / total_fraud

    return GNNRankingMetrics(
        split_name=split_name,
        k=k,
        precision_at_k=float(precision_at_k),
        recall_at_k=float(recall_at_k),
        fraud_count_at_k=fraud_count_at_k,
        total_fraud=total_fraud,
        support=support,
    )


def calculate_ranking_curve(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    k_values: tuple[int, ...] | list[int],
    split_name: str = "test",
) -> GNNRankingResult:
    """Calculate Precision@K and Recall@K for multiple K values."""

    if not k_values:
        raise GNNRankingError(
            "k_values must not be empty."
        )

    metrics = tuple(
        calculate_ranking_metrics(
            labels=labels,
            probabilities=probabilities,
            k=k,
            split_name=split_name,
        )
        for k in k_values
    )

    return GNNRankingResult(
        split_name=split_name,
        metrics=metrics,
    )


def _validate_inputs(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    split_name: str,
) -> None:
    if not isinstance(labels, torch.Tensor):
        raise GNNRankingError(
            "labels must be a torch.Tensor."
        )

    if labels.ndim != 1:
        raise GNNRankingError(
            "labels must be 1-dimensional."
        )

    if labels.dtype != torch.long:
        raise GNNRankingError(
            "labels must use torch.long."
        )

    unique_labels = torch.unique(labels)

    if not torch.all(
        (unique_labels == 0)
        | (unique_labels == 1)
    ):
        raise GNNRankingError(
            "labels must contain only 0 and 1."
        )

    if not isinstance(probabilities, torch.Tensor):
        raise GNNRankingError(
            "probabilities must be a torch.Tensor."
        )

    if probabilities.ndim != 1:
        raise GNNRankingError(
            "probabilities must be 1-dimensional."
        )

    if not torch.is_floating_point(probabilities):
        raise GNNRankingError(
            "probabilities must be floating-point."
        )

    if labels.shape[0] != probabilities.shape[0]:
        raise GNNRankingError(
            "labels and probabilities must have equal length."
        )

    if labels.shape[0] == 0:
        raise GNNRankingError(
            "labels and probabilities cannot be empty."
        )

    if not torch.isfinite(probabilities).all():
        raise GNNRankingError(
            "probabilities must contain only finite values."
        )

    if torch.any(probabilities < 0) or torch.any(
        probabilities > 1
    ):
        raise GNNRankingError(
            "probabilities must be in the range [0, 1]."
        )

    if not isinstance(split_name, str) or not split_name:
        raise GNNRankingError(
            "split_name must be a non-empty string."
        )