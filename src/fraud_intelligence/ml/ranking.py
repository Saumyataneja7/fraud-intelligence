from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class RankingEvaluationError(ValueError):
    """Raised when ranking evaluation inputs are invalid."""


@dataclass(frozen=True)
class RankingMetrics:
    k: int
    precision_at_k: float
    recall_at_k: float
    true_positives_at_k: int
    actual_positive_count: int
    reviewed_count: int


def _validate_inputs(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_true.ndim != 1:
        raise RankingEvaluationError(
            "y_true must be one-dimensional."
        )

    if y_proba.ndim != 1:
        raise RankingEvaluationError(
            "y_proba must be one-dimensional."
        )

    if len(y_true) != len(y_proba):
        raise RankingEvaluationError(
            "y_true and y_proba must have the same length."
        )

    if len(y_true) == 0:
        raise RankingEvaluationError(
            "Ranking evaluation data cannot be empty."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise RankingEvaluationError(
            "y_true must contain only 0 and 1."
        )

    if np.unique(y_true).size < 2:
        raise RankingEvaluationError(
            "y_true must contain both classes."
        )

    if not np.isfinite(y_proba).all():
        raise RankingEvaluationError(
            "y_proba must contain only finite values."
        )

    if np.any(y_proba < 0.0) or np.any(y_proba > 1.0):
        raise RankingEvaluationError(
            "y_proba must contain values between 0 and 1."
        )

    return y_true, y_proba


def calculate_precision_recall_at_k(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    k: int,
) -> RankingMetrics:
    y_true, y_proba = _validate_inputs(
        y_true,
        y_proba,
    )

    if not isinstance(k, (int, np.integer)):
        raise RankingEvaluationError(
            "k must be an integer."
        )

    if k <= 0:
        raise RankingEvaluationError(
            "k must be greater than zero."
        )

    if k > len(y_true):
        raise RankingEvaluationError(
            "k cannot exceed the number of observations."
        )

    # Stable descending ranking.
    ranked_indices = np.argsort(
        -y_proba,
        kind="mergesort",
    )

    top_k_indices = ranked_indices[:k]

    top_k_labels = y_true[top_k_indices]

    true_positives = int(top_k_labels.sum())
    actual_positive_count = int(y_true.sum())

    precision_at_k = (
        true_positives / k
    )

    recall_at_k = (
        true_positives / actual_positive_count
    )

    return RankingMetrics(
        k=int(k),
        precision_at_k=float(precision_at_k),
        recall_at_k=float(recall_at_k),
        true_positives_at_k=true_positives,
        actual_positive_count=actual_positive_count,
        reviewed_count=int(k),
    )


def calculate_ranking_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    k_values: list[int] | tuple[int, ...],
) -> tuple[RankingMetrics, ...]:
    if not k_values:
        raise RankingEvaluationError(
            "At least one k value is required."
        )

    unique_k_values = tuple(dict.fromkeys(k_values))

    return tuple(
        calculate_precision_recall_at_k(
            y_true,
            y_proba,
            k,
        )
        for k in unique_k_values
    )


def ranking_metrics_to_dataframe(
    metrics: tuple[RankingMetrics, ...]
    | list[RankingMetrics],
) -> pd.DataFrame:
    if not metrics:
        raise RankingEvaluationError(
            "At least one ranking metric is required."
        )

    return pd.DataFrame(
        {
            "k": metric.k,
            "precision_at_k": metric.precision_at_k,
            "recall_at_k": metric.recall_at_k,
            "true_positives_at_k": (
                metric.true_positives_at_k
            ),
            "actual_positive_count": (
                metric.actual_positive_count
            ),
            "reviewed_count": metric.reviewed_count,
        }
        for metric in metrics
    )