from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


class ThresholdOptimizationError(ValueError):
    """Raised when threshold optimization inputs are invalid."""


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    predicted_positive_count: int
    actual_positive_count: int


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    selected_threshold: float
    objective: str
    metrics: ThresholdMetrics
    candidates: tuple[ThresholdMetrics, ...]


def _validate_inputs(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    thresholds: list[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_true.ndim != 1:
        raise ThresholdOptimizationError(
            "y_true must be one-dimensional."
        )

    if y_proba.ndim != 1:
        raise ThresholdOptimizationError(
            "y_proba must be one-dimensional."
        )

    if len(y_true) != len(y_proba):
        raise ThresholdOptimizationError(
            "y_true and y_proba must have the same length."
        )

    if len(y_true) == 0:
        raise ThresholdOptimizationError(
            "Threshold optimization data cannot be empty."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ThresholdOptimizationError(
            "y_true must contain only 0 and 1."
        )

    if np.unique(y_true).size < 2:
        raise ThresholdOptimizationError(
            "y_true must contain both classes."
        )

    if not np.isfinite(y_proba).all():
        raise ThresholdOptimizationError(
            "y_proba must contain only finite values."
        )

    if np.any(y_proba < 0.0) or np.any(y_proba > 1.0):
        raise ThresholdOptimizationError(
            "y_proba must contain values between 0 and 1."
        )

    if thresholds is None:
        thresholds_array = np.round(
            np.arange(0.05, 1.00, 0.01),
            decimals=2,
        )
    else:
        thresholds_array = np.asarray(
            thresholds,
            dtype=float,
        )

    if thresholds_array.ndim != 1:
        raise ThresholdOptimizationError(
            "thresholds must be one-dimensional."
        )

    if len(thresholds_array) == 0:
        raise ThresholdOptimizationError(
            "At least one threshold is required."
        )

    if not np.isfinite(thresholds_array).all():
        raise ThresholdOptimizationError(
            "thresholds must contain only finite values."
        )

    if np.any(thresholds_array < 0.0) or np.any(
        thresholds_array > 1.0
    ):
        raise ThresholdOptimizationError(
            "thresholds must be between 0 and 1."
        )

    thresholds_array = np.unique(thresholds_array)

    return y_true, y_proba, thresholds_array


def calculate_threshold_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    if not np.isfinite(threshold):
        raise ThresholdOptimizationError(
            "threshold must be finite."
        )

    if threshold < 0.0 or threshold > 1.0:
        raise ThresholdOptimizationError(
            "threshold must be between 0 and 1."
        )

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if len(y_true) != len(y_proba):
        raise ThresholdOptimizationError(
            "y_true and y_proba must have the same length."
        )

    y_pred = (y_proba >= threshold).astype(int)

    return ThresholdMetrics(
        threshold=float(threshold),
        precision=float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        predicted_positive_count=int(y_pred.sum()),
        actual_positive_count=int(y_true.sum()),
    )


def optimize_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    thresholds: list[float] | np.ndarray | None = None,
    objective: str = "f1",
) -> ThresholdOptimizationResult:
    if objective != "f1":
        raise ThresholdOptimizationError(
            "Only the 'f1' objective is supported in Phase 5.8."
        )

    y_true, y_proba, thresholds_array = _validate_inputs(
        y_true,
        y_proba,
        thresholds,
    )

    candidates = tuple(
        calculate_threshold_metrics(
            y_true,
            y_proba,
            float(threshold),
        )
        for threshold in thresholds_array
    )

    # Deterministic tie-breaking:
    # 1. highest F1
    # 2. highest recall
    # 3. highest precision
    # 4. lowest threshold
    selected = max(
        candidates,
        key=lambda metric: (
            metric.f1,
            metric.recall,
            metric.precision,
            -metric.threshold,
        ),
    )

    return ThresholdOptimizationResult(
        selected_threshold=selected.threshold,
        objective=objective,
        metrics=selected,
        candidates=candidates,
    )


def threshold_result_to_dict(
    result: ThresholdOptimizationResult,
) -> dict[str, object]:
    metrics = result.metrics

    return {
        "selected_threshold": result.selected_threshold,
        "objective": result.objective,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "predicted_positive_count": (
            metrics.predicted_positive_count
        ),
        "actual_positive_count": (
            metrics.actual_positive_count
        ),
    }


def threshold_candidates_to_dataframe(
    result: ThresholdOptimizationResult,
) -> pd.DataFrame:
    if not result.candidates:
        raise ThresholdOptimizationError(
            "Threshold optimization produced no candidates."
        )

    return pd.DataFrame(
        {
            "threshold": metric.threshold,
            "precision": metric.precision,
            "recall": metric.recall,
            "f1": metric.f1,
            "predicted_positive_count": (
                metric.predicted_positive_count
            ),
            "actual_positive_count": (
                metric.actual_positive_count
            ),
        }
        for metric in result.candidates
    )