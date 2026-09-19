from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


DEFAULT_MIN_THRESHOLD = 0.05
DEFAULT_MAX_THRESHOLD = 0.99
DEFAULT_THRESHOLD_STEP = 0.01


class GNNThresholdError(ValueError):
    """Raised when threshold optimization inputs are invalid."""


@dataclass(frozen=True)
class GNNThresholdMetrics:
    """Classification metrics at one candidate threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    predicted_positive: int
    actual_positive: int
    support: int


@dataclass(frozen=True)
class GNNThresholdOptimizationResult:
    """Result of validation-only threshold optimization."""

    split_name: str
    objective: str
    selected_threshold: float
    selected_metrics: GNNThresholdMetrics
    candidates: tuple[GNNThresholdMetrics, ...]

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)


def generate_threshold_grid(
    minimum: float = DEFAULT_MIN_THRESHOLD,
    maximum: float = DEFAULT_MAX_THRESHOLD,
    step: float = DEFAULT_THRESHOLD_STEP,
) -> tuple[float, ...]:
    """Generate a deterministic inclusive threshold grid."""

    if not 0.0 < minimum < 1.0:
        raise GNNThresholdError(
            "minimum must be strictly between 0 and 1."
        )

    if not 0.0 < maximum < 1.0:
        raise GNNThresholdError(
            "maximum must be strictly between 0 and 1."
        )

    if minimum > maximum:
        raise GNNThresholdError(
            "minimum cannot exceed maximum."
        )

    if step <= 0:
        raise GNNThresholdError(
            "step must be greater than zero."
        )

    values: list[float] = []

    current = minimum

    while current <= maximum + step * 1e-9:
        values.append(
            round(current, 10)
        )
        current += step

    return tuple(values)


def calculate_threshold_metrics(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    threshold: float,
) -> GNNThresholdMetrics:
    """Calculate binary classification metrics at one threshold."""

    _validate_inputs(
        labels=labels,
        probabilities=probabilities,
        threshold=threshold,
    )

    y_true = labels.cpu().numpy()

    probabilities_np = probabilities.cpu().numpy()

    predictions = (
        probabilities_np >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    return GNNThresholdMetrics(
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        predicted_positive=int(
            predictions.sum()
        ),
        actual_positive=int(
            y_true.sum()
        ),
        support=len(y_true),
    )


def optimize_gnn_threshold(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    split_name: str = "validation",
    minimum: float = DEFAULT_MIN_THRESHOLD,
    maximum: float = DEFAULT_MAX_THRESHOLD,
    step: float = DEFAULT_THRESHOLD_STEP,
) -> GNNThresholdOptimizationResult:
    """
    Optimize the classification threshold using validation data.

    Objective:
        1. highest F1
        2. higher recall
        3. higher precision
        4. lower threshold
    """

    if not isinstance(split_name, str) or not split_name:
        raise GNNThresholdError(
            "split_name must be a non-empty string."
        )

    thresholds = generate_threshold_grid(
        minimum=minimum,
        maximum=maximum,
        step=step,
    )

    candidates = tuple(
        calculate_threshold_metrics(
            labels=labels,
            probabilities=probabilities,
            threshold=threshold,
        )
        for threshold in thresholds
    )

    selected = min(
        candidates,
        key=lambda candidate: (
            -candidate.f1,
            -candidate.recall,
            -candidate.precision,
            candidate.threshold,
        ),
    )

    return GNNThresholdOptimizationResult(
        split_name=split_name,
        objective="f1",
        selected_threshold=selected.threshold,
        selected_metrics=selected,
        candidates=candidates,
    )


def _validate_inputs(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    threshold: float,
) -> None:
    if not isinstance(labels, torch.Tensor):
        raise GNNThresholdError(
            "labels must be a torch.Tensor."
        )

    if labels.ndim != 1:
        raise GNNThresholdError(
            "labels must be 1-dimensional."
        )

    if labels.dtype != torch.long:
        raise GNNThresholdError(
            "labels must use torch.long."
        )

    unique_labels = torch.unique(labels)

    if not torch.all(
        (unique_labels == 0)
        | (unique_labels == 1)
    ):
        raise GNNThresholdError(
            "labels must contain only 0 and 1."
        )

    if not isinstance(probabilities, torch.Tensor):
        raise GNNThresholdError(
            "probabilities must be a torch.Tensor."
        )

    if probabilities.ndim != 1:
        raise GNNThresholdError(
            "probabilities must be 1-dimensional."
        )

    if not torch.is_floating_point(probabilities):
        raise GNNThresholdError(
            "probabilities must be floating-point."
        )

    if labels.shape[0] != probabilities.shape[0]:
        raise GNNThresholdError(
            "labels and probabilities must have equal length."
        )

    if probabilities.numel() == 0:
        raise GNNThresholdError(
            "labels and probabilities cannot be empty."
        )

    if not torch.isfinite(probabilities).all():
        raise GNNThresholdError(
            "probabilities must contain only finite values."
        )

    if torch.any(probabilities < 0) or torch.any(
        probabilities > 1
    ):
        raise GNNThresholdError(
            "probabilities must be in the range [0, 1]."
        )

    if not 0.0 < threshold < 1.0:
        raise GNNThresholdError(
            "threshold must be strictly between 0 and 1."
        )