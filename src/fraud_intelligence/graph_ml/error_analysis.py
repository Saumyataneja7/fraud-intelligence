from __future__ import annotations

from dataclasses import dataclass

from typing import Sequence

import pandas as pd
import torch


class GNNErrorAnalysisError(ValueError):
    """Raised when GNN error-analysis inputs are invalid."""


@dataclass(frozen=True)
class GNNErrorCounts:
    """Basic classification error counts."""

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    @property
    def total(self) -> int:
        return (
            self.true_negatives
            + self.false_positives
            + self.false_negatives
            + self.true_positives
        )


@dataclass(frozen=True)
class GNNErrorAnalysisResult:
    """Complete diagnostic error-analysis result."""

    split_name: str
    threshold: float
    counts: GNNErrorCounts
    error_frame: pd.DataFrame

    @property
    def false_positive_count(self) -> int:
        return self.counts.false_positives

    @property
    def false_negative_count(self) -> int:
        return self.counts.false_negatives

    @property
    def error_count(self) -> int:
        return (
            self.counts.false_positives
            + self.counts.false_negatives
        )


DEFAULT_THRESHOLD = 0.50


def build_gnn_error_frame(
    transaction_ids: Sequence[str],
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    fraud_scenarios: Sequence[str | None] | None = None,
    amounts: Sequence[float] | None = None,
) -> pd.DataFrame:
    """
    Build a transaction-level diagnostic frame.

    The frame contains:
        transaction_id
        actual_label
        predicted_probability
        predicted_label
        error_type
        fraud_scenario (optional)
        amount (optional)
    """

    _validate_prediction_inputs(
        transaction_ids=transaction_ids,
        labels=labels,
        probabilities=probabilities,
        threshold=threshold,
    )

    n = len(transaction_ids)

    if fraud_scenarios is not None:
        if len(fraud_scenarios) != n:
            raise GNNErrorAnalysisError(
                "fraud_scenarios length must match predictions."
            )

    if amounts is not None:
        if len(amounts) != n:
            raise GNNErrorAnalysisError(
                "amounts length must match predictions."
            )

    labels_cpu = labels.detach().cpu().numpy()
    probabilities_cpu = probabilities.detach().cpu().numpy()

    predictions = (
        probabilities_cpu >= threshold
    ).astype("int64")

    error_types = []

    for actual, predicted in zip(
        labels_cpu,
        predictions,
    ):
        if actual == 0 and predicted == 1:
            error_types.append("false_positive")
        elif actual == 1 and predicted == 0:
            error_types.append("false_negative")
        elif actual == 1 and predicted == 1:
            error_types.append("true_positive")
        else:
            error_types.append("true_negative")

    frame = pd.DataFrame(
        {
            "transaction_id": list(transaction_ids),
            "actual_label": labels_cpu.astype("int64"),
            "predicted_probability": probabilities_cpu.astype(
                "float64"
            ),
            "predicted_label": predictions,
            "error_type": error_types,
        }
    )

    if fraud_scenarios is not None:
        frame["fraud_scenario"] = list(
            fraud_scenarios
        )

    if amounts is not None:
        frame["amount"] = list(amounts)

    return frame


def calculate_gnn_error_counts(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> GNNErrorCounts:
    """Calculate confusion-matrix error counts."""

    _validate_label_probability_tensors(
        labels,
        probabilities,
    )
    _validate_threshold(threshold)

    predictions = (
        probabilities >= threshold
    ).to(torch.long)

    true_negatives = int(
        ((labels == 0) & (predictions == 0)).sum()
    )
    false_positives = int(
        ((labels == 0) & (predictions == 1)).sum()
    )
    false_negatives = int(
        ((labels == 1) & (predictions == 0)).sum()
    )
    true_positives = int(
        ((labels == 1) & (predictions == 1)).sum()
    )

    return GNNErrorCounts(
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_positives=true_positives,
    )


def analyze_gnn_errors(
    transaction_ids: Sequence[str],
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    split_name: str = "test",
    fraud_scenarios: Sequence[str | None] | None = None,
    amounts: Sequence[float] | None = None,
) -> GNNErrorAnalysisResult:
    """Build transaction-level errors and aggregate error counts."""

    frame = build_gnn_error_frame(
        transaction_ids=transaction_ids,
        labels=labels,
        probabilities=probabilities,
        threshold=threshold,
        fraud_scenarios=fraud_scenarios,
        amounts=amounts,
    )

    counts = calculate_gnn_error_counts(
        labels=labels,
        probabilities=probabilities,
        threshold=threshold,
    )

    return GNNErrorAnalysisResult(
        split_name=split_name,
        threshold=threshold,
        counts=counts,
        error_frame=frame,
    )


def summarize_gnn_errors_by_column(
    error_frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Summarize errors by a diagnostic column.

    Returns counts for:
        total
        false_positive
        false_negative
        true_positive
        true_negative
    """

    if not isinstance(error_frame, pd.DataFrame):
        raise GNNErrorAnalysisError(
            "error_frame must be a pandas DataFrame."
        )

    required_columns = {
        column,
        "error_type",
    }

    missing = required_columns.difference(
        error_frame.columns
    )

    if missing:
        raise GNNErrorAnalysisError(
            f"Missing required columns: {sorted(missing)}"
        )

    summary = (
        error_frame
        .groupby(
            [column, "error_type"],
            dropna=False,
        )
        .size()
        .unstack(
            fill_value=0,
        )
    )

    for error_type in (
        "false_positive",
        "false_negative",
        "true_positive",
        "true_negative",
    ):
        if error_type not in summary.columns:
            summary[error_type] = 0

    summary["total"] = summary[
        [
            "false_positive",
            "false_negative",
            "true_positive",
            "true_negative",
        ]
    ].sum(axis=1)

    summary = summary[
        [
            "total",
            "false_positive",
            "false_negative",
            "true_positive",
            "true_negative",
        ]
    ]

    return summary.sort_index()


def summarize_gnn_errors_by_amount_band(
    error_frame: pd.DataFrame,
    bins: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Summarize errors by transaction amount bands."""

    if "amount" not in error_frame.columns:
        raise GNNErrorAnalysisError(
            "error_frame must contain an 'amount' column."
        )

    if bins is None:
        bins = (
            0,
            25,
            50,
            100,
            250,
            500,
            1000,
            float("inf"),
        )

    if len(bins) < 2:
        raise GNNErrorAnalysisError(
            "At least two amount boundaries are required."
        )

    amount_bands = pd.cut(
        error_frame["amount"],
        bins=bins,
        right=False,
        include_lowest=True,
    )

    frame = error_frame.copy()
    frame["amount_band"] = amount_bands

    return summarize_gnn_errors_by_column(
        frame,
        "amount_band",
    )


def extract_false_positives(
    error_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return false-positive transactions."""

    _validate_error_frame(error_frame)

    return error_frame[
        error_frame["error_type"] == "false_positive"
    ].copy()


def extract_false_negatives(
    error_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return false-negative transactions."""

    _validate_error_frame(error_frame)

    return error_frame[
        error_frame["error_type"] == "false_negative"
    ].copy()


def _validate_prediction_inputs(
    transaction_ids: Sequence[str],
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    threshold: float,
) -> None:
    if len(transaction_ids) == 0:
        raise GNNErrorAnalysisError(
            "transaction_ids cannot be empty."
        )

    if len(set(transaction_ids)) != len(transaction_ids):
        raise GNNErrorAnalysisError(
            "transaction_ids must be unique."
        )

    if any(
        not isinstance(transaction_id, str)
        or not transaction_id
        for transaction_id in transaction_ids
    ):
        raise GNNErrorAnalysisError(
            "transaction_ids must contain non-empty strings."
        )

    _validate_label_probability_tensors(
        labels,
        probabilities,
    )
    _validate_threshold(threshold)

    if len(transaction_ids) != labels.shape[0]:
        raise GNNErrorAnalysisError(
            "transaction_ids length must match predictions."
        )


def _validate_label_probability_tensors(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
) -> None:
    if not isinstance(labels, torch.Tensor):
        raise GNNErrorAnalysisError(
            "labels must be a torch.Tensor."
        )

    if labels.ndim != 1:
        raise GNNErrorAnalysisError(
            "labels must be 1-dimensional."
        )

    if labels.dtype != torch.long:
        raise GNNErrorAnalysisError(
            "labels must use torch.long."
        )

    if not torch.all(
        (labels == 0) | (labels == 1)
    ):
        raise GNNErrorAnalysisError(
            "labels must contain only 0 and 1."
        )

    if not isinstance(probabilities, torch.Tensor):
        raise GNNErrorAnalysisError(
            "probabilities must be a torch.Tensor."
        )

    if probabilities.ndim != 1:
        raise GNNErrorAnalysisError(
            "probabilities must be 1-dimensional."
        )

    if not torch.is_floating_point(probabilities):
        raise GNNErrorAnalysisError(
            "probabilities must be floating-point."
        )

    if labels.shape[0] != probabilities.shape[0]:
        raise GNNErrorAnalysisError(
            "labels and probabilities must have equal length."
        )

    if labels.shape[0] == 0:
        raise GNNErrorAnalysisError(
            "labels and probabilities cannot be empty."
        )

    if not torch.isfinite(probabilities).all():
        raise GNNErrorAnalysisError(
            "probabilities must contain only finite values."
        )

    if torch.any(probabilities < 0) or torch.any(
        probabilities > 1
    ):
        raise GNNErrorAnalysisError(
            "probabilities must be in [0, 1]."
        )


def _validate_threshold(
    threshold: float,
) -> None:
    if not isinstance(
        threshold,
        (float, int),
    ) or isinstance(threshold, bool):
        raise GNNErrorAnalysisError(
            "threshold must be numeric."
        )

    if not 0 <= float(threshold) <= 1:
        raise GNNErrorAnalysisError(
            "threshold must be in [0, 1]."
        )


def _validate_error_frame(
    error_frame: pd.DataFrame,
) -> None:
    if not isinstance(error_frame, pd.DataFrame):
        raise GNNErrorAnalysisError(
            "error_frame must be a pandas DataFrame."
        )

    required_columns = {
        "transaction_id",
        "actual_label",
        "predicted_probability",
        "predicted_label",
        "error_type",
    }

    missing = required_columns.difference(
        error_frame.columns
    )

    if missing:
        raise GNNErrorAnalysisError(
            f"Missing required columns: {sorted(missing)}"
        )