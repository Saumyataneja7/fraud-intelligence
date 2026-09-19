from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_intelligence.graph_ml.training import (
    TransactionGraphSAGEClassifier,
)


class GraphEvaluationError(ValueError):
    """Raised when GNN evaluation inputs are invalid."""


@dataclass(frozen=True)
class GNNEvaluationResult:
    """Classification metrics for one graph split."""

    split_name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    support: int
    predicted_positive: int
    actual_positive: int

    @property
    def predicted_negative(self) -> int:
        return self.support - self.predicted_positive

    @property
    def actual_negative(self) -> int:
        return self.support - self.actual_positive

    @property
    def confusion_matrix(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (
                self.true_negative,
                self.false_positive,
            ),
            (
                self.false_negative,
                self.true_positive,
            ),
        )


@dataclass(frozen=True)
class GNNPredictionResult:
    """Predictions produced for a graph split."""

    split_name: str
    node_indices: tuple[int, ...]
    probabilities: tuple[float, ...]
    predictions: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.node_indices)

    @property
    def predicted_positive(self) -> int:
        return sum(self.predictions)


DEFAULT_THRESHOLD = 0.50


@torch.no_grad()
def predict_graph(
    model: TransactionGraphSAGEClassifier,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    mask: torch.Tensor,
    split_name: str,
) -> GNNPredictionResult:
    """
    Generate fraud probabilities and binary predictions for one split.

    The model is evaluated in inference mode and no parameters are updated.
    """

    _validate_prediction_inputs(
        model=model,
        x=x,
        edge_index=edge_index,
        mask=mask,
        split_name=split_name,
    )

    model.eval()

    logits = model(
        x,
        edge_index,
    )

    probabilities = torch.sigmoid(
        logits[mask]
    )

    predictions = (
        probabilities >= DEFAULT_THRESHOLD
    ).long()

    node_indices = (
        torch.nonzero(
            mask,
            as_tuple=False,
        )
        .view(-1)
        .cpu()
        .tolist()
    )

    return GNNPredictionResult(
        split_name=split_name,
        node_indices=tuple(node_indices),
        probabilities=tuple(
            probabilities.cpu().tolist()
        ),
        predictions=tuple(
            predictions.cpu().tolist()
        ),
    )


def calculate_gnn_metrics(
    labels: torch.Tensor,
    prediction_result: GNNPredictionResult,
    threshold: float = DEFAULT_THRESHOLD,
) -> GNNEvaluationResult:
    """
    Calculate classification metrics for one graph split.

    Threshold optimization is deliberately not performed here.
    """

    _validate_labels(labels)

    if not 0.0 < threshold < 1.0:
        raise GraphEvaluationError(
            "threshold must be strictly between 0 and 1."
        )

    indices = list(
        prediction_result.node_indices
    )

    if len(indices) == 0:
        raise GraphEvaluationError(
            "Prediction result contains no nodes."
        )

    if any(
        index < 0 or index >= labels.shape[0]
        for index in indices
    ):
        raise GraphEvaluationError(
            "Prediction contains an out-of-range node index."
        )

    y_true = labels[
        indices
    ].cpu().numpy()

    probabilities = np.asarray(
        prediction_result.probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    if len(y_true) != len(probabilities):
        raise GraphEvaluationError(
            "Labels and predictions must have equal length."
        )

    if not np.isfinite(probabilities).all():
        raise GraphEvaluationError(
            "Prediction probabilities must be finite."
        )

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise GraphEvaluationError(
            "Prediction probabilities must be in [0, 1]."
        )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

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

    actual_classes = np.unique(y_true)

    if len(actual_classes) < 2:
        raise GraphEvaluationError(
            "ROC-AUC and PR-AUC require both classes "
            "to be present in the evaluated split."
        )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    return GNNEvaluationResult(
        split_name=prediction_result.split_name,
        threshold=threshold,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        pr_auc=float(pr_auc),
        roc_auc=float(roc_auc),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
        support=len(y_true),
        predicted_positive=int(predictions.sum()),
        actual_positive=int(y_true.sum()),
    )


def evaluate_graph_split(
    model: TransactionGraphSAGEClassifier,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    split_name: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[GNNPredictionResult, GNNEvaluationResult]:
    """
    Generate predictions and calculate metrics for one graph split.
    """

    if not 0.0 < threshold < 1.0:
        raise GraphEvaluationError(
            "threshold must be strictly between 0 and 1."
        )

    prediction_result = predict_graph(
        model=model,
        x=x,
        edge_index=edge_index,
        mask=mask,
        split_name=split_name,
    )

    metrics = calculate_gnn_metrics(
        labels=labels,
        prediction_result=prediction_result,
        threshold=threshold,
    )

    return prediction_result, metrics


def _validate_prediction_inputs(
    model: TransactionGraphSAGEClassifier,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    mask: torch.Tensor,
    split_name: str,
) -> None:
    if not isinstance(
        model,
        TransactionGraphSAGEClassifier,
    ):
        raise GraphEvaluationError(
            "model must be TransactionGraphSAGEClassifier."
        )

    if not isinstance(x, torch.Tensor):
        raise GraphEvaluationError(
            "x must be a torch.Tensor."
        )

    if x.ndim != 2:
        raise GraphEvaluationError(
            "x must be 2-dimensional."
        )

    if x.shape[0] == 0:
        raise GraphEvaluationError(
            "x must contain at least one node."
        )

    if not torch.is_floating_point(x):
        raise GraphEvaluationError(
            "x must contain floating-point features."
        )

    if not torch.isfinite(x).all():
        raise GraphEvaluationError(
            "x must contain only finite values."
        )

    if not isinstance(edge_index, torch.Tensor):
        raise GraphEvaluationError(
            "edge_index must be a torch.Tensor."
        )

    if edge_index.ndim != 2:
        raise GraphEvaluationError(
            "edge_index must be 2-dimensional."
        )

    if edge_index.shape[0] != 2:
        raise GraphEvaluationError(
            "edge_index must have shape [2, num_edges]."
        )

    if edge_index.dtype != torch.long:
        raise GraphEvaluationError(
            "edge_index must use torch.long."
        )

    if edge_index.numel() > 0:
        if int(edge_index.min()) < 0:
            raise GraphEvaluationError(
                "edge_index contains a negative node index."
            )

        if int(edge_index.max()) >= x.shape[0]:
            raise GraphEvaluationError(
                "edge_index contains an out-of-range node index."
            )

    if not isinstance(mask, torch.Tensor):
        raise GraphEvaluationError(
            "mask must be a torch.Tensor."
        )

    if mask.ndim != 1:
        raise GraphEvaluationError(
            "mask must be 1-dimensional."
        )

    if mask.dtype != torch.bool:
        raise GraphEvaluationError(
            "mask must use torch.bool."
        )

    if mask.shape[0] != x.shape[0]:
        raise GraphEvaluationError(
            "mask must contain one value per node."
        )

    if not mask.any():
        raise GraphEvaluationError(
            f"{split_name} mask contains no nodes."
        )

    if not isinstance(split_name, str) or not split_name:
        raise GraphEvaluationError(
            "split_name must be a non-empty string."
        )


def _validate_labels(
    labels: torch.Tensor,
) -> None:
    if not isinstance(labels, torch.Tensor):
        raise GraphEvaluationError(
            "labels must be a torch.Tensor."
        )

    if labels.ndim != 1:
        raise GraphEvaluationError(
            "labels must be 1-dimensional."
        )

    if labels.dtype != torch.long:
        raise GraphEvaluationError(
            "labels must use torch.long."
        )

    unique_labels = torch.unique(labels)

    if not torch.all(
        (unique_labels == 0)
        | (unique_labels == 1)
    ):
        raise GraphEvaluationError(
            "labels must contain only 0 and 1."
        )