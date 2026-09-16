from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluationError(ValueError):
    """Raised when model evaluation inputs are invalid."""


@dataclass(frozen=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    @property
    def support(self) -> int:
        return (
            self.true_negatives
            + self.false_positives
            + self.false_negatives
            + self.true_positives
        )

    @property
    def predicted_positive_count(self) -> int:
        return (
            self.false_positives
            + self.true_positives
        )

    @property
    def actual_positive_count(self) -> int:
        return (
            self.false_negatives
            + self.true_positives
        )


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    split_name: str
    metrics: ClassificationMetrics


def _validate_inputs(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    if y_true.ndim != 1:
        raise ModelEvaluationError(
            "y_true must be one-dimensional."
        )

    if y_pred.ndim != 1:
        raise ModelEvaluationError(
            "y_pred must be one-dimensional."
        )

    if y_proba.ndim != 1:
        raise ModelEvaluationError(
            "y_proba must be one-dimensional."
        )

    if not (
        len(y_true)
        == len(y_pred)
        == len(y_proba)
    ):
        raise ModelEvaluationError(
            "y_true, y_pred, and y_proba must have the same length."
        )

    if len(y_true) == 0:
        raise ModelEvaluationError(
            "Evaluation data cannot be empty."
        )

    if np.isnan(y_proba).any():
        raise ModelEvaluationError(
            "y_proba contains NaN values."
        )

    if not np.isfinite(y_proba).all():
        raise ModelEvaluationError(
            "y_proba contains non-finite values."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ModelEvaluationError(
            "y_true must contain only 0 and 1."
        )

    if not np.isin(y_pred, [0, 1]).all():
        raise ModelEvaluationError(
            "y_pred must contain only 0 and 1."
        )

    if not np.isfinite(y_proba).all():
        raise ModelEvaluationError(
            "y_proba contains non-finite values."
        )

    if np.any(y_proba < 0.0) or np.any(y_proba > 1.0):
        raise ModelEvaluationError(
            "y_proba must contain values between 0 and 1."
        )

    if np.unique(y_true).size < 2:
        raise ModelEvaluationError(
            "y_true must contain both classes."
        )

    return y_true, y_pred, y_proba


def calculate_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
) -> ClassificationMetrics:
    y_true, y_pred, y_proba = _validate_inputs(
        y_true,
        y_pred,
        y_proba,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return ClassificationMetrics(
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
        pr_auc=float(
            average_precision_score(
                y_true,
                y_proba,
            )
        ),
        roc_auc=float(
            roc_auc_score(
                y_true,
                y_proba,
            )
        ),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )


def evaluate_model(
    model_name: str,
    split_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
) -> EvaluationResult:
    if not model_name.strip():
        raise ModelEvaluationError(
            "model_name cannot be empty."
        )

    if not split_name.strip():
        raise ModelEvaluationError(
            "split_name cannot be empty."
        )

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        y_proba,
    )

    return EvaluationResult(
        model_name=model_name,
        split_name=split_name,
        metrics=metrics,
    )


def evaluation_result_to_dict(
    result: EvaluationResult,
) -> dict[str, object]:
    metrics = result.metrics

    return {
        "model_name": result.model_name,
        "split_name": result.split_name,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "pr_auc": metrics.pr_auc,
        "roc_auc": metrics.roc_auc,
        "true_negatives": metrics.true_negatives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "true_positives": metrics.true_positives,
        "support": metrics.support,
        "predicted_positive_count": (
            metrics.predicted_positive_count
        ),
        "actual_positive_count": (
            metrics.actual_positive_count
        ),
    }


def evaluation_results_to_dataframe(
    results: list[EvaluationResult],
) -> pd.DataFrame:
    if not results:
        raise ModelEvaluationError(
            "At least one evaluation result is required."
        )

    return pd.DataFrame(
        evaluation_result_to_dict(result)
        for result in results
    )