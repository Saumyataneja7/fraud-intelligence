from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class ErrorAnalysisError(ValueError):
    """Raised when error-analysis inputs are invalid."""


@dataclass(frozen=True)
class ErrorCounts:
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

    @property
    def error_count(self) -> int:
        return self.false_positives + self.false_negatives

    @property
    def error_rate(self) -> float:
        if self.total == 0:
            return 0.0

        return self.error_count / self.total


@dataclass(frozen=True)
class ErrorAnalysisResult:
    model_name: str
    split_name: str
    threshold: float
    counts: ErrorCounts


def _validate_inputs(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    if y_true.ndim != 1:
        raise ErrorAnalysisError(
            "y_true must be one-dimensional."
        )

    if y_pred.ndim != 1:
        raise ErrorAnalysisError(
            "y_pred must be one-dimensional."
        )

    if y_proba.ndim != 1:
        raise ErrorAnalysisError(
            "y_proba must be one-dimensional."
        )

    if not (
        len(y_true)
        == len(y_pred)
        == len(y_proba)
    ):
        raise ErrorAnalysisError(
            "y_true, y_pred, and y_proba must have the same length."
        )

    if len(y_true) == 0:
        raise ErrorAnalysisError(
            "Error-analysis data cannot be empty."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ErrorAnalysisError(
            "y_true must contain only 0 and 1."
        )

    if not np.isin(y_pred, [0, 1]).all():
        raise ErrorAnalysisError(
            "y_pred must contain only 0 and 1."
        )

    if not np.isfinite(y_proba).all():
        raise ErrorAnalysisError(
            "y_proba must contain only finite values."
        )

    if np.any(y_proba < 0.0) or np.any(y_proba > 1.0):
        raise ErrorAnalysisError(
            "y_proba must contain values between 0 and 1."
        )

    return y_true, y_pred, y_proba


def calculate_error_counts(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> ErrorCounts:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ErrorAnalysisError(
            "y_true and y_pred must have the same length."
        )

    if len(y_true) == 0:
        raise ErrorAnalysisError(
            "Error-analysis data cannot be empty."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ErrorAnalysisError(
            "y_true must contain only 0 and 1."
        )

    if not np.isin(y_pred, [0, 1]).all():
        raise ErrorAnalysisError(
            "y_pred must contain only 0 and 1."
        )

    true_negative = int(
        ((y_true == 0) & (y_pred == 0)).sum()
    )

    false_positive = int(
        ((y_true == 0) & (y_pred == 1)).sum()
    )

    false_negative = int(
        ((y_true == 1) & (y_pred == 0)).sum()
    )

    true_positive = int(
        ((y_true == 1) & (y_pred == 1)).sum()
    )

    return ErrorCounts(
        true_negatives=true_negative,
        false_positives=false_positive,
        false_negatives=false_negative,
        true_positives=true_positive,
    )


def build_error_analysis(
    model_name: str,
    split_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> ErrorAnalysisResult:
    if not model_name.strip():
        raise ErrorAnalysisError(
            "model_name cannot be empty."
        )

    if not split_name.strip():
        raise ErrorAnalysisError(
            "split_name cannot be empty."
        )

    if not np.isfinite(threshold):
        raise ErrorAnalysisError(
            "threshold must be finite."
        )

    if threshold < 0.0 or threshold > 1.0:
        raise ErrorAnalysisError(
            "threshold must be between 0 and 1."
        )

    y_true, y_pred, y_proba = _validate_inputs(
        y_true,
        y_pred,
        y_proba,
    )

    expected_predictions = (
        y_proba >= threshold
    ).astype(int)

    if not np.array_equal(
        y_pred,
        expected_predictions,
    ):
        raise ErrorAnalysisError(
            "y_pred does not match y_proba at the supplied threshold."
        )

    counts = calculate_error_counts(
        y_true,
        y_pred,
    )

    return ErrorAnalysisResult(
        model_name=model_name,
        split_name=split_name,
        threshold=float(threshold),
        counts=counts,
    )


def build_error_frame(
    data: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise ErrorAnalysisError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise ErrorAnalysisError(
            "data cannot be empty."
        )

    if not np.isfinite(threshold):
        raise ErrorAnalysisError(
            "threshold must be finite."
        )

    if threshold < 0.0 or threshold > 1.0:
        raise ErrorAnalysisError(
            "threshold must be between 0 and 1."
        )

    y_true, _, y_proba = _validate_inputs(
        y_true,
        (y_proba >= threshold).astype(int),
        y_proba,
    )

    if len(data) != len(y_true):
        raise ErrorAnalysisError(
            "data and prediction arrays must have the same length."
        )

    result = data.copy()

    result["actual_label"] = y_true.astype(int)
    result["fraud_probability"] = y_proba
    result["predicted_label"] = (
        y_proba >= threshold
    ).astype(int)

    result["error_type"] = np.select(
        [
            (result["actual_label"] == 0)
            & (result["predicted_label"] == 1),
            (result["actual_label"] == 1)
            & (result["predicted_label"] == 0),
            (result["actual_label"] == 0)
            & (result["predicted_label"] == 0),
            (result["actual_label"] == 1)
            & (result["predicted_label"] == 1),
        ],
        [
            "false_positive",
            "false_negative",
            "true_negative",
            "true_positive",
        ],
        default="unknown",
    )

    return result


def extract_false_positives(
    error_frame: pd.DataFrame,
) -> pd.DataFrame:
    if "error_type" not in error_frame.columns:
        raise ErrorAnalysisError(
            "error_frame must contain error_type."
        )

    return error_frame[
        error_frame["error_type"] == "false_positive"
    ].copy()


def extract_false_negatives(
    error_frame: pd.DataFrame,
) -> pd.DataFrame:
    if "error_type" not in error_frame.columns:
        raise ErrorAnalysisError(
            "error_frame must contain error_type."
        )

    return error_frame[
        error_frame["error_type"] == "false_negative"
    ].copy()


def summarize_errors_by_column(
    error_frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    if column not in error_frame.columns:
        raise ErrorAnalysisError(
            f"Column does not exist: {column}"
        )

    summary = (
        error_frame.groupby(
            column,
            dropna=False,
        )
        .agg(
            total_transactions=(
                "actual_label",
                "size",
            ),
            actual_fraud=(
                "actual_label",
                "sum",
            ),
            predicted_fraud=(
                "predicted_label",
                "sum",
            ),
            false_positives=(
                "error_type",
                lambda values: (
                    values == "false_positive"
                ).sum(),
            ),
            false_negatives=(
                "error_type",
                lambda values: (
                    values == "false_negative"
                ).sum(),
            ),
        )
        .reset_index()
    )

    summary["actual_fraud_rate"] = (
        summary["actual_fraud"]
        / summary["total_transactions"]
    )

    summary["error_rate"] = (
        summary["false_positives"]
        + summary["false_negatives"]
    ) / summary["total_transactions"]

    return summary


def add_amount_band(
    error_frame: pd.DataFrame,
    amount_column: str = "amount",
) -> pd.DataFrame:
    if amount_column not in error_frame.columns:
        raise ErrorAnalysisError(
            f"Column does not exist: {amount_column}"
        )

    result = error_frame.copy()

    result["amount_band"] = pd.cut(
        result[amount_column],
        bins=[
            -np.inf,
            25,
            50,
            100,
            250,
            500,
            1000,
            np.inf,
        ],
        labels=[
            "<=25",
            "25-50",
            "50-100",
            "100-250",
            "250-500",
            "500-1000",
            ">1000",
        ],
        include_lowest=True,
    )

    return result