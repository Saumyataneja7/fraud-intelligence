from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_intelligence.ml.error_analysis import (
    ErrorAnalysisError,
    ErrorAnalysisResult,
    ErrorCounts,
    add_amount_band,
    build_error_analysis,
    build_error_frame,
    calculate_error_counts,
    extract_false_negatives,
    extract_false_positives,
    summarize_errors_by_column,
)


def make_data() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array(
        [0, 0, 0, 1, 1, 1]
    )

    y_proba = np.array(
        [0.1, 0.7, 0.2, 0.3, 0.8, 0.9]
    )

    return y_true, y_proba


def test_error_counts() -> None:
    y_true = np.array(
        [0, 0, 0, 1, 1, 1]
    )

    y_pred = np.array(
        [0, 1, 0, 0, 1, 1]
    )

    counts = calculate_error_counts(
        y_true,
        y_pred,
    )

    assert isinstance(counts, ErrorCounts)

    assert counts.true_negatives == 2
    assert counts.false_positives == 1
    assert counts.false_negatives == 1
    assert counts.true_positives == 2

    assert counts.total == 6
    assert counts.error_count == 2
    assert counts.error_rate == pytest.approx(
        2 / 6
    )


def test_build_error_analysis() -> None:
    y_true, y_proba = make_data()

    threshold = 0.5

    y_pred = (
        y_proba >= threshold
    ).astype(int)

    result = build_error_analysis(
        model_name="xgboost",
        split_name="validation",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        threshold=threshold,
    )

    assert isinstance(
        result,
        ErrorAnalysisResult,
    )

    assert result.model_name == "xgboost"
    assert result.split_name == "validation"
    assert result.threshold == 0.5


def test_build_error_frame() -> None:
    data = pd.DataFrame(
        {
            "transaction_id": [
                "T1",
                "T2",
                "T3",
                "T4",
            ],
            "amount": [
                10.0,
                50.0,
                100.0,
                200.0,
            ],
        }
    )

    y_true = np.array(
        [0, 1, 0, 1]
    )

    y_proba = np.array(
        [0.1, 0.8, 0.7, 0.2]
    )

    result = build_error_frame(
        data,
        y_true,
        y_proba,
        threshold=0.5,
    )

    assert len(result) == 4

    assert {
        "actual_label",
        "fraud_probability",
        "predicted_label",
        "error_type",
    }.issubset(result.columns)

    assert result.loc[0, "error_type"] == (
        "true_negative"
    )

    assert result.loc[1, "error_type"] == (
        "true_positive"
    )

    assert result.loc[2, "error_type"] == (
        "false_positive"
    )

    assert result.loc[3, "error_type"] == (
        "false_negative"
    )


def test_extract_false_positives() -> None:
    data = pd.DataFrame(
        {
            "actual_label": [0, 1, 0],
            "predicted_label": [1, 1, 0],
            "error_type": [
                "false_positive",
                "true_positive",
                "true_negative",
            ],
        }
    )

    false_positives = extract_false_positives(
        data
    )

    assert len(false_positives) == 1
    assert (
        false_positives.iloc[0]["error_type"]
        == "false_positive"
    )


def test_extract_false_negatives() -> None:
    data = pd.DataFrame(
        {
            "actual_label": [0, 1, 0],
            "predicted_label": [0, 0, 0],
            "error_type": [
                "true_negative",
                "false_negative",
                "true_negative",
            ],
        }
    )

    false_negatives = extract_false_negatives(
        data
    )

    assert len(false_negatives) == 1
    assert (
        false_negatives.iloc[0]["error_type"]
        == "false_negative"
    )


def test_summarize_errors_by_column() -> None:
    data = pd.DataFrame(
        {
            "transaction_type": [
                "purchase",
                "purchase",
                "transfer",
                "transfer",
            ],
            "actual_label": [0, 1, 0, 1],
            "predicted_label": [1, 1, 0, 0],
            "error_type": [
                "false_positive",
                "true_positive",
                "true_negative",
                "false_negative",
            ],
        }
    )

    summary = summarize_errors_by_column(
        data,
        "transaction_type",
    )

    assert len(summary) == 2

    assert {
        "transaction_type",
        "total_transactions",
        "actual_fraud",
        "predicted_fraud",
        "false_positives",
        "false_negatives",
        "actual_fraud_rate",
        "error_rate",
    }.issubset(summary.columns)


def test_add_amount_band() -> None:
    data = pd.DataFrame(
        {
            "amount": [
                10,
                30,
                75,
                150,
                300,
                750,
                1500,
            ]
        }
    )

    result = add_amount_band(data)

    assert "amount_band" in result.columns
    assert result["amount_band"].notna().all()


def test_empty_data_raises_error() -> None:
    with pytest.raises(
        ErrorAnalysisError,
        match="cannot be empty",
    ):
        calculate_error_counts(
            np.array([]),
            np.array([]),
        )


def test_length_mismatch_raises_error() -> None:
    with pytest.raises(
        ErrorAnalysisError,
        match="same length",
    ):
        calculate_error_counts(
            np.array([0, 1]),
            np.array([0]),
        )


def test_invalid_labels_raise_error() -> None:
    with pytest.raises(
        ErrorAnalysisError,
        match="only 0 and 1",
    ):
        calculate_error_counts(
            np.array([0, 2]),
            np.array([0, 1]),
        )


def test_invalid_predictions_raise_error() -> None:
    with pytest.raises(
        ErrorAnalysisError,
        match="only 0 and 1",
    ):
        calculate_error_counts(
            np.array([0, 1]),
            np.array([0, 2]),
        )


def test_invalid_probability_raises_error() -> None:
    with pytest.raises(
        ErrorAnalysisError,
        match="between 0 and 1",
    ):
        build_error_analysis(
            model_name="model",
            split_name="validation",
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            y_proba=np.array([0.2, 1.5]),
            threshold=0.5,
        )


def test_threshold_prediction_mismatch_raises_error() -> None:
    with pytest.raises(
        ErrorAnalysisError,
        match="does not match",
    ):
        build_error_analysis(
            model_name="model",
            split_name="validation",
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 0]),
            y_proba=np.array([0.2, 0.8]),
            threshold=0.5,
        )


def test_invalid_threshold_raises_error() -> None:
    with pytest.raises(
        ErrorAnalysisError,
        match="between 0 and 1",
    ):
        build_error_analysis(
            model_name="model",
            split_name="validation",
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            y_proba=np.array([0.2, 0.8]),
            threshold=1.5,
        )


def test_missing_summary_column_raises_error() -> None:
    data = pd.DataFrame(
        {
            "actual_label": [0, 1],
            "predicted_label": [0, 1],
            "error_type": [
                "true_negative",
                "true_positive",
            ],
        }
    )

    with pytest.raises(
        ErrorAnalysisError,
        match="Column does not exist",
    ):
        summarize_errors_by_column(
            data,
            "transaction_type",
        )


def test_missing_amount_column_raises_error() -> None:
    data = pd.DataFrame(
        {
            "value": [10, 20]
        }
    )

    with pytest.raises(
        ErrorAnalysisError,
        match="Column does not exist",
    ):
        add_amount_band(data)