from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_intelligence.ml.evaluation import (
    ClassificationMetrics,
    EvaluationResult,
    ModelEvaluationError,
    calculate_classification_metrics,
    evaluate_model,
    evaluation_result_to_dict,
    evaluation_results_to_dataframe,
)


def test_calculate_classification_metrics() -> None:
    y_true = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1]
    )

    y_pred = np.array(
        [0, 0, 1, 1, 0, 0, 1, 1]
    )

    y_proba = np.array(
        [0.1, 0.2, 0.7, 0.8, 0.3, 0.4, 0.9, 0.95]
    )

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        y_proba,
    )

    assert isinstance(
        metrics,
        ClassificationMetrics,
    )

    assert metrics.true_negatives == 2
    assert metrics.false_positives == 2
    assert metrics.false_negatives == 2
    assert metrics.true_positives == 2

    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)

    assert 0.0 <= metrics.pr_auc <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_support_is_correct() -> None:
    metrics = ClassificationMetrics(
        precision=0.5,
        recall=0.5,
        f1=0.5,
        pr_auc=0.7,
        roc_auc=0.8,
        true_negatives=10,
        false_positives=2,
        false_negatives=3,
        true_positives=5,
    )

    assert metrics.support == 20


def test_predicted_positive_count() -> None:
    metrics = ClassificationMetrics(
        precision=0.5,
        recall=0.5,
        f1=0.5,
        pr_auc=0.7,
        roc_auc=0.8,
        true_negatives=10,
        false_positives=2,
        false_negatives=3,
        true_positives=5,
    )

    assert metrics.predicted_positive_count == 7


def test_actual_positive_count() -> None:
    metrics = ClassificationMetrics(
        precision=0.5,
        recall=0.5,
        f1=0.5,
        pr_auc=0.7,
        roc_auc=0.8,
        true_negatives=10,
        false_positives=2,
        false_negatives=3,
        true_positives=5,
    )

    assert metrics.actual_positive_count == 8


def test_evaluate_model_returns_result() -> None:
    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_pred = np.array(
        [0, 1, 0, 1]
    )

    y_proba = np.array(
        [0.1, 0.7, 0.3, 0.9]
    )

    result = evaluate_model(
        model_name="logistic_regression",
        split_name="validation",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    assert isinstance(
        result,
        EvaluationResult,
    )

    assert result.model_name == "logistic_regression"
    assert result.split_name == "validation"


def test_result_to_dict() -> None:
    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_pred = np.array(
        [0, 1, 0, 1]
    )

    y_proba = np.array(
        [0.1, 0.7, 0.3, 0.9]
    )

    result = evaluate_model(
        model_name="random_forest",
        split_name="test",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    data = evaluation_result_to_dict(result)

    assert data["model_name"] == "random_forest"
    assert data["split_name"] == "test"
    assert "precision" in data
    assert "recall" in data
    assert "f1" in data
    assert "pr_auc" in data
    assert "roc_auc" in data
    assert "true_positives" in data
    assert "false_positives" in data
    assert "true_negatives" in data
    assert "false_negatives" in data


def test_results_to_dataframe() -> None:
    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_pred = np.array(
        [0, 1, 0, 1]
    )

    y_proba = np.array(
        [0.1, 0.7, 0.3, 0.9]
    )

    results = [
        evaluate_model(
            model_name="logistic_regression",
            split_name="validation",
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        ),
        evaluate_model(
            model_name="random_forest",
            split_name="validation",
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        ),
    ]

    dataframe = evaluation_results_to_dataframe(
        results
    )

    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 2
    assert set(dataframe["model_name"]) == {
        "logistic_regression",
        "random_forest",
    }


def test_empty_evaluation_data_raises_error() -> None:
    with pytest.raises(
        ModelEvaluationError,
        match="Evaluation data cannot be empty",
    ):
        calculate_classification_metrics(
            np.array([]),
            np.array([]),
            np.array([]),
        )


def test_length_mismatch_raises_error() -> None:
    with pytest.raises(
        ModelEvaluationError,
        match="same length",
    ):
        calculate_classification_metrics(
            np.array([0, 1]),
            np.array([0]),
            np.array([0.1, 0.9]),
        )


def test_invalid_true_labels_raise_error() -> None:
    with pytest.raises(
        ModelEvaluationError,
        match="y_true must contain only 0 and 1",
    ):
        calculate_classification_metrics(
            np.array([0, 1, 2]),
            np.array([0, 1, 1]),
            np.array([0.1, 0.8, 0.9]),
        )


def test_invalid_predictions_raise_error() -> None:
    with pytest.raises(
        ModelEvaluationError,
        match="y_pred must contain only 0 and 1",
    ):
        calculate_classification_metrics(
            np.array([0, 1, 1]),
            np.array([0, 1, 2]),
            np.array([0.1, 0.8, 0.9]),
        )


def test_probability_out_of_range_raises_error() -> None:
    with pytest.raises(
        ModelEvaluationError,
        match="between 0 and 1",
    ):
        calculate_classification_metrics(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0.1, 1.2]),
        )


def test_nan_probability_raises_error() -> None:
    with pytest.raises(
        ModelEvaluationError,
        match="NaN",
    ):
        calculate_classification_metrics(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0.1, np.nan]),
        )


def test_single_class_true_labels_raise_error() -> None:
    with pytest.raises(
        ModelEvaluationError,
        match="both classes",
    ):
        calculate_classification_metrics(
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0.1, 0.2, 0.3]),
        )


def test_empty_model_name_raises_error() -> None:
    with pytest.raises(
        ModelEvaluationError,
        match="model_name cannot be empty",
    ):
        evaluate_model(
            model_name="",
            split_name="test",
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            y_proba=np.array([0.1, 0.9]),
        )


def test_empty_split_name_raises_error() -> None:
    with pytest.raises(
        ModelEvaluationError,
        match="split_name cannot be empty",
    ):
        evaluate_model(
            model_name="xgboost",
            split_name="",
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            y_proba=np.array([0.1, 0.9]),
        )


def test_empty_results_raise_error() -> None:
    with pytest.raises(
        ModelEvaluationError,
        match="At least one evaluation result",
    ):
        evaluation_results_to_dataframe([])