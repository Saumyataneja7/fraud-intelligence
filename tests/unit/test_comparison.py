from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_intelligence.ml.comparison import (
    ModelComparisonError,
    ModelComparisonResult,
    build_model_comparison,
    build_ranking_comparison,
    comparison_to_dataframe,
    validate_comparison_schema,
)
from fraud_intelligence.ml.evaluation import (
    evaluate_model,
)
from fraud_intelligence.ml.ranking import (
    calculate_ranking_metrics,
)


def make_evaluation_result(
    model_name: str,
):
    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_pred = np.array(
        [0, 1, 0, 1]
    )

    y_proba = np.array(
        [0.1, 0.7, 0.3, 0.9]
    )

    return evaluate_model(
        model_name=model_name,
        split_name="validation",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )


def test_build_model_comparison() -> None:
    results = [
        make_evaluation_result(
            "logistic_regression"
        ),
        make_evaluation_result(
            "random_forest"
        ),
        make_evaluation_result(
            "xgboost"
        ),
    ]

    comparison = build_model_comparison(
        results
    )

    assert isinstance(
        comparison,
        ModelComparisonResult,
    )

    assert len(comparison.rows) == 3

    assert {
        row.model_name
        for row in comparison.rows
    } == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }


def test_comparison_to_dataframe() -> None:
    results = [
        make_evaluation_result(
            "logistic_regression"
        ),
        make_evaluation_result(
            "xgboost"
        ),
    ]

    comparison = build_model_comparison(
        results
    )

    dataframe = comparison_to_dataframe(
        comparison
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )

    assert len(dataframe) == 2

    assert {
        "model_name",
        "split_name",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    }.issubset(dataframe.columns)


def test_comparison_schema_is_valid() -> None:
    result = make_evaluation_result(
        "xgboost"
    )

    comparison = build_model_comparison(
        [result]
    )

    dataframe = comparison_to_dataframe(
        comparison
    )

    validate_comparison_schema(
        dataframe
    )


def test_ranking_comparison() -> None:
    y_true = np.array(
        [0, 0, 0, 1, 1, 1]
    )

    y_proba = np.array(
        [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    )

    metrics = calculate_ranking_metrics(
        y_true,
        y_proba,
        k_values=[2, 4],
    )

    dataframe = build_ranking_comparison(
        {
            "logistic_regression": metrics,
            "xgboost": metrics,
        }
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )

    assert len(dataframe) == 4

    assert set(
        dataframe["model_name"]
    ) == {
        "logistic_regression",
        "xgboost",
    }


def test_empty_results_raise_error() -> None:
    with pytest.raises(
        ModelComparisonError,
        match="At least one evaluation result",
    ):
        build_model_comparison([])


def test_invalid_evaluation_result_raises_error() -> None:
    with pytest.raises(
        ModelComparisonError,
        match="EvaluationResult",
    ):
        build_model_comparison(
            ["invalid"]
        )


def test_invalid_comparison_type_raises_error() -> None:
    with pytest.raises(
        ModelComparisonError,
        match="ModelComparisonResult",
    ):
        comparison_to_dataframe(
            "invalid"
        )


def test_empty_comparison_rows_raise_error() -> None:
    comparison = ModelComparisonResult(
        rows=()
    )

    with pytest.raises(
        ModelComparisonError,
        match="no rows",
    ):
        comparison_to_dataframe(
            comparison
        )


def test_empty_ranking_results_raise_error() -> None:
    with pytest.raises(
        ModelComparisonError,
        match="At least one model",
    ):
        build_ranking_comparison({})


def test_empty_model_name_raises_error() -> None:
    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_proba = np.array(
        [0.1, 0.2, 0.8, 0.9]
    )

    metrics = calculate_ranking_metrics(
        y_true,
        y_proba,
        k_values=[2],
    )

    with pytest.raises(
        ModelComparisonError,
        match="Model name cannot be empty",
    ):
        build_ranking_comparison(
            {
                "": metrics
            }
        )


def test_empty_ranking_metrics_raise_error() -> None:
    with pytest.raises(
        ModelComparisonError,
        match="No ranking metrics",
    ):
        build_ranking_comparison(
            {
                "xgboost": []
            }
        )


def test_invalid_ranking_metric_raises_error() -> None:
    with pytest.raises(
        ModelComparisonError,
        match="RankingMetrics",
    ):
        build_ranking_comparison(
            {
                "xgboost": ["invalid"]
            }
        )


def test_missing_comparison_columns_raise_error() -> None:
    dataframe = pd.DataFrame(
        {
            "model_name": [
                "xgboost"
            ]
        }
    )

    with pytest.raises(
        ModelComparisonError,
        match="missing required columns",
    ):
        validate_comparison_schema(
            dataframe
        )