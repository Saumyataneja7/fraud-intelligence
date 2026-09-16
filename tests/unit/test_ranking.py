from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_intelligence.ml.ranking import (
    RankingEvaluationError,
    RankingMetrics,
    calculate_precision_recall_at_k,
    calculate_ranking_metrics,
    ranking_metrics_to_dataframe,
)


def make_data() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1]
    )

    y_proba = np.array(
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80, 0.90]
    )

    return y_true, y_proba


def test_precision_recall_at_k() -> None:
    y_true, y_proba = make_data()

    metrics = calculate_precision_recall_at_k(
        y_true,
        y_proba,
        k=4,
    )

    assert isinstance(
        metrics,
        RankingMetrics,
    )

    assert metrics.k == 4

    # Top four probabilities correspond to:
    # 1, 1, 1, 1
    assert metrics.true_positives_at_k == 4

    assert metrics.actual_positive_count == 4

    assert metrics.precision_at_k == pytest.approx(1.0)
    assert metrics.recall_at_k == pytest.approx(1.0)


def test_precision_at_k_when_top_k_contains_false_positives() -> None:
    y_true = np.array(
        [0, 1, 0, 1, 0, 1]
    )

    y_proba = np.array(
        [0.95, 0.90, 0.85, 0.80, 0.10, 0.05]
    )

    metrics = calculate_precision_recall_at_k(
        y_true,
        y_proba,
        k=3,
    )

    assert metrics.true_positives_at_k == 1

    assert metrics.precision_at_k == pytest.approx(
        1 / 3
    )

    assert metrics.recall_at_k == pytest.approx(
        1 / 3
    )


def test_k_one() -> None:
    y_true = np.array([0, 1, 1])
    y_proba = np.array([0.2, 0.9, 0.8])

    metrics = calculate_precision_recall_at_k(
        y_true,
        y_proba,
        k=1,
    )

    assert metrics.true_positives_at_k == 1
    assert metrics.precision_at_k == 1.0
    assert metrics.recall_at_k == pytest.approx(0.5)


def test_k_equal_to_dataset_size() -> None:
    y_true, y_proba = make_data()

    metrics = calculate_precision_recall_at_k(
        y_true,
        y_proba,
        k=len(y_true),
    )

    assert metrics.true_positives_at_k == int(
        y_true.sum()
    )

    assert metrics.precision_at_k == pytest.approx(
        y_true.mean()
    )

    assert metrics.recall_at_k == pytest.approx(1.0)


def test_multiple_k_values() -> None:
    y_true, y_proba = make_data()

    metrics = calculate_ranking_metrics(
        y_true,
        y_proba,
        k_values=[2, 4, 6],
    )

    assert len(metrics) == 3

    assert [metric.k for metric in metrics] == [
        2,
        4,
        6,
    ]


def test_duplicate_k_values_are_removed() -> None:
    y_true, y_proba = make_data()

    metrics = calculate_ranking_metrics(
        y_true,
        y_proba,
        k_values=[2, 2, 4, 4],
    )

    assert [metric.k for metric in metrics] == [
        2,
        4,
    ]


def test_ranking_is_deterministic_for_tied_probabilities() -> None:
    y_true = np.array(
        [1, 0, 1, 0]
    )

    y_proba = np.array(
        [0.8, 0.8, 0.5, 0.5]
    )

    metrics = calculate_precision_recall_at_k(
        y_true,
        y_proba,
        k=2,
    )

    # Stable sorting preserves the original order
    # for the tied 0.8 probabilities.
    assert metrics.true_positives_at_k == 1


def test_dataframe_conversion() -> None:
    y_true, y_proba = make_data()

    metrics = calculate_ranking_metrics(
        y_true,
        y_proba,
        k_values=[2, 4],
    )

    dataframe = ranking_metrics_to_dataframe(
        metrics
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )

    assert len(dataframe) == 2

    assert set(
        [
            "k",
            "precision_at_k",
            "recall_at_k",
            "true_positives_at_k",
            "actual_positive_count",
            "reviewed_count",
        ]
    ).issubset(dataframe.columns)


def test_empty_data_raises_error() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="cannot be empty",
    ):
        calculate_precision_recall_at_k(
            np.array([]),
            np.array([]),
            k=1,
        )


def test_length_mismatch_raises_error() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="same length",
    ):
        calculate_precision_recall_at_k(
            np.array([0, 1]),
            np.array([0.8]),
            k=1,
        )


def test_invalid_labels_raise_error() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="only 0 and 1",
    ):
        calculate_precision_recall_at_k(
            np.array([0, 1, 2]),
            np.array([0.1, 0.8, 0.9]),
            k=2,
        )


def test_single_class_raises_error() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="both classes",
    ):
        calculate_precision_recall_at_k(
            np.array([0, 0, 0]),
            np.array([0.1, 0.2, 0.3]),
            k=2,
        )


def test_probability_out_of_range_raises_error() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="between 0 and 1",
    ):
        calculate_precision_recall_at_k(
            np.array([0, 1]),
            np.array([0.2, 1.5]),
            k=1,
        )


def test_non_finite_probability_raises_error() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="finite",
    ):
        calculate_precision_recall_at_k(
            np.array([0, 1]),
            np.array([0.2, np.nan]),
            k=1,
        )


def test_k_must_be_positive() -> None:
    y_true, y_proba = make_data()

    with pytest.raises(
        RankingEvaluationError,
        match="greater than zero",
    ):
        calculate_precision_recall_at_k(
            y_true,
            y_proba,
            k=0,
        )


def test_k_cannot_exceed_dataset_size() -> None:
    y_true, y_proba = make_data()

    with pytest.raises(
        RankingEvaluationError,
        match="cannot exceed",
    ):
        calculate_precision_recall_at_k(
            y_true,
            y_proba,
            k=len(y_true) + 1,
        )


def test_k_must_be_integer() -> None:
    y_true, y_proba = make_data()

    with pytest.raises(
        RankingEvaluationError,
        match="integer",
    ):
        calculate_precision_recall_at_k(
            y_true,
            y_proba,
            k=2.5,
        )


def test_empty_k_values_raise_error() -> None:
    y_true, y_proba = make_data()

    with pytest.raises(
        RankingEvaluationError,
        match="At least one k",
    ):
        calculate_ranking_metrics(
            y_true,
            y_proba,
            k_values=[],
        )


def test_empty_metrics_dataframe_raises_error() -> None:
    with pytest.raises(
        RankingEvaluationError,
        match="At least one ranking metric",
    ):
        ranking_metrics_to_dataframe([])