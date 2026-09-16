from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_intelligence.ml.threshold import (
    ThresholdMetrics,
    ThresholdOptimizationError,
    ThresholdOptimizationResult,
    calculate_threshold_metrics,
    optimize_threshold,
    threshold_candidates_to_dataframe,
    threshold_result_to_dict,
)


def make_data() -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1]
    )

    y_proba = np.array(
        [0.05, 0.10, 0.30, 0.60, 0.55, 0.70, 0.80, 0.95]
    )

    return y_true, y_proba


def test_calculate_threshold_metrics() -> None:
    y_true, y_proba = make_data()

    metrics = calculate_threshold_metrics(
        y_true,
        y_proba,
        threshold=0.5,
    )

    assert isinstance(
        metrics,
        ThresholdMetrics,
    )

    assert metrics.threshold == 0.5
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0


def test_threshold_prediction_uses_greater_than_or_equal() -> None:
    y_true = np.array([0, 1])
    y_proba = np.array([0.5, 0.5])

    metrics = calculate_threshold_metrics(
        y_true,
        y_proba,
        threshold=0.5,
    )

    assert metrics.predicted_positive_count == 2


def test_optimize_threshold_returns_result() -> None:
    y_true, y_proba = make_data()

    result = optimize_threshold(
        y_true,
        y_proba,
    )

    assert isinstance(
        result,
        ThresholdOptimizationResult,
    )

    assert 0.0 <= result.selected_threshold <= 1.0
    assert result.objective == "f1"


def test_selected_threshold_is_one_of_candidates() -> None:
    y_true, y_proba = make_data()

    thresholds = [0.2, 0.4, 0.6, 0.8]

    result = optimize_threshold(
        y_true,
        y_proba,
        thresholds=thresholds,
    )

    assert result.selected_threshold in thresholds


def test_custom_threshold_grid_is_preserved() -> None:
    y_true, y_proba = make_data()

    thresholds = [0.1, 0.3, 0.7, 0.9]

    result = optimize_threshold(
        y_true,
        y_proba,
        thresholds=thresholds,
    )

    assert len(result.candidates) == len(thresholds)

    assert {
        candidate.threshold
        for candidate in result.candidates
    } == set(thresholds)


def test_selected_result_matches_maximum_f1() -> None:
    y_true, y_proba = make_data()

    thresholds = [0.2, 0.4, 0.6, 0.8]

    result = optimize_threshold(
        y_true,
        y_proba,
        thresholds=thresholds,
    )

    best_f1 = max(
        candidate.f1
        for candidate in result.candidates
    )

    assert result.metrics.f1 == pytest.approx(best_f1)


def test_tie_breaking_prefers_higher_recall() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])

    result = optimize_threshold(
        y_true,
        y_proba,
        thresholds=[0.5, 0.8],
    )

    assert result.selected_threshold == 0.5


def test_empty_data_raises_error() -> None:
    with pytest.raises(
        ThresholdOptimizationError,
        match="cannot be empty",
    ):
        optimize_threshold(
            np.array([]),
            np.array([]),
        )


def test_length_mismatch_raises_error() -> None:
    with pytest.raises(
        ThresholdOptimizationError,
        match="same length",
    ):
        optimize_threshold(
            np.array([0, 1]),
            np.array([0.2]),
        )


def test_invalid_labels_raise_error() -> None:
    with pytest.raises(
        ThresholdOptimizationError,
        match="only 0 and 1",
    ):
        optimize_threshold(
            np.array([0, 1, 2]),
            np.array([0.1, 0.8, 0.9]),
        )


def test_single_class_raises_error() -> None:
    with pytest.raises(
        ThresholdOptimizationError,
        match="both classes",
    ):
        optimize_threshold(
            np.array([0, 0, 0]),
            np.array([0.1, 0.2, 0.3]),
        )


def test_probability_out_of_range_raises_error() -> None:
    with pytest.raises(
        ThresholdOptimizationError,
        match="between 0 and 1",
    ):
        optimize_threshold(
            np.array([0, 1]),
            np.array([0.2, 1.5]),
        )


def test_nan_probability_raises_error() -> None:
    with pytest.raises(
        ThresholdOptimizationError,
        match="finite",
    ):
        optimize_threshold(
            np.array([0, 1]),
            np.array([0.2, np.nan]),
        )


def test_invalid_threshold_raises_error() -> None:
    with pytest.raises(
        ThresholdOptimizationError,
        match="between 0 and 1",
    ):
        calculate_threshold_metrics(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            threshold=1.5,
        )


def test_empty_threshold_list_raises_error() -> None:
    with pytest.raises(
        ThresholdOptimizationError,
        match="At least one threshold",
    ):
        optimize_threshold(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            thresholds=[],
        )


def test_invalid_objective_raises_error() -> None:
    with pytest.raises(
        ThresholdOptimizationError,
        match="Only the 'f1' objective",
    ):
        optimize_threshold(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            objective="recall",
        )


def test_result_to_dict() -> None:
    y_true, y_proba = make_data()

    result = optimize_threshold(
        y_true,
        y_proba,
        thresholds=[0.3, 0.5, 0.7],
    )

    data = threshold_result_to_dict(result)

    assert "selected_threshold" in data
    assert "objective" in data
    assert "precision" in data
    assert "recall" in data
    assert "f1" in data


def test_candidates_to_dataframe() -> None:
    y_true, y_proba = make_data()

    result = optimize_threshold(
        y_true,
        y_proba,
        thresholds=[0.3, 0.5, 0.7],
    )

    dataframe = threshold_candidates_to_dataframe(
        result
    )

    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 3
    assert set(
        [
            "threshold",
            "precision",
            "recall",
            "f1",
            "predicted_positive_count",
            "actual_positive_count",
        ]
    ).issubset(dataframe.columns)