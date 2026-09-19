import pytest
import torch

from fraud_intelligence.graph_ml.threshold import (
    DEFAULT_MAX_THRESHOLD,
    DEFAULT_MIN_THRESHOLD,
    DEFAULT_THRESHOLD_STEP,
    GNNThresholdError,
    GNNThresholdMetrics,
    GNNThresholdOptimizationResult,
    calculate_threshold_metrics,
    generate_threshold_grid,
    optimize_gnn_threshold,
)


def make_labels() -> torch.Tensor:
    return torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1],
        dtype=torch.long,
    )


def make_probabilities() -> torch.Tensor:
    return torch.tensor(
        [0.05, 0.15, 0.25, 0.45, 0.55, 0.70, 0.85, 0.95],
        dtype=torch.float32,
    )


def test_default_threshold_configuration() -> None:
    assert DEFAULT_MIN_THRESHOLD == 0.05
    assert DEFAULT_MAX_THRESHOLD == 0.99
    assert DEFAULT_THRESHOLD_STEP == 0.01


def test_default_threshold_grid() -> None:
    thresholds = generate_threshold_grid()

    assert thresholds[0] == 0.05
    assert thresholds[-1] == 0.99
    assert len(thresholds) == 95


def test_custom_threshold_grid() -> None:
    thresholds = generate_threshold_grid(
        minimum=0.1,
        maximum=0.5,
        step=0.1,
    )

    assert thresholds == (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    )


def test_threshold_grid_rejects_invalid_minimum() -> None:
    with pytest.raises(GNNThresholdError):
        generate_threshold_grid(
            minimum=0.0,
        )


def test_threshold_grid_rejects_invalid_maximum() -> None:
    with pytest.raises(GNNThresholdError):
        generate_threshold_grid(
            maximum=1.0,
        )


def test_threshold_grid_rejects_reversed_range() -> None:
    with pytest.raises(GNNThresholdError):
        generate_threshold_grid(
            minimum=0.8,
            maximum=0.2,
        )


def test_threshold_grid_rejects_invalid_step() -> None:
    with pytest.raises(GNNThresholdError):
        generate_threshold_grid(
            step=0,
        )


def test_threshold_metrics_return_correct_type() -> None:
    result = calculate_threshold_metrics(
        labels=make_labels(),
        probabilities=make_probabilities(),
        threshold=0.5,
    )

    assert isinstance(
        result,
        GNNThresholdMetrics,
    )


def test_threshold_metrics_support() -> None:
    result = calculate_threshold_metrics(
        labels=make_labels(),
        probabilities=make_probabilities(),
        threshold=0.5,
    )

    assert result.support == 8


def test_threshold_metrics_actual_positive() -> None:
    result = calculate_threshold_metrics(
        labels=make_labels(),
        probabilities=make_probabilities(),
        threshold=0.5,
    )

    assert result.actual_positive == 4


def test_threshold_metrics_predicted_positive() -> None:
    result = calculate_threshold_metrics(
        labels=make_labels(),
        probabilities=make_probabilities(),
        threshold=0.5,
    )

    assert result.predicted_positive == 4


def test_threshold_metrics_perfect_separation() -> None:
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    probabilities = torch.tensor(
        [0.1, 0.2, 0.8, 0.9],
        dtype=torch.float32,
    )

    result = calculate_threshold_metrics(
        labels=labels,
        probabilities=probabilities,
        threshold=0.5,
    )

    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


def test_lower_threshold_increases_or_preserves_recall() -> None:
    labels = make_labels()
    probabilities = make_probabilities()

    high = calculate_threshold_metrics(
        labels,
        probabilities,
        threshold=0.8,
    )

    low = calculate_threshold_metrics(
        labels,
        probabilities,
        threshold=0.3,
    )

    assert low.recall >= high.recall


def test_optimizer_returns_correct_type() -> None:
    result = optimize_gnn_threshold(
        labels=make_labels(),
        probabilities=make_probabilities(),
    )

    assert isinstance(
        result,
        GNNThresholdOptimizationResult,
    )


def test_optimizer_uses_validation_split_name() -> None:
    result = optimize_gnn_threshold(
        labels=make_labels(),
        probabilities=make_probabilities(),
        split_name="validation",
    )

    assert result.split_name == "validation"


def test_optimizer_objective_is_f1() -> None:
    result = optimize_gnn_threshold(
        labels=make_labels(),
        probabilities=make_probabilities(),
    )

    assert result.objective == "f1"


def test_optimizer_returns_candidate_grid() -> None:
    result = optimize_gnn_threshold(
        labels=make_labels(),
        probabilities=make_probabilities(),
    )

    assert result.candidate_count == 95


def test_optimizer_selected_threshold_is_candidate() -> None:
    result = optimize_gnn_threshold(
        labels=make_labels(),
        probabilities=make_probabilities(),
    )

    candidate_thresholds = {
        candidate.threshold
        for candidate in result.candidates
    }

    assert result.selected_threshold in candidate_thresholds


def test_optimizer_selected_metrics_match_selected_threshold() -> None:
    result = optimize_gnn_threshold(
        labels=make_labels(),
        probabilities=make_probabilities(),
    )

    assert (
        result.selected_metrics.threshold
        == result.selected_threshold
    )


def test_optimizer_rejects_empty_data() -> None:
    labels = torch.empty(
        0,
        dtype=torch.long,
    )

    probabilities = torch.empty(
        0,
        dtype=torch.float32,
    )

    with pytest.raises(GNNThresholdError):
        optimize_gnn_threshold(
            labels,
            probabilities,
        )


def test_optimizer_rejects_nonbinary_labels() -> None:
    labels = torch.tensor(
        [0, 1, 2, 1],
        dtype=torch.long,
    )

    probabilities = torch.tensor(
        [0.1, 0.2, 0.8, 0.9],
        dtype=torch.float32,
    )

    with pytest.raises(GNNThresholdError):
        optimize_gnn_threshold(
            labels,
            probabilities,
        )


def test_optimizer_rejects_mismatched_lengths() -> None:
    labels = torch.tensor(
        [0, 1, 1],
        dtype=torch.long,
    )

    probabilities = torch.tensor(
        [0.1, 0.8],
        dtype=torch.float32,
    )

    with pytest.raises(GNNThresholdError):
        optimize_gnn_threshold(
            labels,
            probabilities,
        )


def test_optimizer_rejects_nan_probabilities() -> None:
    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    probabilities = torch.tensor(
        [float("nan"), 0.8],
        dtype=torch.float32,
    )

    with pytest.raises(GNNThresholdError):
        optimize_gnn_threshold(
            labels,
            probabilities,
        )


def test_optimizer_rejects_out_of_range_probabilities() -> None:
    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    probabilities = torch.tensor(
        [-0.1, 1.2],
        dtype=torch.float32,
    )

    with pytest.raises(GNNThresholdError):
        optimize_gnn_threshold(
            labels,
            probabilities,
        )


def test_threshold_metrics_reject_non_float_probabilities() -> None:
    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    probabilities = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    with pytest.raises(GNNThresholdError):
        calculate_threshold_metrics(
            labels,
            probabilities,
            0.5,
        )


def test_threshold_metrics_reject_invalid_threshold() -> None:
    with pytest.raises(GNNThresholdError):
        calculate_threshold_metrics(
            make_labels(),
            make_probabilities(),
            0.0,
        )


def test_optimizer_rejects_empty_split_name() -> None:
    with pytest.raises(GNNThresholdError):
        optimize_gnn_threshold(
            make_labels(),
            make_probabilities(),
            split_name="",
        )