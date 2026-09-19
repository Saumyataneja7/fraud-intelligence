import pytest
import torch

from fraud_intelligence.graph_ml.ranking import (
    GNNRankingError,
    GNNRankingMetrics,
    GNNRankingResult,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_ranking_curve,
    calculate_ranking_metrics,
)


def make_labels() -> torch.Tensor:
    return torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1],
        dtype=torch.long,
    )


def make_probabilities() -> torch.Tensor:
    return torch.tensor(
        [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90],
        dtype=torch.float32,
    )


def test_ranking_metrics_type() -> None:
    result = calculate_ranking_metrics(
        make_labels(),
        make_probabilities(),
        k=4,
    )

    assert isinstance(
        result,
        GNNRankingMetrics,
    )


def test_top_k_contains_highest_probabilities() -> None:
    labels = torch.tensor(
        [1, 0, 0, 1],
        dtype=torch.long,
    )

    probabilities = torch.tensor(
        [0.90, 0.80, 0.10, 0.70],
        dtype=torch.float32,
    )

    result = calculate_ranking_metrics(
        labels,
        probabilities,
        k=2,
    )

    assert result.fraud_count_at_k == 1


def test_precision_at_k() -> None:
    result = calculate_ranking_metrics(
        make_labels(),
        make_probabilities(),
        k=4,
    )

    # Top 4 are probabilities 0.9, 0.8, 0.7, 0.6.
    # All four are fraud.
    assert result.precision_at_k == pytest.approx(1.0)


def test_recall_at_k() -> None:
    result = calculate_ranking_metrics(
        make_labels(),
        make_probabilities(),
        k=4,
    )

    assert result.recall_at_k == pytest.approx(1.0)


def test_precision_function() -> None:
    result = calculate_precision_at_k(
        make_labels(),
        make_probabilities(),
        k=4,
    )

    assert result == pytest.approx(1.0)


def test_recall_function() -> None:
    result = calculate_recall_at_k(
        make_labels(),
        make_probabilities(),
        k=4,
    )

    assert result == pytest.approx(1.0)


def test_k_one() -> None:
    result = calculate_ranking_metrics(
        make_labels(),
        make_probabilities(),
        k=1,
    )

    assert result.fraud_count_at_k == 1
    assert result.precision_at_k == pytest.approx(1.0)
    assert result.recall_at_k == pytest.approx(0.25)


def test_k_equals_support() -> None:
    result = calculate_ranking_metrics(
        make_labels(),
        make_probabilities(),
        k=8,
    )

    assert result.precision_at_k == pytest.approx(0.5)
    assert result.recall_at_k == pytest.approx(1.0)


def test_fraud_count_at_k() -> None:
    result = calculate_ranking_metrics(
        make_labels(),
        make_probabilities(),
        k=6,
    )

    assert result.fraud_count_at_k == 4


def test_total_fraud() -> None:
    result = calculate_ranking_metrics(
        make_labels(),
        make_probabilities(),
        k=4,
    )

    assert result.total_fraud == 4


def test_support() -> None:
    result = calculate_ranking_metrics(
        make_labels(),
        make_probabilities(),
        k=4,
    )

    assert result.support == 8


def test_split_name_is_preserved() -> None:
    result = calculate_ranking_metrics(
        make_labels(),
        make_probabilities(),
        k=4,
        split_name="validation",
    )

    assert result.split_name == "validation"


def test_ranking_curve_type() -> None:
    result = calculate_ranking_curve(
        make_labels(),
        make_probabilities(),
        k_values=[1, 2, 4],
    )

    assert isinstance(
        result,
        GNNRankingResult,
    )


def test_ranking_curve_contains_requested_k_values() -> None:
    result = calculate_ranking_curve(
        make_labels(),
        make_probabilities(),
        k_values=[1, 2, 4],
    )

    assert result.k_values == (
        1,
        2,
        4,
    )


def test_ranking_curve_metric_count() -> None:
    result = calculate_ranking_curve(
        make_labels(),
        make_probabilities(),
        k_values=[1, 2, 4],
    )

    assert len(result.metrics) == 3


def test_ranking_is_stable_for_equal_probabilities() -> None:
    labels = torch.tensor(
        [1, 0, 1, 0],
        dtype=torch.long,
    )

    probabilities = torch.tensor(
        [0.8, 0.8, 0.8, 0.8],
        dtype=torch.float32,
    )

    result = calculate_ranking_metrics(
        labels,
        probabilities,
        k=2,
    )

    # Stable ordering preserves original order:
    # index 0 then index 1.
    assert result.fraud_count_at_k == 1


def test_k_zero_is_rejected() -> None:
    with pytest.raises(GNNRankingError):
        calculate_ranking_metrics(
            make_labels(),
            make_probabilities(),
            k=0,
        )


def test_negative_k_is_rejected() -> None:
    with pytest.raises(GNNRankingError):
        calculate_ranking_metrics(
            make_labels(),
            make_probabilities(),
            k=-1,
        )


def test_k_larger_than_support_is_rejected() -> None:
    with pytest.raises(GNNRankingError):
        calculate_ranking_metrics(
            make_labels(),
            make_probabilities(),
            k=9,
        )


def test_non_integer_k_is_rejected() -> None:
    with pytest.raises(GNNRankingError):
        calculate_ranking_metrics(
            make_labels(),
            make_probabilities(),
            k=2.5,
        )


def test_empty_k_values_are_rejected() -> None:
    with pytest.raises(GNNRankingError):
        calculate_ranking_curve(
            make_labels(),
            make_probabilities(),
            k_values=[],
        )


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(GNNRankingError):
        calculate_ranking_metrics(
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            torch.tensor(
                [0.1],
                dtype=torch.float32,
            ),
            k=1,
        )


def test_nonbinary_labels_are_rejected() -> None:
    with pytest.raises(GNNRankingError):
        calculate_ranking_metrics(
            torch.tensor(
                [0, 1, 2],
                dtype=torch.long,
            ),
            torch.tensor(
                [0.1, 0.8, 0.9],
                dtype=torch.float32,
            ),
            k=2,
        )


def test_non_float_probabilities_are_rejected() -> None:
    with pytest.raises(GNNRankingError):
        calculate_ranking_metrics(
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            k=1,
        )


def test_nan_probabilities_are_rejected() -> None:
    with pytest.raises(GNNRankingError):
        calculate_ranking_metrics(
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            torch.tensor(
                [float("nan"), 0.9],
                dtype=torch.float32,
            ),
            k=1,
        )


def test_probability_out_of_range_is_rejected() -> None:
    with pytest.raises(GNNRankingError):
        calculate_ranking_metrics(
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            torch.tensor(
                [-0.1, 1.1],
                dtype=torch.float32,
            ),
            k=1,
        )


def test_zero_fraud_is_rejected_for_recall() -> None:
    labels = torch.zeros(
        5,
        dtype=torch.long,
    )

    probabilities = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        dtype=torch.float32,
    )

    with pytest.raises(GNNRankingError):
        calculate_ranking_metrics(
            labels,
            probabilities,
            k=2,
        )


def test_empty_inputs_are_rejected() -> None:
    labels = torch.empty(
        0,
        dtype=torch.long,
    )

    probabilities = torch.empty(
        0,
        dtype=torch.float32,
    )

    with pytest.raises(GNNRankingError):
        calculate_ranking_metrics(
            labels,
            probabilities,
            k=1,
        )


def test_empty_split_name_is_rejected() -> None:
    with pytest.raises(GNNRankingError):
        calculate_ranking_metrics(
            make_labels(),
            make_probabilities(),
            k=1,
            split_name="",
        )


def test_recall_never_exceeds_one() -> None:
    result = calculate_ranking_metrics(
        make_labels(),
        make_probabilities(),
        k=3,
    )

    assert 0.0 <= result.recall_at_k <= 1.0


def test_precision_never_exceeds_one() -> None:
    result = calculate_ranking_metrics(
        make_labels(),
        make_probabilities(),
        k=3,
    )

    assert 0.0 <= result.precision_at_k <= 1.0