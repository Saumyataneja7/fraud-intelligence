import pytest
import torch

from fraud_intelligence.graph_ml.evaluation import (
    DEFAULT_THRESHOLD,
    GNNEvaluationResult,
    GNNPredictionResult,
    GraphEvaluationError,
    calculate_gnn_metrics,
    evaluate_graph_split,
    predict_graph,
)
from fraud_intelligence.graph_ml.graphsage import (
    GraphSAGEConfig,
)
from fraud_intelligence.graph_ml.training import (
    TransactionGraphSAGEClassifier,
)


def make_model() -> TransactionGraphSAGEClassifier:
    torch.manual_seed(42)

    return TransactionGraphSAGEClassifier(
        GraphSAGEConfig(
            in_channels=5,
            hidden_channels=8,
            out_channels=4,
            dropout=0.0,
        )
    )


def make_data(nodes: int = 20):
    torch.manual_seed(42)

    x = torch.randn(
        nodes,
        5,
    )

    source = torch.arange(
        0,
        nodes - 1,
        dtype=torch.long,
    )

    target = torch.arange(
        1,
        nodes,
        dtype=torch.long,
    )

    edge_index = torch.stack(
        [
            torch.cat([source, target]),
            torch.cat([target, source]),
        ]
    )

    labels = torch.zeros(
        nodes,
        dtype=torch.long,
    )

    labels[:4] = 1

    mask = torch.zeros(
        nodes,
        dtype=torch.bool,
    )

    mask[:10] = True

    return (
        x,
        edge_index,
        labels,
        mask,
    )


def make_prediction_result(
    probabilities: tuple[float, ...],
) -> GNNPredictionResult:
    predictions = tuple(
        int(probability >= DEFAULT_THRESHOLD)
        for probability in probabilities
    )

    return GNNPredictionResult(
        split_name="validation",
        node_indices=tuple(
            range(len(probabilities))
        ),
        probabilities=probabilities,
        predictions=predictions,
    )


def test_default_threshold() -> None:
    assert DEFAULT_THRESHOLD == 0.50


def test_prediction_result_count() -> None:
    result = make_prediction_result(
        (0.1, 0.8, 0.2)
    )

    assert result.count == 3


def test_prediction_result_positive_count() -> None:
    result = make_prediction_result(
        (0.1, 0.8, 0.2, 0.9)
    )

    assert result.predicted_positive == 2


def test_predict_graph_returns_prediction_result() -> None:
    (
        x,
        edge_index,
        _labels,
        mask,
    ) = make_data()

    result = predict_graph(
        model=make_model(),
        x=x,
        edge_index=edge_index,
        mask=mask,
        split_name="validation",
    )

    assert isinstance(
        result,
        GNNPredictionResult,
    )


def test_prediction_count_matches_mask() -> None:
    (
        x,
        edge_index,
        _labels,
        mask,
    ) = make_data()

    result = predict_graph(
        model=make_model(),
        x=x,
        edge_index=edge_index,
        mask=mask,
        split_name="validation",
    )

    assert result.count == int(mask.sum())


def test_prediction_probabilities_are_between_zero_and_one() -> None:
    (
        x,
        edge_index,
        _labels,
        mask,
    ) = make_data()

    result = predict_graph(
        model=make_model(),
        x=x,
        edge_index=edge_index,
        mask=mask,
        split_name="validation",
    )

    assert all(
        0.0 <= probability <= 1.0
        for probability in result.probabilities
    )


def test_predictions_are_binary() -> None:
    (
        x,
        edge_index,
        _labels,
        mask,
    ) = make_data()

    result = predict_graph(
        model=make_model(),
        x=x,
        edge_index=edge_index,
        mask=mask,
        split_name="validation",
    )

    assert set(result.predictions) <= {0, 1}


def test_prediction_indices_match_mask() -> None:
    (
        x,
        edge_index,
        _labels,
        mask,
    ) = make_data()

    result = predict_graph(
        model=make_model(),
        x=x,
        edge_index=edge_index,
        mask=mask,
        split_name="validation",
    )

    expected = tuple(
        torch.nonzero(
            mask,
            as_tuple=False,
        )
        .view(-1)
        .tolist()
    )

    assert result.node_indices == expected


def test_metrics_return_expected_type() -> None:
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    result = make_prediction_result(
        (0.1, 0.2, 0.8, 0.9)
    )

    metrics = calculate_gnn_metrics(
        labels,
        result,
    )

    assert isinstance(
        metrics,
        GNNEvaluationResult,
    )


def test_perfect_predictions_have_perfect_metrics() -> None:
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    result = make_prediction_result(
        (0.01, 0.10, 0.90, 0.99)
    )

    metrics = calculate_gnn_metrics(
        labels,
        result,
    )

    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)


def test_confusion_matrix_is_correct() -> None:
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    result = make_prediction_result(
        (0.1, 0.8, 0.2, 0.9)
    )

    metrics = calculate_gnn_metrics(
        labels,
        result,
    )

    assert metrics.true_negative == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.true_positive == 1


def test_support_is_correct() -> None:
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    result = make_prediction_result(
        (0.1, 0.2, 0.8, 0.9)
    )

    metrics = calculate_gnn_metrics(
        labels,
        result,
    )

    assert metrics.support == 4
    assert metrics.actual_positive == 2
    assert metrics.predicted_positive == 2


def test_confusion_matrix_property() -> None:
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    result = make_prediction_result(
        (0.1, 0.8, 0.2, 0.9)
    )

    metrics = calculate_gnn_metrics(
        labels,
        result,
    )

    assert metrics.confusion_matrix == (
        (1, 1),
        (1, 1),
    )


def test_split_name_is_preserved() -> None:
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    result = make_prediction_result(
        (0.1, 0.2, 0.8, 0.9)
    )

    metrics = calculate_gnn_metrics(
        labels,
        result,
    )

    assert metrics.split_name == "validation"


def test_custom_threshold_is_supported() -> None:
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    result = GNNPredictionResult(
        split_name="validation",
        node_indices=(0, 1, 2, 3),
        probabilities=(0.4, 0.6, 0.7, 0.8),
        predictions=(0, 1, 1, 1),
    )

    metrics = calculate_gnn_metrics(
        labels,
        result,
        threshold=0.7,
    )

    assert metrics.threshold == pytest.approx(0.7)


def test_threshold_zero_is_rejected() -> None:
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    result = make_prediction_result(
        (0.1, 0.2, 0.8, 0.9)
    )

    with pytest.raises(GraphEvaluationError):
        calculate_gnn_metrics(
            labels,
            result,
            threshold=0.0,
        )


def test_threshold_one_is_rejected() -> None:
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    result = make_prediction_result(
        (0.1, 0.2, 0.8, 0.9)
    )

    with pytest.raises(GraphEvaluationError):
        calculate_gnn_metrics(
            labels,
            result,
            threshold=1.0,
        )


def test_missing_class_is_rejected() -> None:
    labels = torch.tensor(
        [0, 0, 0, 0],
        dtype=torch.long,
    )

    result = make_prediction_result(
        (0.1, 0.2, 0.3, 0.4)
    )

    with pytest.raises(GraphEvaluationError):
        calculate_gnn_metrics(
            labels,
            result,
        )


def test_out_of_range_node_index_is_rejected() -> None:
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    result = GNNPredictionResult(
        split_name="validation",
        node_indices=(0, 1, 2, 99),
        probabilities=(0.1, 0.2, 0.8, 0.9),
        predictions=(0, 0, 1, 1),
    )

    with pytest.raises(GraphEvaluationError):
        calculate_gnn_metrics(
            labels,
            result,
        )


def test_empty_prediction_result_is_rejected() -> None:
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.long,
    )

    result = GNNPredictionResult(
        split_name="validation",
        node_indices=(),
        probabilities=(),
        predictions=(),
    )

    with pytest.raises(GraphEvaluationError):
        calculate_gnn_metrics(
            labels,
            result,
        )


def test_nonbinary_labels_are_rejected() -> None:
    labels = torch.tensor(
        [0, 1, 2, 1],
        dtype=torch.long,
    )

    result = make_prediction_result(
        (0.1, 0.2, 0.8, 0.9)
    )

    with pytest.raises(GraphEvaluationError):
        calculate_gnn_metrics(
            labels,
            result,
        )


def test_float_labels_are_rejected() -> None:
    labels = torch.tensor(
        [0, 0, 1, 1],
        dtype=torch.float32,
    )

    result = make_prediction_result(
        (0.1, 0.2, 0.8, 0.9)
    )

    with pytest.raises(GraphEvaluationError):
        calculate_gnn_metrics(
            labels,
            result,
        )


def test_evaluate_graph_split_runs_end_to_end() -> None:
    (
        x,
        edge_index,
        labels,
        mask,
    ) = make_data()

    predictions, metrics = evaluate_graph_split(
        model=make_model(),
        x=x,
        edge_index=edge_index,
        labels=labels,
        mask=mask,
        split_name="validation",
    )

    assert isinstance(
        predictions,
        GNNPredictionResult,
    )

    assert isinstance(
        metrics,
        GNNEvaluationResult,
    )

    assert metrics.support == predictions.count


def test_evaluation_does_not_change_model_parameters() -> None:
    (
        x,
        edge_index,
        _labels,
        mask,
    ) = make_data()

    model = make_model()

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    predict_graph(
        model=model,
        x=x,
        edge_index=edge_index,
        mask=mask,
        split_name="validation",
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(
            before[name],
            parameter,
        )


def test_model_is_returned_to_eval_mode() -> None:
    (
        x,
        edge_index,
        _labels,
        mask,
    ) = make_data()

    model = make_model()
    model.train()

    predict_graph(
        model=model,
        x=x,
        edge_index=edge_index,
        mask=mask,
        split_name="validation",
    )

    assert model.training is False