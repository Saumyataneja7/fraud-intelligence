import pytest
import torch

from fraud_intelligence.graph_ml.graphsage import (
    GraphSAGEConfig,
    GraphSAGEError,
    TransactionGraphSAGE,
    build_graphsage_baseline,
)


def make_features(
    nodes: int = 10,
    features: int = 5,
) -> torch.Tensor:
    return torch.arange(
        nodes * features,
        dtype=torch.float32,
    ).reshape(nodes, features)


def make_edge_index(
    nodes: int = 10,
) -> torch.Tensor:
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

    return torch.stack(
        [
            torch.cat([source, target]),
            torch.cat([target, source]),
        ],
        dim=0,
    )


def test_config_is_created() -> None:
    config = GraphSAGEConfig(
        in_channels=53,
    )

    assert config.in_channels == 53
    assert config.hidden_channels == 64
    assert config.out_channels == 32
    assert config.dropout == 0.2


def test_model_is_created() -> None:
    model = build_graphsage_baseline(
        in_channels=53,
    )

    assert isinstance(
        model,
        TransactionGraphSAGE,
    )


def test_model_has_two_sage_layers() -> None:
    model = build_graphsage_baseline(
        in_channels=53,
    )

    assert model.conv1.in_channels == 53
    assert model.conv1.out_channels == 64

    assert model.conv2.in_channels == 64
    assert model.conv2.out_channels == 32


def test_forward_returns_node_embeddings() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    x = make_features()
    edge_index = make_edge_index()

    output = model(
        x,
        edge_index,
    )

    assert output.shape == (10, 32)


def test_forward_output_is_float_tensor() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    output = model(
        make_features(),
        make_edge_index(),
    )

    assert output.dtype == torch.float32


def test_forward_output_is_finite() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    output = model(
        make_features(),
        make_edge_index(),
    )

    assert torch.isfinite(output).all()


def test_different_node_count_is_supported() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    x = make_features(
        nodes=25,
        features=5,
    )

    edge_index = make_edge_index(
        nodes=25,
    )

    output = model(
        x,
        edge_index,
    )

    assert output.shape == (25, 32)


def test_empty_edge_index_is_supported() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    x = make_features()

    edge_index = torch.empty(
        (2, 0),
        dtype=torch.long,
    )

    output = model(
        x,
        edge_index,
    )

    assert output.shape == (10, 32)


def test_wrong_feature_dimension_is_rejected() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    x = make_features(
        features=4,
    )

    with pytest.raises(GraphSAGEError):
        model(
            x,
            make_edge_index(),
        )


def test_integer_features_are_rejected() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    x = torch.ones(
        (10, 5),
        dtype=torch.long,
    )

    with pytest.raises(GraphSAGEError):
        model(
            x,
            make_edge_index(),
        )


def test_nan_features_are_rejected() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    x = make_features()
    x[0, 0] = float("nan")

    with pytest.raises(GraphSAGEError):
        model(
            x,
            make_edge_index(),
        )


def test_infinite_features_are_rejected() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    x = make_features()
    x[0, 0] = float("inf")

    with pytest.raises(GraphSAGEError):
        model(
            x,
            make_edge_index(),
        )


def test_wrong_edge_index_shape_is_rejected() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    edge_index = torch.zeros(
        (3, 10),
        dtype=torch.long,
    )

    with pytest.raises(GraphSAGEError):
        model(
            make_features(),
            edge_index,
        )


def test_non_long_edge_index_is_rejected() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    edge_index = make_edge_index().float()

    with pytest.raises(GraphSAGEError):
        model(
            make_features(),
            edge_index,
        )


def test_negative_edge_index_is_rejected() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    edge_index = make_edge_index()
    edge_index[0, 0] = -1

    with pytest.raises(GraphSAGEError):
        model(
            make_features(),
            edge_index,
        )


def test_out_of_range_edge_index_is_rejected() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    edge_index = make_edge_index()
    edge_index[0, 0] = 100

    with pytest.raises(GraphSAGEError):
        model(
            make_features(),
            edge_index,
        )


def test_invalid_input_channels_are_rejected() -> None:
    with pytest.raises(GraphSAGEError):
        GraphSAGEConfig(
            in_channels=0,
        )


def test_invalid_hidden_channels_are_rejected() -> None:
    with pytest.raises(GraphSAGEError):
        GraphSAGEConfig(
            in_channels=5,
            hidden_channels=0,
        )


def test_invalid_output_channels_are_rejected() -> None:
    with pytest.raises(GraphSAGEError):
        GraphSAGEConfig(
            in_channels=5,
            out_channels=0,
        )


def test_invalid_dropout_is_rejected() -> None:
    with pytest.raises(GraphSAGEError):
        GraphSAGEConfig(
            in_channels=5,
            dropout=1.0,
        )


def test_negative_dropout_is_rejected() -> None:
    with pytest.raises(GraphSAGEError):
        GraphSAGEConfig(
            in_channels=5,
            dropout=-0.1,
        )


def test_model_is_deterministically_constructible() -> None:
    torch.manual_seed(42)

    model_a = build_graphsage_baseline(
        in_channels=5,
    )

    torch.manual_seed(42)

    model_b = build_graphsage_baseline(
        in_channels=5,
    )

    for parameter_a, parameter_b in zip(
        model_a.parameters(),
        model_b.parameters(),
    ):
        assert torch.equal(
            parameter_a,
            parameter_b,
        )


def test_model_parameters_require_grad() -> None:
    model = build_graphsage_baseline(
        in_channels=5,
    )

    assert all(
        parameter.requires_grad
        for parameter in model.parameters()
    )


def test_custom_architecture_configuration() -> None:
    model = build_graphsage_baseline(
        in_channels=53,
        hidden_channels=128,
        out_channels=16,
        dropout=0.1,
    )

    assert model.config.in_channels == 53
    assert model.config.hidden_channels == 128
    assert model.config.out_channels == 16
    assert model.config.dropout == 0.1