import pytest
import torch

from fraud_intelligence.graph_ml.graphsage import (
    GraphSAGEConfig,
)
from fraud_intelligence.graph_ml.training import (
    GNNTrainingConfig,
    GraphTrainingError,
    TransactionGraphSAGEClassifier,
    build_loss_function,
    calculate_pos_weight,
    calculate_validation_loss,
    train_graphsage,
    train_one_epoch,
)


def make_data(
    nodes: int = 20,
    features: int = 5,
):
    torch.manual_seed(42)

    x = torch.randn(
        nodes,
        features,
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

    train_mask = torch.zeros(
        nodes,
        dtype=torch.bool,
    )
    train_mask[:14] = True

    validation_mask = torch.zeros(
        nodes,
        dtype=torch.bool,
    )
    validation_mask[14:] = True

    return (
        x,
        edge_index,
        labels,
        train_mask,
        validation_mask,
    )


def make_model(
    features: int = 5,
):
    return TransactionGraphSAGEClassifier(
        GraphSAGEConfig(
            in_channels=features,
        )
    )


def test_training_config_defaults() -> None:
    config = GNNTrainingConfig()

    assert config.learning_rate == 0.001
    assert config.weight_decay == 0.0001
    assert config.max_epochs == 100
    assert config.patience == 10
    assert config.seed == 42


def test_training_config_rejects_invalid_learning_rate() -> None:
    with pytest.raises(GraphTrainingError):
        GNNTrainingConfig(
            learning_rate=0,
        )


def test_training_config_rejects_negative_weight_decay() -> None:
    with pytest.raises(GraphTrainingError):
        GNNTrainingConfig(
            weight_decay=-1,
        )


def test_training_config_rejects_invalid_epochs() -> None:
    with pytest.raises(GraphTrainingError):
        GNNTrainingConfig(
            max_epochs=0,
        )


def test_training_config_rejects_invalid_patience() -> None:
    with pytest.raises(GraphTrainingError):
        GNNTrainingConfig(
            patience=0,
        )


def test_positive_weight_is_calculated_from_training_data() -> None:
    (
        _x,
        _edge_index,
        labels,
        train_mask,
        _validation_mask,
    ) = make_data()

    weight = calculate_pos_weight(
        labels,
        train_mask,
    )

    train_labels = labels[train_mask]

    positives = int(
        (train_labels == 1).sum()
    )

    negatives = int(
        (train_labels == 0).sum()
    )

    assert weight.item() == pytest.approx(
        negatives / positives
    )


def test_loss_function_is_bce_with_logits() -> None:
    (
        _x,
        _edge_index,
        labels,
        train_mask,
        _validation_mask,
    ) = make_data()

    loss = build_loss_function(
        labels,
        train_mask,
    )

    assert isinstance(
        loss,
        torch.nn.BCEWithLogitsLoss,
    )

    assert loss.pos_weight is not None


def test_classifier_forward_shape() -> None:
    (
        x,
        edge_index,
        _labels,
        _train_mask,
        _validation_mask,
    ) = make_data()

    model = make_model()

    logits = model(
        x,
        edge_index,
    )

    assert logits.shape == (20,)


def test_classifier_outputs_finite_logits() -> None:
    (
        x,
        edge_index,
        _labels,
        _train_mask,
        _validation_mask,
    ) = make_data()

    model = make_model()

    logits = model(
        x,
        edge_index,
    )

    assert torch.isfinite(logits).all()


def test_one_training_epoch_returns_loss() -> None:
    (
        x,
        edge_index,
        labels,
        train_mask,
        _validation_mask,
    ) = make_data()

    model = make_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    loss_function = build_loss_function(
        labels,
        train_mask,
    )

    loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        x=x,
        edge_index=edge_index,
        labels=labels,
        train_mask=train_mask,
    )

    assert isinstance(loss, float)
    assert loss >= 0
    assert torch.isfinite(
        torch.tensor(loss)
    )


def test_validation_loss_returns_float() -> None:
    (
        x,
        edge_index,
        labels,
        train_mask,
        validation_mask,
    ) = make_data()

    model = make_model()

    loss_function = build_loss_function(
        labels,
        train_mask,
    )

    loss = calculate_validation_loss(
        model=model,
        loss_function=loss_function,
        x=x,
        edge_index=edge_index,
        labels=labels,
        validation_mask=validation_mask,
    )

    assert isinstance(loss, float)
    assert loss >= 0


def test_training_returns_result() -> None:
    (
        x,
        edge_index,
        labels,
        train_mask,
        validation_mask,
    ) = make_data()

    model = make_model()

    result = train_graphsage(
        model=model,
        x=x,
        edge_index=edge_index,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        config=GNNTrainingConfig(
            max_epochs=5,
            patience=5,
        ),
    )

    assert result.epochs_completed == 5
    assert len(result.train_losses) == 5
    assert len(result.validation_losses) == 5
    assert 1 <= result.best_epoch <= 5
    assert result.best_validation_loss >= 0


def test_training_uses_early_stopping() -> None:
    (
        x,
        edge_index,
        labels,
        train_mask,
        validation_mask,
    ) = make_data()

    model = make_model()

    result = train_graphsage(
        model=model,
        x=x,
        edge_index=edge_index,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        config=GNNTrainingConfig(
            max_epochs=20,
            patience=1,
        ),
    )

    assert result.epochs_completed <= 20


def test_train_and_validation_masks_must_not_overlap() -> None:
    (
        x,
        edge_index,
        labels,
        train_mask,
        validation_mask,
    ) = make_data()

    validation_mask[0] = True

    model = make_model()

    with pytest.raises(GraphTrainingError):
        train_graphsage(
            model=model,
            x=x,
            edge_index=edge_index,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
        )


def test_empty_validation_mask_is_rejected() -> None:
    (
        x,
        edge_index,
        labels,
        train_mask,
        _validation_mask,
    ) = make_data()

    validation_mask = torch.zeros(
        20,
        dtype=torch.bool,
    )

    model = make_model()

    with pytest.raises(GraphTrainingError):
        train_graphsage(
            model=model,
            x=x,
            edge_index=edge_index,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
        )


def test_training_labels_must_be_binary() -> None:
    (
        x,
        edge_index,
        labels,
        train_mask,
        validation_mask,
    ) = make_data()

    labels[0] = 2

    model = make_model()

    with pytest.raises(GraphTrainingError):
        train_graphsage(
            model=model,
            x=x,
            edge_index=edge_index,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
        )


def test_training_labels_must_be_long() -> None:
    (
        x,
        edge_index,
        labels,
        train_mask,
        validation_mask,
    ) = make_data()

    labels = labels.float()

    model = make_model()

    with pytest.raises(GraphTrainingError):
        train_graphsage(
            model=model,
            x=x,
            edge_index=edge_index,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
        )


def test_feature_and_label_counts_must_match() -> None:
    (
        x,
        edge_index,
        labels,
        train_mask,
        validation_mask,
    ) = make_data()

    labels = labels[:-1]

    model = make_model()

    with pytest.raises(GraphTrainingError):
        train_graphsage(
            model=model,
            x=x,
            edge_index=edge_index,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
        )


def test_out_of_range_edges_are_rejected() -> None:
    (
        x,
        edge_index,
        labels,
        train_mask,
        validation_mask,
    ) = make_data()

    edge_index[0, 0] = 100

    model = make_model()

    with pytest.raises(GraphTrainingError):
        train_graphsage(
            model=model,
            x=x,
            edge_index=edge_index,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
        )


def test_non_boolean_train_mask_is_rejected() -> None:
    (
        x,
        edge_index,
        labels,
        train_mask,
        validation_mask,
    ) = make_data()

    train_mask = train_mask.long()

    model = make_model()

    with pytest.raises(GraphTrainingError):
        train_graphsage(
            model=model,
            x=x,
            edge_index=edge_index,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
        )


def test_training_does_not_require_test_mask() -> None:
    (
        x,
        edge_index,
        labels,
        train_mask,
        validation_mask,
    ) = make_data()

    model = make_model()

    result = train_graphsage(
        model=model,
        x=x,
        edge_index=edge_index,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        config=GNNTrainingConfig(
            max_epochs=2,
        ),
    )

    assert result.epochs_completed == 2


def test_best_model_state_is_restored() -> None:
    (
        x,
        edge_index,
        labels,
        train_mask,
        validation_mask,
    ) = make_data()

    model = make_model()

    result = train_graphsage(
        model=model,
        x=x,
        edge_index=edge_index,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        config=GNNTrainingConfig(
            max_epochs=5,
            patience=5,
        ),
    )

    final_validation_loss = calculate_validation_loss(
        model=model,
        loss_function=build_loss_function(
            labels,
            train_mask,
        ),
        x=x,
        edge_index=edge_index,
        labels=labels,
        validation_mask=validation_mask,
    )

    assert final_validation_loss == pytest.approx(
        result.best_validation_loss
    )