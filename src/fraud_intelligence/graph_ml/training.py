from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from fraud_intelligence.graph_ml.graphsage import (
    GraphSAGEConfig,
    TransactionGraphSAGE,
)


class GraphTrainingError(ValueError):
    """Raised when GNN training inputs or configuration are invalid."""


@dataclass(frozen=True)
class GNNTrainingConfig:
    """Configuration for the Phase 7.6 GNN training pipeline."""

    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 100
    patience: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise GraphTrainingError(
                "learning_rate must be greater than zero."
            )

        if self.weight_decay < 0:
            raise GraphTrainingError(
                "weight_decay must be non-negative."
            )

        if self.max_epochs <= 0:
            raise GraphTrainingError(
                "max_epochs must be greater than zero."
            )

        if self.patience <= 0:
            raise GraphTrainingError(
                "patience must be greater than zero."
            )

        if self.seed < 0:
            raise GraphTrainingError(
                "seed must be non-negative."
            )


@dataclass(frozen=True)
class GNNTrainingResult:
    """Training history and best-model information."""

    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int

    @property
    def train_loss(self) -> float:
        return self.train_losses[-1]

    @property
    def validation_loss(self) -> float:
        return self.validation_losses[-1]


class TransactionGraphSAGEClassifier(nn.Module):
    """
    GraphSAGE encoder followed by a binary fraud classification head.

    The model produces one fraud logit per transaction node.
    """

    def __init__(
        self,
        graphsage_config: GraphSAGEConfig,
    ) -> None:
        super().__init__()

        self.encoder = TransactionGraphSAGE(
            graphsage_config,
        )

        self.classifier = nn.Linear(
            graphsage_config.out_channels,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.encoder(
            x,
            edge_index,
        )

        logits = self.classifier(
            embeddings,
        )

        return logits.squeeze(-1)


def set_deterministic_seed(seed: int) -> None:
    """Set deterministic random seeds for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_pos_weight(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate positive-class weighting from training labels only.
    """

    _validate_labels(labels)
    _validate_mask(
        train_mask,
        labels.shape[0],
        "train_mask",
    )

    train_labels = labels[train_mask]

    positive_count = int(
        (train_labels == 1).sum().item()
    )

    negative_count = int(
        (train_labels == 0).sum().item()
    )

    if positive_count == 0:
        raise GraphTrainingError(
            "Training split contains no positive labels."
        )

    if negative_count == 0:
        raise GraphTrainingError(
            "Training split contains no negative labels."
        )

    weight = negative_count / positive_count

    return torch.tensor(
        weight,
        dtype=torch.float32,
    )


def build_loss_function(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> nn.BCEWithLogitsLoss:
    """Build the weighted binary classification loss."""

    pos_weight = calculate_pos_weight(
        labels,
        train_mask,
    )

    return nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )


def train_one_epoch(
    model: TransactionGraphSAGEClassifier,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.BCEWithLogitsLoss,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> float:
    """Run exactly one training epoch."""

    _validate_training_inputs(
        model=model,
        x=x,
        edge_index=edge_index,
        labels=labels,
        mask=train_mask,
    )

    model.train()
    optimizer.zero_grad()

    logits = model(
        x,
        edge_index,
    )

    loss = loss_function(
        logits[train_mask],
        labels[train_mask].float(),
    )

    loss.backward()
    optimizer.step()

    return float(loss.detach().cpu().item())


@torch.no_grad()
def calculate_validation_loss(
    model: TransactionGraphSAGEClassifier,
    loss_function: nn.BCEWithLogitsLoss,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    validation_mask: torch.Tensor,
) -> float:
    """Calculate validation loss without updating model parameters."""

    _validate_training_inputs(
        model=model,
        x=x,
        edge_index=edge_index,
        labels=labels,
        mask=validation_mask,
    )

    model.eval()

    logits = model(
        x,
        edge_index,
    )

    loss = loss_function(
        logits[validation_mask],
        labels[validation_mask].float(),
    )

    return float(loss.detach().cpu().item())


def train_graphsage(
    model: TransactionGraphSAGEClassifier,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    config: GNNTrainingConfig | None = None,
) -> GNNTrainingResult:
    """
    Train GraphSAGE using train nodes and select the best epoch
    using validation loss.

    The test mask is intentionally not accepted here.
    """

    if config is None:
        config = GNNTrainingConfig()

    set_deterministic_seed(config.seed)

    _validate_training_inputs(
        model=model,
        x=x,
        edge_index=edge_index,
        labels=labels,
        mask=train_mask,
    )

    _validate_mask(
        validation_mask,
        labels.shape[0],
        "validation_mask",
    )

    if not validation_mask.any():
        raise GraphTrainingError(
            "validation_mask must contain at least one node."
        )

    if torch.any(
        train_mask & validation_mask
    ):
        raise GraphTrainingError(
            "Training and validation masks must not overlap."
        )

    loss_function = build_loss_function(
        labels,
        train_mask,
    )

    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_losses: list[float] = []
    validation_losses: list[float] = []

    best_validation_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            x=x,
            edge_index=edge_index,
            labels=labels,
            train_mask=train_mask,
        )

        validation_loss = calculate_validation_loss(
            model=model,
            loss_function=loss_function,
            x=x,
            edge_index=edge_index,
            labels=labels,
            validation_mask=validation_mask,
        )

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state_dict is None:
        raise GraphTrainingError(
            "Training completed without a valid model state."
        )

    model.load_state_dict(
        best_state_dict
    )

    return GNNTrainingResult(
        train_losses=tuple(train_losses),
        validation_losses=tuple(validation_losses),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        epochs_completed=len(train_losses),
    )


def _validate_training_inputs(
    model: TransactionGraphSAGEClassifier,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if not isinstance(model, TransactionGraphSAGEClassifier):
        raise GraphTrainingError(
            "model must be TransactionGraphSAGEClassifier."
        )

    if not isinstance(x, torch.Tensor):
        raise GraphTrainingError(
            "x must be a torch.Tensor."
        )

    if x.ndim != 2:
        raise GraphTrainingError(
            "x must be 2-dimensional."
        )

    if x.shape[0] == 0:
        raise GraphTrainingError(
            "x must contain at least one node."
        )

    if not torch.is_floating_point(x):
        raise GraphTrainingError(
            "x must contain floating-point features."
        )

    if not torch.isfinite(x).all():
        raise GraphTrainingError(
            "x must contain only finite values."
        )

    if not isinstance(edge_index, torch.Tensor):
        raise GraphTrainingError(
            "edge_index must be a torch.Tensor."
        )

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise GraphTrainingError(
            "edge_index must have shape [2, num_edges]."
        )

    if edge_index.dtype != torch.long:
        raise GraphTrainingError(
            "edge_index must use torch.long."
        )

    if edge_index.numel() > 0:
        if int(edge_index.min()) < 0:
            raise GraphTrainingError(
                "edge_index contains a negative node index."
            )

        if int(edge_index.max()) >= x.shape[0]:
            raise GraphTrainingError(
                "edge_index contains an out-of-range node index."
            )

    _validate_labels(labels)

    if labels.shape[0] != x.shape[0]:
        raise GraphTrainingError(
            "labels and x must contain the same number of nodes."
        )

    _validate_mask(
        mask,
        x.shape[0],
        "mask",
    )

    if not mask.any():
        raise GraphTrainingError(
            "Training mask must contain at least one node."
        )


def _validate_labels(
    labels: torch.Tensor,
) -> None:
    if not isinstance(labels, torch.Tensor):
        raise GraphTrainingError(
            "labels must be a torch.Tensor."
        )

    if labels.ndim != 1:
        raise GraphTrainingError(
            "labels must be 1-dimensional."
        )

    if labels.dtype != torch.long:
        raise GraphTrainingError(
            "labels must use torch.long."
        )

    unique_labels = torch.unique(labels)

    if not torch.all(
        (unique_labels == 0)
        | (unique_labels == 1)
    ):
        raise GraphTrainingError(
            "labels must contain only 0 and 1."
        )


def _validate_mask(
    mask: torch.Tensor,
    node_count: int,
    name: str,
) -> None:
    if not isinstance(mask, torch.Tensor):
        raise GraphTrainingError(
            f"{name} must be a torch.Tensor."
        )

    if mask.ndim != 1:
        raise GraphTrainingError(
            f"{name} must be 1-dimensional."
        )

    if mask.dtype != torch.bool:
        raise GraphTrainingError(
            f"{name} must use torch.bool."
        )

    if mask.shape[0] != node_count:
        raise GraphTrainingError(
            f"{name} must have one entry per node."
        )