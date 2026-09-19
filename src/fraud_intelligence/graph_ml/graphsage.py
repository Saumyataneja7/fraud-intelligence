from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


DEFAULT_HIDDEN_CHANNELS: Final[int] = 64
DEFAULT_OUT_CHANNELS: Final[int] = 32
DEFAULT_DROPOUT: Final[float] = 0.2


class GraphSAGEError(ValueError):
    """Raised when GraphSAGE configuration or input is invalid."""


@dataclass(frozen=True)
class GraphSAGEConfig:
    """Configuration for the transaction GraphSAGE baseline."""

    in_channels: int
    hidden_channels: int = DEFAULT_HIDDEN_CHANNELS
    out_channels: int = DEFAULT_OUT_CHANNELS
    dropout: float = DEFAULT_DROPOUT

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise GraphSAGEError(
                "in_channels must be greater than zero."
            )

        if self.hidden_channels <= 0:
            raise GraphSAGEError(
                "hidden_channels must be greater than zero."
            )

        if self.out_channels <= 0:
            raise GraphSAGEError(
                "out_channels must be greater than zero."
            )

        if not 0.0 <= self.dropout < 1.0:
            raise GraphSAGEError(
                "dropout must be in the range [0.0, 1.0)."
            )


class TransactionGraphSAGE(nn.Module):
    """
    Two-layer GraphSAGE encoder for transaction nodes.

    This is an architecture-only baseline for Phase 7.5.
    Training is intentionally handled in Phase 7.6.
    """

    def __init__(
        self,
        config: GraphSAGEConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.conv1 = SAGEConv(
            config.in_channels,
            config.hidden_channels,
        )

        self.conv2 = SAGEConv(
            config.hidden_channels,
            config.out_channels,
        )

        self.dropout = nn.Dropout(
            p=config.dropout,
        )

        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce transaction node embeddings.
        """

        _validate_inputs(
            x=x,
            edge_index=edge_index,
            expected_in_channels=self.config.in_channels,
        )

        x = self.conv1(
            x,
            edge_index,
        )

        x = self.activation(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index,
        )

        return x


def _validate_inputs(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    expected_in_channels: int,
) -> None:
    if not isinstance(x, torch.Tensor):
        raise GraphSAGEError(
            "x must be a torch.Tensor."
        )

    if x.ndim != 2:
        raise GraphSAGEError(
            "x must be a 2-dimensional tensor."
        )

    if x.shape[1] != expected_in_channels:
        raise GraphSAGEError(
            "Input feature dimension does not match "
            f"expected in_channels={expected_in_channels}; "
            f"got {x.shape[1]}."
        )

    if not torch.is_floating_point(x):
        raise GraphSAGEError(
            "x must contain floating-point features."
        )

    if not torch.isfinite(x).all():
        raise GraphSAGEError(
            "x must contain only finite values."
        )

    if not isinstance(edge_index, torch.Tensor):
        raise GraphSAGEError(
            "edge_index must be a torch.Tensor."
        )

    if edge_index.ndim != 2:
        raise GraphSAGEError(
            "edge_index must be 2-dimensional."
        )

    if edge_index.shape[0] != 2:
        raise GraphSAGEError(
            "edge_index must have shape [2, num_edges]."
        )

    if edge_index.dtype != torch.long:
        raise GraphSAGEError(
            "edge_index must use torch.long."
        )

    if edge_index.numel() > 0:
        if int(edge_index.min()) < 0:
            raise GraphSAGEError(
                "edge_index contains a negative node index."
            )

        if int(edge_index.max()) >= x.shape[0]:
            raise GraphSAGEError(
                "edge_index contains a node index outside "
                "the feature matrix."
            )


def build_graphsage_baseline(
    in_channels: int,
    hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
    out_channels: int = DEFAULT_OUT_CHANNELS,
    dropout: float = DEFAULT_DROPOUT,
) -> TransactionGraphSAGE:
    """
    Construct the Phase 7.5 GraphSAGE baseline.
    """

    config = GraphSAGEConfig(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
    )

    return TransactionGraphSAGE(config)