from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch_geometric.data import HeteroData

from fraud_intelligence.graph.contracts import (
    EDGE_TYPES,
    NODE_TYPES,
)
from fraud_intelligence.graph.heterogeneous import (
    GRAPH_EDGE_TYPES,
    HeterogeneousGraph,
)
from fraud_intelligence.graph.nodes import NODE_ID_COLUMN


class GraphValidationError(ValueError):
    """Raised when a heterogeneous graph is invalid."""


EXPECTED_EDGE_INDEX_DIMENSIONS: Final[int] = 2
EXPECTED_EDGE_INDEX_ROWS: Final[int] = 2


@dataclass(frozen=True)
class GraphValidationResult:
    """Summary of a successful graph validation."""

    valid: bool
    node_count: int
    edge_count: int
    node_types: tuple[str, ...]
    edge_types: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        if not self.valid:
            raise GraphValidationError(
                "GraphValidationResult cannot represent "
                "an invalid graph."
            )


def _validate_graph_object(
    graph: HeterogeneousGraph,
) -> None:
    if not isinstance(
        graph,
        HeterogeneousGraph,
    ):
        raise GraphValidationError(
            "graph must be a HeterogeneousGraph."
        )

    if not isinstance(
        graph.data,
        HeteroData,
    ):
        raise GraphValidationError(
            "graph.data must be a HeteroData object."
        )


def _validate_node_type_contract(
    graph: HeterogeneousGraph,
) -> None:
    actual_node_types = set(
        graph.data.node_types
    )

    expected_node_types = set(
        NODE_TYPES
    )

    missing = (
        expected_node_types
        - actual_node_types
    )

    if missing:
        raise GraphValidationError(
            "Missing node types: "
            + ", ".join(sorted(missing))
        )

    unexpected = (
        actual_node_types
        - expected_node_types
    )

    if unexpected:
        raise GraphValidationError(
            "Unexpected node types: "
            + ", ".join(sorted(unexpected))
        )


def _validate_edge_type_contract(
    graph: HeterogeneousGraph,
) -> None:
    actual_edge_types = set(
        graph.data.edge_types
    )

    expected_edge_types = set(
        GRAPH_EDGE_TYPES.values()
    )

    missing = (
        expected_edge_types
        - actual_edge_types
    )

    if missing:
        missing_text = ", ".join(
            str(edge_type)
            for edge_type in sorted(missing)
        )

        raise GraphValidationError(
            "Missing edge types: "
            + missing_text
        )

    unexpected = (
        actual_edge_types
        - expected_edge_types
    )

    if unexpected:
        unexpected_text = ", ".join(
            str(edge_type)
            for edge_type in sorted(unexpected)
        )

        raise GraphValidationError(
            "Unexpected edge types: "
            + unexpected_text
        )


def _validate_node_ids(
    graph: HeterogeneousGraph,
) -> None:
    for node_type in NODE_TYPES:
        store = graph.data[node_type]

        if NODE_ID_COLUMN not in store:
            raise GraphValidationError(
                f"Missing {NODE_ID_COLUMN} for "
                f"node type {node_type}."
            )

        node_ids = store[
            NODE_ID_COLUMN
        ]

        expected_count = graph.node_counts[
            node_type
        ]

        if not isinstance(
            node_ids,
            torch.Tensor,
        ):
            raise GraphValidationError(
                f"Node IDs for {node_type} "
                "must be torch.Tensor."
            )

        if node_ids.dtype != torch.long:
            raise GraphValidationError(
                f"Node IDs for {node_type} "
                "must use torch.long."
            )

        if node_ids.ndim != 1:
            raise GraphValidationError(
                f"Node IDs for {node_type} "
                "must be one-dimensional."
            )

        if node_ids.numel() != expected_count:
            raise GraphValidationError(
                f"Node ID count mismatch for "
                f"{node_type}."
            )

        expected_ids = torch.arange(
            expected_count,
            dtype=torch.long,
        )

        if not torch.equal(
            node_ids,
            expected_ids,
        ):
            raise GraphValidationError(
                f"Node IDs for {node_type} "
                "must be contiguous and zero-based."
            )


def _validate_edge_index(
    graph: HeterogeneousGraph,
    edge_type: tuple[str, str, str],
) -> None:
    source_type, relation_type, target_type = (
        edge_type
    )

    store = graph.data[edge_type]

    if "edge_index" not in store:
        raise GraphValidationError(
            f"Missing edge_index for "
            f"{edge_type}."
        )

    edge_index = store.edge_index

    if not isinstance(
        edge_index,
        torch.Tensor,
    ):
        raise GraphValidationError(
            f"edge_index for {edge_type} "
            "must be a torch.Tensor."
        )

    if edge_index.dtype != torch.long:
        raise GraphValidationError(
            f"edge_index for {edge_type} "
            "must use torch.long."
        )

    if edge_index.ndim != EXPECTED_EDGE_INDEX_DIMENSIONS:
        raise GraphValidationError(
            f"edge_index for {edge_type} "
            "must have two dimensions."
        )

    if (
        edge_index.shape[0]
        != EXPECTED_EDGE_INDEX_ROWS
    ):
        raise GraphValidationError(
            f"edge_index for {edge_type} "
            "must have shape [2, E]."
        )

    expected_edge_count = graph.edge_counts[
        relation_type
    ]

    if (
        edge_index.shape[1]
        != expected_edge_count
    ):
        raise GraphValidationError(
            f"Edge count mismatch for "
            f"{edge_type}: expected "
            f"{expected_edge_count}, got "
            f"{edge_index.shape[1]}."
        )

    source_count = graph.node_counts[
        source_type
    ]

    target_count = graph.node_counts[
        target_type
    ]

    if edge_index.numel() == 0:
        return

    source_ids = edge_index[0]
    target_ids = edge_index[1]

    if (
        source_ids.min().item() < 0
        or source_ids.max().item()
        >= source_count
    ):
        raise GraphValidationError(
            f"Source node IDs for {edge_type} "
            "are outside the valid range."
        )

    if (
        target_ids.min().item() < 0
        or target_ids.max().item()
        >= target_count
    ):
        raise GraphValidationError(
            f"Target node IDs for {edge_type} "
            "are outside the valid range."
        )


def _validate_no_duplicate_edges(
    graph: HeterogeneousGraph,
    edge_type: tuple[str, str, str],
) -> None:
    edge_index = graph.data[
        edge_type
    ].edge_index

    if edge_index.shape[1] <= 1:
        return

    # Transpose into [E, 2] so each row represents
    # one source-target relationship.
    edge_pairs = edge_index.t()

    unique_pairs = torch.unique(
        edge_pairs,
        dim=0,
    )

    if unique_pairs.shape[0] != edge_pairs.shape[0]:
        raise GraphValidationError(
            f"Duplicate edges detected for "
            f"{edge_type}."
        )


def _validate_edge_contracts(
    graph: HeterogeneousGraph,
) -> None:
    for edge_type in EDGE_TYPES:
        if edge_type not in graph.edge_counts:
            raise GraphValidationError(
                f"Missing edge count metadata for "
                f"{edge_type}."
            )

        graph_edge_type = GRAPH_EDGE_TYPES[
            edge_type
        ]

        _validate_edge_index(
            graph,
            graph_edge_type,
        )

        _validate_no_duplicate_edges(
            graph,
            graph_edge_type,
        )


def _validate_count_metadata(
    graph: HeterogeneousGraph,
) -> None:
    if set(graph.node_counts) != set(
        NODE_TYPES
    ):
        raise GraphValidationError(
            "node_counts metadata does not match "
            "the canonical node types."
        )

    if set(graph.edge_counts) != set(
        EDGE_TYPES
    ):
        raise GraphValidationError(
            "edge_counts metadata does not match "
            "the canonical edge types."
        )

    for node_type, count in (
        graph.node_counts.items()
    ):
        if not isinstance(count, int):
            raise GraphValidationError(
                f"Node count for {node_type} "
                "must be an integer."
            )

        if count < 0:
            raise GraphValidationError(
                f"Node count for {node_type} "
                "cannot be negative."
            )

    for edge_type, count in (
        graph.edge_counts.items()
    ):
        if not isinstance(count, int):
            raise GraphValidationError(
                f"Edge count for {edge_type} "
                "must be an integer."
            )

        if count < 0:
            raise GraphValidationError(
                f"Edge count for {edge_type} "
                "cannot be negative."
            )


def validate_graph(
    graph: HeterogeneousGraph,
) -> GraphValidationResult:
    """
    Perform complete structural validation of the
    Phase 6.5 heterogeneous graph.
    """

    _validate_graph_object(graph)
    _validate_count_metadata(graph)
    _validate_node_type_contract(graph)
    _validate_edge_type_contract(graph)
    _validate_node_ids(graph)
    _validate_edge_contracts(graph)

    return GraphValidationResult(
        valid=True,
        node_count=graph.total_nodes,
        edge_count=graph.total_edges,
        node_types=tuple(
            graph.data.node_types
        ),
        edge_types=tuple(
            graph.data.edge_types
        ),
    )