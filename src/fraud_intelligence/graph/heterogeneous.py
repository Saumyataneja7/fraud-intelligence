from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from fraud_intelligence.graph.contracts import (
    EDGE_TYPES,
    NODE_TYPES,
)
from fraud_intelligence.graph.edges import (
    EdgeTable,
    SOURCE_NODE_COLUMN,
    TARGET_NODE_COLUMN,
    validate_edge_table,
)
from fraud_intelligence.graph.nodes import (
    NODE_ID_COLUMN,
    NodeTable,
    validate_node_table,
)


class HeterogeneousGraphError(ValueError):
    """Raised when heterogeneous graph construction fails."""


GRAPH_EDGE_TYPES: Final[
    dict[str, tuple[str, str, str]]
] = {
    "customer_account": (
        "customer",
        "customer_account",
        "account",
    ),
    "account_card": (
        "account",
        "account_card",
        "card",
    ),
    "customer_transaction": (
        "customer",
        "customer_transaction",
        "transaction",
    ),
    "account_transaction": (
        "account",
        "account_transaction",
        "transaction",
    ),
    "card_transaction": (
        "card",
        "card_transaction",
        "transaction",
    ),
    "customer_device": (
        "customer",
        "customer_device",
        "device",
    ),
    "customer_ip": (
        "customer",
        "customer_ip",
        "ip",
    ),
    "customer_merchant": (
        "customer",
        "customer_merchant",
        "merchant",
    ),
    "transaction_merchant": (
        "transaction",
        "transaction_merchant",
        "merchant",
    ),
    "transaction_device": (
        "transaction",
        "transaction_device",
        "device",
    ),
    "transaction_ip": (
        "transaction",
        "transaction_ip",
        "ip",
    ),
}


@dataclass(frozen=True)
class HeterogeneousGraph:
    """Constructed heterogeneous graph and basic metadata."""

    data: HeteroData
    node_counts: dict[str, int]
    edge_counts: dict[str, int]

    @property
    def total_nodes(self) -> int:
        return sum(self.node_counts.values())

    @property
    def total_edges(self) -> int:
        return sum(self.edge_counts.values())


def _validate_node_tables(
    node_tables: dict[str, NodeTable],
) -> None:
    if not node_tables:
        raise HeterogeneousGraphError(
            "node_tables cannot be empty."
        )

    expected_types = set(NODE_TYPES)

    actual_types = set(node_tables)

    missing = expected_types - actual_types

    if missing:
        raise HeterogeneousGraphError(
            "Missing node types: "
            + ", ".join(sorted(missing))
        )

    unknown = actual_types - expected_types

    if unknown:
        raise HeterogeneousGraphError(
            "Unknown node types: "
            + ", ".join(sorted(unknown))
        )

    for node_type, node_table in node_tables.items():
        if node_type != node_table.node_type:
            raise HeterogeneousGraphError(
                "Node table key does not match "
                f"NodeTable.node_type: {node_type}"
            )

        validate_node_table(node_table)


def _validate_edge_tables(
    edge_tables: dict[str, EdgeTable],
) -> None:
    if not edge_tables:
        raise HeterogeneousGraphError(
            "edge_tables cannot be empty."
        )

    expected_types = set(EDGE_TYPES)

    actual_types = set(edge_tables)

    missing = expected_types - actual_types

    if missing:
        raise HeterogeneousGraphError(
            "Missing edge types: "
            + ", ".join(sorted(missing))
        )

    unknown = actual_types - expected_types

    if unknown:
        raise HeterogeneousGraphError(
            "Unknown edge types: "
            + ", ".join(sorted(unknown))
        )

    for edge_type, edge_table in edge_tables.items():
        if edge_type != edge_table.edge_type:
            raise HeterogeneousGraphError(
                "Edge table key does not match "
                f"EdgeTable.edge_type: {edge_type}"
            )

        validate_edge_table(edge_table)


def _validate_edge_endpoints(
    edge_table: EdgeTable,
    node_tables: dict[str, NodeTable],
) -> None:
    source_type = edge_table.source_node_type
    target_type = edge_table.target_node_type

    if source_type not in node_tables:
        raise HeterogeneousGraphError(
            f"Missing source node table: {source_type}"
        )

    if target_type not in node_tables:
        raise HeterogeneousGraphError(
            f"Missing target node table: {target_type}"
        )

    source_count = node_tables[
        source_type
    ].node_count

    target_count = node_tables[
        target_type
    ].node_count

    dataframe = edge_table.dataframe

    source_ids = dataframe[
        SOURCE_NODE_COLUMN
    ]

    target_ids = dataframe[
        TARGET_NODE_COLUMN
    ]

    if (
        len(source_ids) > 0
        and (
            source_ids.min() < 0
            or source_ids.max() >= source_count
        )
    ):
        raise HeterogeneousGraphError(
            f"Source node IDs for {edge_table.edge_type} "
            "are outside the valid node range."
        )

    if (
        len(target_ids) > 0
        and (
            target_ids.min() < 0
            or target_ids.max() >= target_count
        )
    ):
        raise HeterogeneousGraphError(
            f"Target node IDs for {edge_table.edge_type} "
            "are outside the valid node range."
        )


def _build_edge_index(
    edge_table: EdgeTable,
) -> torch.Tensor:
    dataframe = edge_table.dataframe

    if dataframe.empty:
        return torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    source = torch.as_tensor(
        dataframe[
            SOURCE_NODE_COLUMN
        ].to_numpy(),
        dtype=torch.long,
    )

    target = torch.as_tensor(
        dataframe[
            TARGET_NODE_COLUMN
        ].to_numpy(),
        dtype=torch.long,
    )

    return torch.stack(
        [source, target],
        dim=0,
    )


def _add_node_types(
    data: HeteroData,
    node_tables: dict[str, NodeTable],
) -> dict[str, int]:
    node_counts: dict[str, int] = {}

    for node_type in NODE_TYPES:
        node_table = node_tables[node_type]

        count = node_table.node_count

        node_counts[node_type] = count

        # Explicitly establish the node population.
        data[node_type].num_nodes = count

        # Keep the internal graph IDs aligned with
        # Phase 6.2 / Phase 6.4.
        data[node_type][
            NODE_ID_COLUMN
        ] = torch.arange(
            count,
            dtype=torch.long,
        )

    return node_counts


def _add_edge_types(
    data: HeteroData,
    edge_tables: dict[str, EdgeTable],
    node_tables: dict[str, NodeTable],
) -> dict[str, int]:
    edge_counts: dict[str, int] = {}

    for edge_type in EDGE_TYPES:
        edge_table = edge_tables[edge_type]

        _validate_edge_endpoints(
            edge_table,
            node_tables,
        )

        expected_tuple = GRAPH_EDGE_TYPES[
            edge_type
        ]

        actual_tuple = (
            edge_table.source_node_type,
            edge_table.edge_type,
            edge_table.target_node_type,
        )

        if actual_tuple != expected_tuple:
            raise HeterogeneousGraphError(
                f"Edge type definition mismatch for "
                f"{edge_type}: expected "
                f"{expected_tuple}, got "
                f"{actual_tuple}"
            )

        edge_index = _build_edge_index(
            edge_table
        )

        data[
            expected_tuple
        ].edge_index = edge_index

        edge_counts[edge_type] = (
            edge_table.edge_count
        )

    return edge_counts


def build_heterogeneous_graph(
    node_tables: dict[str, NodeTable],
    edge_tables: dict[str, EdgeTable],
) -> HeterogeneousGraph:
    """
    Construct the Phase 6.5 heterogeneous graph.

    This function builds graph topology only. It does not
    derive topology from fraud labels or future transactions.
    """

    _validate_node_tables(node_tables)
    _validate_edge_tables(edge_tables)

    data = HeteroData()

    node_counts = _add_node_types(
        data,
        node_tables,
    )

    edge_counts = _add_edge_types(
        data,
        edge_tables,
        node_tables,
    )

    return HeterogeneousGraph(
        data=data,
        node_counts=node_counts,
        edge_counts=edge_counts,
    )


def validate_heterogeneous_graph(
    graph: HeterogeneousGraph,
) -> None:
    """Validate basic structural integrity of the constructed graph."""

    if not isinstance(
        graph,
        HeterogeneousGraph,
    ):
        raise HeterogeneousGraphError(
            "graph must be a HeterogeneousGraph."
        )

    data = graph.data

    if not isinstance(
        data,
        HeteroData,
    ):
        raise HeterogeneousGraphError(
            "graph.data must be a HeteroData object."
        )

    for node_type in NODE_TYPES:
        if node_type not in data.node_types:
            raise HeterogeneousGraphError(
                f"Missing graph node type: {node_type}"
            )

        expected_count = graph.node_counts[
            node_type
        ]

        actual_count = data[
            node_type
        ].num_nodes

        if actual_count != expected_count:
            raise HeterogeneousGraphError(
                f"Node count mismatch for "
                f"{node_type}: expected "
                f"{expected_count}, got "
                f"{actual_count}"
            )

        node_ids = data[
            node_type
        ][NODE_ID_COLUMN]

        expected_ids = torch.arange(
            expected_count,
            dtype=torch.long,
        )

        if not torch.equal(
            node_ids,
            expected_ids,
        ):
            raise HeterogeneousGraphError(
                f"Invalid graph IDs for node type "
                f"{node_type}."
            )

    for edge_type, edge_count in (
        graph.edge_counts.items()
    ):
        edge_tuple = GRAPH_EDGE_TYPES[
            edge_type
        ]

        if edge_tuple not in data.edge_types:
            raise HeterogeneousGraphError(
                f"Missing graph edge type: "
                f"{edge_type}"
            )

        actual_edge_index = data[
            edge_tuple
        ].edge_index

        if actual_edge_index.dtype != torch.long:
            raise HeterogeneousGraphError(
                f"edge_index for {edge_type} "
                "must use torch.long."
            )

        if actual_edge_index.ndim != 2:
            raise HeterogeneousGraphError(
                f"edge_index for {edge_type} "
                "must be 2-dimensional."
            )

        if actual_edge_index.shape[0] != 2:
            raise HeterogeneousGraphError(
                f"edge_index for {edge_type} "
                "must have shape [2, E]."
            )

        if actual_edge_index.shape[1] != edge_count:
            raise HeterogeneousGraphError(
                f"Edge count mismatch for "
                f"{edge_type}: expected "
                f"{edge_count}, got "
                f"{actual_edge_index.shape[1]}"
            )

    if graph.total_nodes <= 0:
        raise HeterogeneousGraphError(
            "Graph must contain at least one node."
        )