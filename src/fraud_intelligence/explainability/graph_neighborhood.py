from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from fraud_intelligence.explainability.contracts import (
    GNNExplanationContract,
    GNNExplanationContractError,
)


class GraphNeighborhoodError(ValueError):
    """Raised when graph neighborhood extraction fails."""


@dataclass(frozen=True)
class NeighborhoodNode:
    """One node included in a transaction explanation."""

    node_type: str
    node_index: int
    hop: int


@dataclass(frozen=True)
class SupportingEdge:
    """One graph edge supporting the explanation."""

    edge_type: str
    source_type: str
    source_index: int
    target_type: str
    target_index: int
    hop: int


@dataclass(frozen=True)
class GNNNeighborhoodExplanation:
    """Historical graph neighborhood around one transaction."""

    transaction_node_index: int
    transaction_timestamp: pd.Timestamp
    nodes: tuple[NeighborhoodNode, ...]
    edges: tuple[SupportingEdge, ...]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def neighbor_nodes(
        self,
    ) -> tuple[NeighborhoodNode, ...]:
        return tuple(
            node
            for node in self.nodes
            if not (
                node.node_type == "transaction"
                and node.node_index
                == self.transaction_node_index
            )
        )

    @property
    def supporting_edges(
        self,
    ) -> tuple[SupportingEdge, ...]:
        return self.edges

    def node_types(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    node.node_type
                    for node in self.nodes
                }
            )
        )

    def edge_types(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    edge.edge_type
                    for edge in self.edges
                }
            )
        )


def _validate_transaction_index(
    graph: HeteroData,
    transaction_node_index: int,
) -> None:
    if not isinstance(
        transaction_node_index,
        int,
    ):
        raise GraphNeighborhoodError(
            "transaction_node_index must be an integer."
        )

    if transaction_node_index < 0:
        raise GraphNeighborhoodError(
            "transaction_node_index cannot be negative."
        )

    if "transaction" not in graph.node_types:
        raise GraphNeighborhoodError(
            "Graph does not contain transaction nodes."
        )

    count = graph["transaction"].num_nodes

    if transaction_node_index >= count:
        raise GraphNeighborhoodError(
            "transaction_node_index is outside "
            "the transaction node range."
        )


def _normalize_timestamp(
    timestamp: object,
) -> pd.Timestamp:
    result = pd.Timestamp(timestamp)

    if result.tzinfo is None:
        raise GraphNeighborhoodError(
            "Target transaction timestamp must be timezone-aware."
        )

    return result


def _edge_triplets(
    graph: HeteroData,
    edge_type: tuple[str, str, str],
) -> list[tuple[int, int]]:
    source_type, _, target_type = edge_type

    store = graph[edge_type]

    if "edge_index" not in store:
        return []

    edge_index = store.edge_index

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise GraphNeighborhoodError(
            f"Invalid edge_index for {edge_type}."
        )

    result = []

    for source, target in zip(
        edge_index[0].tolist(),
        edge_index[1].tolist(),
    ):
        result.append(
            (
                int(source),
                int(target),
            )
        )

    return result


def _build_adjacency(
    graph: HeteroData,
    allowed_edge_types: Iterable[str],
) -> dict[
    tuple[str, int],
    list[
        tuple[
            str,
            str,
            int,
            int,
        ]
    ],
]:
    adjacency: dict[
        tuple[str, int],
        list[
            tuple[
                str,
                str,
                int,
                int,
            ]
        ],
    ] = {}

    allowed = set(allowed_edge_types)

    for edge_type in graph.edge_types:
        source_type, relation, target_type = edge_type

        if relation not in allowed:
            continue

        for source, target in _edge_triplets(
            graph,
            edge_type,
        ):
            adjacency.setdefault(
                (source_type, source),
                [],
            ).append(
                (
                    relation,
                    target_type,
                    target,
                    0,
                )
            )

            adjacency.setdefault(
                (target_type, target),
                [],
            ).append(
                (
                    relation,
                    source_type,
                    source,
                    0,
                )
            )

    for key in adjacency:
        adjacency[key].sort(
            key=lambda value: (
                value[0],
                value[1],
                value[2],
            )
        )

    return adjacency


def _validate_allowed_graph(
    graph: HeteroData,
    contract: GNNExplanationContract,
) -> None:
    node_types = tuple(
        sorted(graph.node_types)
    )

    contract.validate_node_types(
        node_types
    )

    edge_relations = tuple(
        sorted(
            {
                edge_type[1]
                for edge_type in graph.edge_types
            }
        )
    )

    contract.validate_edge_types(
        edge_relations
    )


def _get_transaction_timestamp(
    transaction_timestamps: pd.Series,
    transaction_node_index: int,
) -> pd.Timestamp:
    if not isinstance(
        transaction_timestamps,
        pd.Series,
    ):
        raise GraphNeighborhoodError(
            "transaction_timestamps must be a pandas Series."
        )

    if len(transaction_timestamps) <= transaction_node_index:
        raise GraphNeighborhoodError(
            "transaction_timestamps does not cover "
            "the requested transaction node."
        )

    timestamp = transaction_timestamps.iloc[
        transaction_node_index
    ]

    return _normalize_timestamp(timestamp)


def _collect_neighborhood(
    *,
    graph: HeteroData,
    transaction_node_index: int,
    max_hops: int,
    adjacency: dict[
        tuple[str, int],
        list[
            tuple[
                str,
                str,
                int,
                int,
            ]
        ],
    ],
) -> tuple[
    tuple[NeighborhoodNode, ...],
    tuple[SupportingEdge, ...],
]:
    start = (
        "transaction",
        transaction_node_index,
    )

    visited: dict[
        tuple[str, int],
        int,
    ] = {
        start: 0
    }

    queue: list[
        tuple[
            str,
            int,
            int,
        ]
    ] = [
        (
            "transaction",
            transaction_node_index,
            0,
        )
    ]

    collected_edges: set[
        tuple[
            str,
            str,
            int,
            str,
            int,
            int,
        ]
    ] = set()

    cursor = 0

    while cursor < len(queue):
        node_type, node_index, hop = queue[cursor]
        cursor += 1

        if hop >= max_hops:
            continue

        current = (
            node_type,
            node_index,
        )

        for (
            relation,
            neighbor_type,
            neighbor_index,
            _,
        ) in adjacency.get(
            current,
            [],
        ):
            neighbor = (
                neighbor_type,
                neighbor_index,
            )

            next_hop = hop + 1

            collected_edges.add(
                (
                    relation,
                    node_type,
                    node_index,
                    neighbor_type,
                    neighbor_index,
                    next_hop,
                )
            )

            previous_hop = visited.get(
                neighbor
            )

            if (
                previous_hop is None
                or next_hop < previous_hop
            ):
                visited[neighbor] = next_hop

                queue.append(
                    (
                        neighbor_type,
                        neighbor_index,
                        next_hop,
                    )
                )

    nodes = tuple(
        NeighborhoodNode(
            node_type=node_type,
            node_index=node_index,
            hop=hop,
        )
        for (
            node_type,
            node_index,
        ), hop in sorted(
            visited.items(),
            key=lambda item: (
                item[1],
                item[0][0],
                item[0][1],
            ),
        )
    )

    edges = tuple(
        SupportingEdge(
            edge_type=relation,
            source_type=source_type,
            source_index=source_index,
            target_type=target_type,
            target_index=target_index,
            hop=hop,
        )
        for (
            relation,
            source_type,
            source_index,
            target_type,
            target_index,
            hop,
        ) in sorted(
            collected_edges,
            key=lambda item: (
                item[5],
                item[0],
                item[1],
                item[2],
                item[3],
                item[4],
            ),
        )
    )

    return nodes, edges


def _filter_future_transaction_context(
    *,
    neighborhood: GNNNeighborhoodExplanation,
    transaction_timestamps: pd.Series,
) -> GNNNeighborhoodExplanation:
    target_timestamp = (
        neighborhood.transaction_timestamp
    )

    valid_nodes = []

    for node in neighborhood.nodes:
        if node.node_type != "transaction":
            valid_nodes.append(node)
            continue

        node_timestamp = _get_transaction_timestamp(
            transaction_timestamps,
            node.node_index,
        )

        if (
            node.node_index
            == neighborhood.transaction_node_index
        ):
            valid_nodes.append(node)
            continue

        if node_timestamp < target_timestamp:
            valid_nodes.append(node)

    valid_node_keys = {
        (
            node.node_type,
            node.node_index,
        )
        for node in valid_nodes
    }

    valid_edges = []

    for edge in neighborhood.edges:
        source_key = (
            edge.source_type,
            edge.source_index,
        )

        target_key = (
            edge.target_type,
            edge.target_index,
        )

        if (
            source_key in valid_node_keys
            and target_key in valid_node_keys
        ):
            valid_edges.append(edge)

    return GNNNeighborhoodExplanation(
        transaction_node_index=(
            neighborhood.transaction_node_index
        ),
        transaction_timestamp=(
            neighborhood.transaction_timestamp
        ),
        nodes=tuple(valid_nodes),
        edges=tuple(valid_edges),
    )


def validate_gnn_neighborhood(
    result: GNNNeighborhoodExplanation,
    contract: GNNExplanationContract | None = None,
) -> None:
    if contract is None:
        contract = GNNExplanationContract()

    if not isinstance(
        result,
        GNNNeighborhoodExplanation,
    ):
        raise GraphNeighborhoodError(
            "result must be a GNNNeighborhoodExplanation."
        )

    if result.transaction_node_index < 0:
        raise GraphNeighborhoodError(
            "Transaction node index cannot be negative."
        )

    if result.transaction_timestamp.tzinfo is None:
        raise GraphNeighborhoodError(
            "Transaction timestamp must be timezone-aware."
        )

    node_keys = set()

    for node in result.nodes:
        if node.node_type not in contract.allowed_node_types:
            raise GNNExplanationContractError(
                f"Unknown node type: {node.node_type}."
            )

        if node.node_index < 0:
            raise GraphNeighborhoodError(
                "Node index cannot be negative."
            )

        contract.validate_hops(node.hop)

        key = (
            node.node_type,
            node.node_index,
        )

        if key in node_keys:
            raise GraphNeighborhoodError(
                "Duplicate node in neighborhood."
            )

        node_keys.add(key)

    target_key = (
        "transaction",
        result.transaction_node_index,
    )

    if target_key not in node_keys:
        raise GraphNeighborhoodError(
            "Target transaction must be present "
            "in the neighborhood."
        )

    for edge in result.edges:
        if edge.edge_type not in contract.allowed_edge_types:
            raise GNNExplanationContractError(
                f"Unknown edge type: {edge.edge_type}."
            )

        if (
            edge.source_type,
            edge.source_index,
        ) not in node_keys:
            raise GraphNeighborhoodError(
                "Edge source is missing from neighborhood."
            )

        if (
            edge.target_type,
            edge.target_index,
        ) not in node_keys:
            raise GraphNeighborhoodError(
                "Edge target is missing from neighborhood."
            )

        contract.validate_hops(edge.hop)


def extract_gnn_neighborhood(
    *,
    graph: HeteroData,
    transaction_node_index: int,
    transaction_timestamps: pd.Series,
    hops: int = 2,
    contract: GNNExplanationContract | None = None,
) -> GNNNeighborhoodExplanation:
    """
    Extract a temporally safe graph neighborhood around
    one transaction.

    Graph topology comes directly from the frozen Phase 6
    heterogeneous graph.

    Transaction timestamps are used only to remove transaction
    nodes that are not strictly earlier than the target.
    """
    if contract is None:
        contract = GNNExplanationContract()

    contract.validate_hops(hops)

    _validate_transaction_index(
        graph,
        transaction_node_index,
    )

    _validate_allowed_graph(
        graph,
        contract,
    )

    target_timestamp = _get_transaction_timestamp(
        transaction_timestamps,
        transaction_node_index,
    )

    adjacency = _build_adjacency(
        graph,
        contract.allowed_edge_types,
    )

    nodes, edges = _collect_neighborhood(
        graph=graph,
        transaction_node_index=(
            transaction_node_index
        ),
        max_hops=hops,
        adjacency=adjacency,
    )

    result = GNNNeighborhoodExplanation(
        transaction_node_index=(
            transaction_node_index
        ),
        transaction_timestamp=target_timestamp,
        nodes=nodes,
        edges=edges,
    )

    result = _filter_future_transaction_context(
        neighborhood=result,
        transaction_timestamps=transaction_timestamps,
    )

    validate_gnn_neighborhood(
        result,
        contract,
    )

    return result