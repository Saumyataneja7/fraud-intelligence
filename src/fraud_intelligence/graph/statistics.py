from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import pandas as pd
import torch

from fraud_intelligence.graph.contracts import (
    EDGE_TYPES,
    NODE_TYPES,
)
from fraud_intelligence.graph.heterogeneous import (
    HeterogeneousGraph,
)


class GraphStatisticsError(ValueError):
    """Raised when graph statistics cannot be computed."""


@dataclass(frozen=True)
class GraphStatistics:
    """Summary statistics for a heterogeneous fraud graph."""

    node_counts: dict[str, int]
    edge_counts: dict[str, int]

    total_nodes: int
    total_edges: int

    isolated_node_counts: dict[str, int]

    degree_statistics: pd.DataFrame
    connected_components: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable graph statistics."""

        return {
            "node_counts": dict(self.node_counts),
            "edge_counts": dict(self.edge_counts),
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "isolated_node_counts": dict(
                self.isolated_node_counts
            ),
            "degree_statistics": self.degree_statistics.to_dict(
                orient="records"
            ),
            "connected_components": self.connected_components,
        }


def _validate_graph(graph: HeterogeneousGraph) -> None:
    if not isinstance(graph, HeterogeneousGraph):
        raise GraphStatisticsError(
            "graph must be a HeterogeneousGraph."
        )

    if set(graph.node_counts) != set(NODE_TYPES):
        raise GraphStatisticsError(
            "Graph node metadata does not match canonical node types."
        )

    if set(graph.edge_counts) != set(EDGE_TYPES):
        raise GraphStatisticsError(
            "Graph edge metadata does not match canonical edge types."
        )


def _degree_statistics(
    graph: HeterogeneousGraph,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for node_type in NODE_TYPES:
        node_count = int(graph.node_counts[node_type])

        if node_count == 0:
            rows.append(
                {
                    "node_type": node_type,
                    "node_count": 0,
                    "min_degree": 0.0,
                    "max_degree": 0.0,
                    "mean_degree": 0.0,
                    "median_degree": 0.0,
                    "p95_degree": 0.0,
                }
            )
            continue

        degrees = torch.zeros(
            node_count,
            dtype=torch.long,
        )

        for edge_type in graph.data.edge_types:
            source_type, _, target_type = edge_type

            edge_index = graph.data[edge_type].edge_index

            if source_type == node_type:
                degrees.index_add_(
                    0,
                    edge_index[0],
                    torch.ones(
                        edge_index.shape[1],
                        dtype=torch.long,
                    ),
                )

            if target_type == node_type:
                degrees.index_add_(
                    0,
                    edge_index[1],
                    torch.ones(
                        edge_index.shape[1],
                        dtype=torch.long,
                    ),
                )

        values = degrees.to(torch.float64).numpy()

        rows.append(
            {
                "node_type": node_type,
                "node_count": node_count,
                "min_degree": float(values.min()),
                "max_degree": float(values.max()),
                "mean_degree": float(values.mean()),
                "median_degree": float(
                    pd.Series(values).median()
                ),
                "p95_degree": float(
                    pd.Series(values).quantile(0.95)
                ),
            }
        )

    return pd.DataFrame(rows)


def _isolated_node_counts(
    graph: HeterogeneousGraph,
) -> dict[str, int]:
    result: dict[str, int] = {}

    for node_type in NODE_TYPES:
        node_count = int(graph.node_counts[node_type])

        if node_count == 0:
            result[node_type] = 0
            continue

        degrees = torch.zeros(
            node_count,
            dtype=torch.long,
        )

        for edge_type in graph.data.edge_types:
            source_type, _, target_type = edge_type
            edge_index = graph.data[edge_type].edge_index

            if source_type == node_type:
                degrees.index_add_(
                    0,
                    edge_index[0],
                    torch.ones(
                        edge_index.shape[1],
                        dtype=torch.long,
                    ),
                )

            if target_type == node_type:
                degrees.index_add_(
                    0,
                    edge_index[1],
                    torch.ones(
                        edge_index.shape[1],
                        dtype=torch.long,
                    ),
                )

        result[node_type] = int((degrees == 0).sum().item())

    return result


def _build_entity_network(
    graph: HeterogeneousGraph,
) -> nx.Graph:
    """
    Build an undirected NetworkX projection across all node types.

    Node identities are represented as (node_type, node_id), preventing
    collisions between graph IDs belonging to different node types.
    """

    network = nx.Graph()

    for node_type in NODE_TYPES:
        count = int(graph.node_counts[node_type])

        for node_id in range(count):
            network.add_node(
                (node_type, node_id),
                node_type=node_type,
                node_id=node_id,
            )

    for edge_type in graph.data.edge_types:
        source_type, _, target_type = edge_type
        edge_index = graph.data[edge_type].edge_index

        for source_id, target_id in zip(
            edge_index[0].tolist(),
            edge_index[1].tolist(),
        ):
            network.add_edge(
                (source_type, source_id),
                (target_type, target_id),
            )

    return network


def _connected_component_statistics(
    graph: HeterogeneousGraph,
) -> dict[str, Any]:
    network = _build_entity_network(graph)

    if network.number_of_nodes() == 0:
        return {
            "component_count": 0,
            "largest_component_size": 0,
            "largest_component_fraction": 0.0,
            "size_distribution": [],
        }

    components = list(nx.connected_components(network))
    sizes = sorted(
        (len(component) for component in components),
        reverse=True,
    )

    return {
        "component_count": len(sizes),
        "largest_component_size": sizes[0],
        "largest_component_fraction": float(
            sizes[0] / network.number_of_nodes()
        ),
        "size_distribution": sizes,
    }


def calculate_graph_statistics(
    graph: HeterogeneousGraph,
) -> GraphStatistics:
    """Calculate structural statistics for a heterogeneous graph."""

    _validate_graph(graph)

    node_counts = {
        node_type: int(graph.node_counts[node_type])
        for node_type in NODE_TYPES
    }

    edge_counts = {
        edge_type: int(graph.edge_counts[edge_type])
        for edge_type in EDGE_TYPES
    }

    total_nodes = sum(node_counts.values())
    total_edges = sum(edge_counts.values())

    isolated = _isolated_node_counts(graph)
    degree_stats = _degree_statistics(graph)
    components = _connected_component_statistics(graph)

    return GraphStatistics(
        node_counts=node_counts,
        edge_counts=edge_counts,
        total_nodes=total_nodes,
        total_edges=total_edges,
        isolated_node_counts=isolated,
        degree_statistics=degree_stats,
        connected_components=components,
    )