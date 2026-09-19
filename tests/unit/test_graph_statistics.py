from __future__ import annotations

import pandas as pd
import pytest

from fraud_intelligence.graph.contracts import (
    EDGE_TYPES,
    NODE_TYPES,
)
from fraud_intelligence.graph.edges import EdgeTable
from fraud_intelligence.graph.heterogeneous import (
    build_heterogeneous_graph,
)
from fraud_intelligence.graph.nodes import NodeTable
from fraud_intelligence.graph.statistics import (
    GraphStatisticsError,
    calculate_graph_statistics,
)


def _build_graph():
    node_counts = {
        "customer": 2,
        "account": 2,
        "card": 2,
        "transaction": 2,
        "merchant": 2,
        "device": 2,
        "ip": 2,
    }

    source_columns = {
        "customer": "customer_id",
        "account": "account_id",
        "card": "card_id",
        "transaction": "transaction_id",
        "merchant": "merchant_id",
        "device": "device_id",
        "ip": "ip_id",
    }

    node_tables = {}

    for node_type in NODE_TYPES:
        source_column = source_columns[node_type]

        node_tables[node_type] = NodeTable(
            node_type=node_type,
            dataframe=pd.DataFrame(
                {
                    source_column: [
                        f"{node_type}_0",
                        f"{node_type}_1",
                    ],
                    "node_id": [0, 1],
                }
            ),
            source_id_column=source_column,
        )

    endpoints = {
        "customer_account": (
            "customer",
            "account",
        ),
        "account_card": (
            "account",
            "card",
        ),
        "customer_transaction": (
            "customer",
            "transaction",
        ),
        "account_transaction": (
            "account",
            "transaction",
        ),
        "card_transaction": (
            "card",
            "transaction",
        ),
        "customer_device": (
            "customer",
            "device",
        ),
        "customer_ip": (
            "customer",
            "ip",
        ),
        "customer_merchant": (
            "customer",
            "merchant",
        ),
        "transaction_merchant": (
            "transaction",
            "merchant",
        ),
        "transaction_device": (
            "transaction",
            "device",
        ),
        "transaction_ip": (
            "transaction",
            "ip",
        ),
    }

    edge_tables = {}

    for edge_type in EDGE_TYPES:
        source_type, target_type = endpoints[edge_type]

        edge_tables[edge_type] = EdgeTable(
            edge_type=edge_type,
            source_node_type=source_type,
            target_node_type=target_type,
            dataframe=pd.DataFrame(
                {
                    "source_node_id": [0],
                    "target_node_id": [0],
                }
            ),
        )

    return build_heterogeneous_graph(
        node_tables=node_tables,
        edge_tables=edge_tables,
    )


def test_calculate_graph_statistics() -> None:
    graph = _build_graph()

    statistics = calculate_graph_statistics(graph)

    assert statistics.total_nodes == 14
    assert statistics.total_edges == 11


def test_node_counts_match_graph() -> None:
    graph = _build_graph()

    statistics = calculate_graph_statistics(graph)

    assert statistics.node_counts == {
        "customer": 2,
        "account": 2,
        "card": 2,
        "transaction": 2,
        "merchant": 2,
        "device": 2,
        "ip": 2,
    }


def test_edge_counts_match_graph() -> None:
    graph = _build_graph()

    statistics = calculate_graph_statistics(graph)

    for edge_type in EDGE_TYPES:
        assert statistics.edge_counts[edge_type] == 1


def test_degree_statistics_contains_all_node_types() -> None:
    graph = _build_graph()

    statistics = calculate_graph_statistics(graph)

    assert set(
        statistics.degree_statistics["node_type"]
    ) == set(NODE_TYPES)


def test_degree_statistics_have_expected_columns() -> None:
    graph = _build_graph()

    statistics = calculate_graph_statistics(graph)

    assert list(statistics.degree_statistics.columns) == [
        "node_type",
        "node_count",
        "min_degree",
        "max_degree",
        "mean_degree",
        "median_degree",
        "p95_degree",
    ]


def test_degree_statistics_are_non_negative() -> None:
    graph = _build_graph()

    statistics = calculate_graph_statistics(graph)

    numeric_columns = [
        "min_degree",
        "max_degree",
        "mean_degree",
        "median_degree",
        "p95_degree",
    ]

    assert (
        statistics.degree_statistics[numeric_columns]
        >= 0
    ).all().all()


def test_isolated_nodes_are_detected() -> None:
    graph = _build_graph()

    statistics = calculate_graph_statistics(graph)

    # Node 1 of every type has no edges in this test graph.
    assert statistics.isolated_node_counts["customer"] == 1
    assert statistics.isolated_node_counts["account"] == 1
    assert statistics.isolated_node_counts["card"] == 1
    assert statistics.isolated_node_counts["transaction"] == 1
    assert statistics.isolated_node_counts["merchant"] == 1
    assert statistics.isolated_node_counts["device"] == 1
    assert statistics.isolated_node_counts["ip"] == 1


def test_connected_components_are_calculated() -> None:
    graph = _build_graph()

    statistics = calculate_graph_statistics(graph)

    components = statistics.connected_components

    assert components["component_count"] == 8
    assert components["largest_component_size"] == 7


def test_largest_component_fraction_is_valid() -> None:
    graph = _build_graph()

    statistics = calculate_graph_statistics(graph)

    fraction = statistics.connected_components[
        "largest_component_fraction"
    ]

    assert 0.0 < fraction <= 1.0


def test_size_distribution_is_descending() -> None:
    graph = _build_graph()

    statistics = calculate_graph_statistics(graph)

    sizes = statistics.connected_components[
        "size_distribution"
    ]

    assert sizes == sorted(sizes, reverse=True)


def test_to_dict_returns_expected_structure() -> None:
    graph = _build_graph()

    statistics = calculate_graph_statistics(graph)
    result = statistics.to_dict()

    assert isinstance(result, dict)
    assert isinstance(result["node_counts"], dict)
    assert isinstance(result["edge_counts"], dict)
    assert isinstance(result["degree_statistics"], list)
    assert isinstance(result["connected_components"], dict)


def test_total_nodes_are_sum_of_node_counts() -> None:
    graph = _build_graph()

    statistics = calculate_graph_statistics(graph)

    assert statistics.total_nodes == sum(
        statistics.node_counts.values()
    )


def test_total_edges_are_sum_of_edge_counts() -> None:
    graph = _build_graph()

    statistics = calculate_graph_statistics(graph)

    assert statistics.total_edges == sum(
        statistics.edge_counts.values()
    )


def test_invalid_graph_type_is_rejected() -> None:
    with pytest.raises(
        GraphStatisticsError,
        match="HeterogeneousGraph",
    ):
        calculate_graph_statistics("not-a-graph")


def test_statistics_are_deterministic() -> None:
    graph = _build_graph()

    first = calculate_graph_statistics(graph)
    second = calculate_graph_statistics(graph)

    assert first.node_counts == second.node_counts
    assert first.edge_counts == second.edge_counts
    assert (
        first.isolated_node_counts
        == second.isolated_node_counts
    )
    assert (
        first.connected_components
        == second.connected_components
    )

    pd.testing.assert_frame_equal(
        first.degree_statistics,
        second.degree_statistics,
    )