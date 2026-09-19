from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from fraud_intelligence.graph.contracts import (
    EDGE_TYPES,
    NODE_TYPES,
)
from fraud_intelligence.graph.edges import (
    build_account_card_edges,
    build_customer_account_edges,
)
from fraud_intelligence.graph.heterogeneous import (
    GRAPH_EDGE_TYPES,
    HeterogeneousGraph,
    HeterogeneousGraphError,
    build_heterogeneous_graph,
    validate_heterogeneous_graph,
)
from fraud_intelligence.graph.nodes import (
    build_account_nodes,
    build_card_nodes,
    build_customer_nodes,
    build_device_nodes,
    build_ip_nodes,
    build_merchant_nodes,
    build_transaction_nodes,
)


def _build_test_nodes() -> dict:
    return {
        "customer": build_customer_nodes(
            pd.DataFrame(
                {
                    "customer_id": [
                        "C001",
                        "C002",
                    ]
                }
            )
        ),
        "account": build_account_nodes(
            pd.DataFrame(
                {
                    "account_id": [
                        "A001",
                        "A002",
                    ]
                }
            )
        ),
        "card": build_card_nodes(
            pd.DataFrame(
                {
                    "card_id": [
                        "CARD001",
                        "CARD002",
                    ]
                }
            )
        ),
        "transaction": build_transaction_nodes(
            pd.DataFrame(
                {
                    "transaction_id": [
                        "T001",
                        "T002",
                    ],
                    "timestamp": pd.to_datetime(
                        [
                            "2025-01-01",
                            "2025-01-02",
                        ],
                        utc=True,
                    ),
                    "amount": [
                        10.0,
                        20.0,
                    ],
                    "currency": [
                        "USD",
                        "USD",
                    ],
                    "payment_method": [
                        "card",
                        "wallet",
                    ],
                    "transaction_type": [
                        "purchase",
                        "payment",
                    ],
                    "is_fraud": [
                        0,
                        1,
                    ],
                    "fraud_scenario": [
                        None,
                        "ACCOUNT_TAKEOVER",
                    ],
                }
            )
        ),
        "merchant": build_merchant_nodes(
            pd.DataFrame(
                {
                    "merchant_id": [
                        "M001",
                        "M002",
                    ]
                }
            )
        ),
        "device": build_device_nodes(
            pd.DataFrame(
                {
                    "device_id": [
                        "D001",
                        "D002",
                    ]
                }
            )
        ),
        "ip": build_ip_nodes(
            pd.DataFrame(
                {
                    "ip_id": [
                        "IP001",
                        "IP002",
                    ]
                }
            )
        ),
    }


def _build_test_edges(
    nodes: dict,
) -> dict:
    customer_account = (
        build_customer_account_edges(
            pd.DataFrame(
                {
                    "customer_id": [
                        "C001",
                        "C002",
                    ],
                    "account_id": [
                        "A001",
                        "A002",
                    ],
                }
            ),
            nodes["customer"],
            nodes["account"],
        )
    )

    account_card = build_account_card_edges(
        pd.DataFrame(
            {
                "account_id": [
                    "A001",
                    "A002",
                ],
                "card_id": [
                    "CARD001",
                    "CARD002",
                ],
            }
        ),
        nodes["account"],
        nodes["card"],
    )

    empty_edges = {}

    for edge_type in EDGE_TYPES:
        if edge_type == "customer_account":
            empty_edges[edge_type] = (
                customer_account
            )
        elif edge_type == "account_card":
            empty_edges[edge_type] = (
                account_card
            )
        else:
            source_type, _, target_type = (
                GRAPH_EDGE_TYPES[edge_type]
            )

            from fraud_intelligence.graph.edges import (
                EdgeTable,
            )

            empty_edges[edge_type] = EdgeTable(
                edge_type=edge_type,
                source_node_type=source_type,
                target_node_type=target_type,
                dataframe=pd.DataFrame(
                    {
                        "source_node_id": pd.Series(
                            dtype="int64"
                        ),
                        "target_node_id": pd.Series(
                            dtype="int64"
                        ),
                    }
                ),
            )

    return empty_edges


def test_build_heterogeneous_graph() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    graph = build_heterogeneous_graph(
        nodes,
        edges,
    )

    assert isinstance(
        graph.data,
        HeteroData,
    )


def test_all_node_types_exist() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    graph = build_heterogeneous_graph(
        nodes,
        edges,
    )

    assert set(
        graph.data.node_types
    ) == set(NODE_TYPES)


def test_all_edge_types_exist() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    graph = build_heterogeneous_graph(
        nodes,
        edges,
    )

    assert set(
        graph.data.edge_types
    ) == set(
        GRAPH_EDGE_TYPES.values()
    )


def test_node_counts_are_preserved() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    graph = build_heterogeneous_graph(
        nodes,
        edges,
    )

    for node_type in NODE_TYPES:
        assert (
            graph.data[
                node_type
            ].num_nodes
            == nodes[node_type].node_count
        )


def test_edge_index_shape() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    graph = build_heterogeneous_graph(
        nodes,
        edges,
    )

    edge_index = graph.data[
        (
            "customer",
            "customer_account",
            "account",
        )
    ].edge_index

    assert edge_index.shape == (2, 2)


def test_edge_index_dtype() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    graph = build_heterogeneous_graph(
        nodes,
        edges,
    )

    for edge_tuple in graph.data.edge_types:
        assert (
            graph.data[
                edge_tuple
            ].edge_index.dtype
            == torch.long
        )


def test_graph_ids_are_zero_based() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    graph = build_heterogeneous_graph(
        nodes,
        edges,
    )

    for node_type in NODE_TYPES:
        ids = graph.data[
            node_type
        ].node_id

        assert ids.tolist() == [0, 1]


def test_edge_counts_are_preserved() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    graph = build_heterogeneous_graph(
        nodes,
        edges,
    )

    for edge_type in EDGE_TYPES:
        assert (
            graph.edge_counts[edge_type]
            == edges[edge_type].edge_count
        )


def test_total_node_count() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    graph = build_heterogeneous_graph(
        nodes,
        edges,
    )

    assert graph.total_nodes == 14


def test_total_edge_count() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    graph = build_heterogeneous_graph(
        nodes,
        edges,
    )

    assert graph.total_edges == 4


def test_graph_validation_passes() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    graph = build_heterogeneous_graph(
        nodes,
        edges,
    )

    validate_heterogeneous_graph(
        graph
    )


def test_missing_node_type_is_rejected() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    del nodes["device"]

    with pytest.raises(
        HeterogeneousGraphError,
        match="Missing node types",
    ):
        build_heterogeneous_graph(
            nodes,
            edges,
        )


def test_missing_edge_type_is_rejected() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    del edges["customer_ip"]

    with pytest.raises(
        HeterogeneousGraphError,
        match="Missing edge types",
    ):
        build_heterogeneous_graph(
            nodes,
            edges,
        )


def test_invalid_edge_endpoint_is_rejected() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    from fraud_intelligence.graph.edges import (
        EdgeTable,
    )

    edges["customer_account"] = EdgeTable(
        edge_type="customer_account",
        source_node_type="customer",
        target_node_type="account",
        dataframe=pd.DataFrame(
            {
                "source_node_id": [
                    999
                ],
                "target_node_id": [
                    0
                ],
            }
        ),
    )

    with pytest.raises(
        HeterogeneousGraphError,
        match="outside the valid node range",
    ):
        build_heterogeneous_graph(
            nodes,
            edges,
        )


def test_graph_data_is_heterogeneous() -> None:
    nodes = _build_test_nodes()
    edges = _build_test_edges(nodes)

    graph = build_heterogeneous_graph(
        nodes,
        edges,
    )

    assert isinstance(
        graph.data,
        HeteroData,
    )

    assert len(graph.data.node_types) > 1
    assert len(graph.data.edge_types) > 1


def test_invalid_graph_object_is_rejected() -> None:
    with pytest.raises(
        HeterogeneousGraphError,
        match="must be a HeterogeneousGraph",
    ):
        validate_heterogeneous_graph(
            "not a graph"
        )