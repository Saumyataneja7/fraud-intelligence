from __future__ import annotations

import pandas as pd
import pytest
import torch

from fraud_intelligence.graph.edges import (
    EdgeTable,
    SOURCE_NODE_COLUMN,
    TARGET_NODE_COLUMN,
)
from fraud_intelligence.graph.heterogeneous import (
    GRAPH_EDGE_TYPES,
    build_heterogeneous_graph,
)
from fraud_intelligence.graph.validation import (
    GraphValidationError,
    validate_graph,
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
from fraud_intelligence.graph.contracts import (
    EDGE_TYPES,
)


def _build_nodes() -> dict:
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


def _empty_edge(
    edge_type: str,
) -> EdgeTable:
    source_type, _, target_type = (
        GRAPH_EDGE_TYPES[edge_type]
    )

    return EdgeTable(
        edge_type=edge_type,
        source_node_type=source_type,
        target_node_type=target_type,
        dataframe=pd.DataFrame(
            {
                SOURCE_NODE_COLUMN: pd.Series(
                    dtype="int64"
                ),
                TARGET_NODE_COLUMN: pd.Series(
                    dtype="int64"
                ),
            }
        ),
    )


def _build_edges(
    nodes: dict,
) -> dict:
    from fraud_intelligence.graph.edges import (
        build_account_card_edges,
        build_customer_account_edges,
    )

    edges = {
        edge_type: _empty_edge(edge_type)
        for edge_type in EDGE_TYPES
    }

    edges["customer_account"] = (
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

    edges["account_card"] = (
        build_account_card_edges(
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
    )

    return edges


def _build_graph():
    nodes = _build_nodes()
    edges = _build_edges(nodes)

    return build_heterogeneous_graph(
        nodes,
        edges,
    )


def test_valid_graph_passes_validation() -> None:
    graph = _build_graph()

    result = validate_graph(graph)

    assert result.valid is True
    assert result.node_count == 14
    assert result.edge_count == 4


def test_validation_reports_node_types() -> None:
    graph = _build_graph()

    result = validate_graph(graph)

    assert set(result.node_types) == {
        "customer",
        "account",
        "card",
        "transaction",
        "merchant",
        "device",
        "ip",
    }


def test_validation_reports_edge_types() -> None:
    graph = _build_graph()

    result = validate_graph(graph)

    assert set(result.edge_types) == set(
        GRAPH_EDGE_TYPES.values()
    )


def test_missing_node_id_is_rejected() -> None:
    graph = _build_graph()

    del graph.data["customer"].node_id

    with pytest.raises(
        GraphValidationError,
        match="Missing node_id",
    ):
        validate_graph(graph)


def test_wrong_node_id_dtype_is_rejected() -> None:
    graph = _build_graph()

    graph.data["customer"].node_id = (
        graph.data["customer"]
        .node_id
        .float()
    )

    with pytest.raises(
        GraphValidationError,
        match="must use torch.long",
    ):
        validate_graph(graph)


def test_non_contiguous_node_ids_are_rejected() -> None:
    graph = _build_graph()

    graph.data["customer"].node_id = (
        torch.tensor(
            [0, 2],
            dtype=torch.long,
        )
    )

    with pytest.raises(
        GraphValidationError,
        match="contiguous and zero-based",
    ):
        validate_graph(graph)


def test_wrong_edge_index_dtype_is_rejected() -> None:
    graph = _build_graph()

    edge_type = (
        "customer",
        "customer_account",
        "account",
    )

    graph.data[
        edge_type
    ].edge_index = (
        graph.data[
            edge_type
        ].edge_index.float()
    )

    with pytest.raises(
        GraphValidationError,
        match="edge_index.*must use torch.long",
    ):
        validate_graph(graph)


def test_wrong_edge_index_shape_is_rejected() -> None:
    graph = _build_graph()

    edge_type = (
        "customer",
        "customer_account",
        "account",
    )

    graph.data[
        edge_type
    ].edge_index = torch.tensor(
        [[0], [0], [1]],
        dtype=torch.long,
    )

    with pytest.raises(
        GraphValidationError,
        match="shape \\[2, E\\]",
    ):
        validate_graph(graph)


def test_out_of_range_source_is_rejected() -> None:
    graph = _build_graph()

    edge_type = (
        "customer",
        "customer_account",
        "account",
    )

    graph.data[
        edge_type
    ].edge_index = torch.tensor(
        [
            [999],
            [0],
        ],
        dtype=torch.long,
    )

    graph.edge_counts[
        "customer_account"
    ] = 1

    with pytest.raises(
        GraphValidationError,
        match="Source node IDs",
    ):
        validate_graph(graph)


def test_out_of_range_target_is_rejected() -> None:
    graph = _build_graph()

    edge_type = (
        "customer",
        "customer_account",
        "account",
    )

    graph.data[
        edge_type
    ].edge_index = torch.tensor(
        [
            [0],
            [999],
        ],
        dtype=torch.long,
    )

    graph.edge_counts[
        "customer_account"
    ] = 1

    with pytest.raises(
        GraphValidationError,
        match="Target node IDs",
    ):
        validate_graph(graph)


def test_duplicate_edges_are_rejected() -> None:
    graph = _build_graph()

    edge_type = (
        "customer",
        "customer_account",
        "account",
    )

    graph.data[
        edge_type
    ].edge_index = torch.tensor(
        [
            [0, 0],
            [0, 0],
        ],
        dtype=torch.long,
    )

    graph.edge_counts[
        "customer_account"
    ] = 2

    with pytest.raises(
        GraphValidationError,
        match="Duplicate edges",
    ):
        validate_graph(graph)


def test_edge_count_mismatch_is_rejected() -> None:
    graph = _build_graph()

    graph.edge_counts[
        "customer_account"
    ] = 999

    with pytest.raises(
        GraphValidationError,
        match="Edge count mismatch",
    ):
        validate_graph(graph)


def test_node_count_metadata_mismatch_is_rejected() -> None:
    graph = _build_graph()

    graph.node_counts[
        "customer"
    ] = 999

    with pytest.raises(
        GraphValidationError,
        match="Node ID count mismatch",
    ):
        validate_graph(graph)


def test_invalid_graph_object_is_rejected() -> None:
    with pytest.raises(
        GraphValidationError,
        match="must be a HeterogeneousGraph",
    ):
        validate_graph(None)


def test_empty_edge_index_is_valid() -> None:
    graph = _build_graph()

    for edge_type in EDGE_TYPES:
        graph_edge_type = GRAPH_EDGE_TYPES[
            edge_type
        ]

        graph.data[
            graph_edge_type
        ].edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        graph.edge_counts[
            edge_type
        ] = 0

    result = validate_graph(graph)

    assert result.valid is True
    assert result.edge_count == 0