import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from fraud_intelligence.explainability.contracts import (
    GNNExplanationContract,
    GNNExplanationContractError,
)
from fraud_intelligence.explainability.graph_neighborhood import (
    GNNNeighborhoodExplanation,
    GraphNeighborhoodError,
    NeighborhoodNode,
    SupportingEdge,
    extract_gnn_neighborhood,
    validate_gnn_neighborhood,
)


def make_graph() -> HeteroData:
    graph = HeteroData()

    graph["transaction"].num_nodes = 3
    graph["customer"].num_nodes = 2
    graph["device"].num_nodes = 2
    graph["ip"].num_nodes = 2
    graph["merchant"].num_nodes = 2

    graph[
        "transaction",
        "customer_transaction",
        "customer",
    ].edge_index = torch.tensor(
        [
            [0, 1],
            [0, 0],
        ],
        dtype=torch.long,
    )

    graph[
        "customer",
        "customer_device",
        "device",
    ].edge_index = torch.tensor(
        [
            [0],
            [0],
        ],
        dtype=torch.long,
    )

    graph[
        "customer",
        "customer_ip",
        "ip",
    ].edge_index = torch.tensor(
        [
            [0],
            [0],
        ],
        dtype=torch.long,
    )

    graph[
        "transaction",
        "transaction_merchant",
        "merchant",
    ].edge_index = torch.tensor(
        [
            [0],
            [0],
        ],
        dtype=torch.long,
    )

    return graph


def make_timestamps() -> pd.Series:
    return pd.Series(
        pd.to_datetime(
            [
                "2025-01-01 10:00:00+00:00",
                "2025-01-01 09:00:00+00:00",
                "2025-01-01 11:00:00+00:00",
            ],
            utc=True,
        )
    )


def test_neighborhood_contains_target():
    result = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=1,
    )

    assert (
        "transaction",
        0,
    ) in {
        (
            node.node_type,
            node.node_index,
        )
        for node in result.nodes
    }


def test_one_hop_contains_direct_neighbors():
    result = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=1,
    )

    nodes = {
        (
            node.node_type,
            node.node_index,
        )
        for node in result.nodes
    }

    assert (
        "customer",
        0,
    ) in nodes

    assert (
        "merchant",
        0,
    ) in nodes

    assert (
        "device",
        0,
    ) not in nodes


def test_two_hops_contains_indirect_neighbors():
    result = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=2,
    )

    nodes = {
        (
            node.node_type,
            node.node_index,
        )
        for node in result.nodes
    }

    assert (
        "customer",
        0,
    ) in nodes

    assert (
        "device",
        0,
    ) in nodes

    assert (
        "ip",
        0,
    ) in nodes


def test_future_transaction_is_excluded():
    result = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=2,
    )

    nodes = {
        (
            node.node_type,
            node.node_index,
        )
        for node in result.nodes
    }

    assert (
        "transaction",
        2,
    ) not in nodes


def test_past_transaction_is_allowed():
    result = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=2,
    )

    nodes = {
        (
            node.node_type,
            node.node_index,
        )
        for node in result.nodes
    }

    assert (
        "transaction",
        1,
    ) in nodes


def test_same_timestamp_transaction_is_excluded():
    graph = make_graph()

    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2025-01-01 10:00:00+00:00",
                "2025-01-01 10:00:00+00:00",
                "2025-01-01 11:00:00+00:00",
            ],
            utc=True,
        )
    )

    result = extract_gnn_neighborhood(
        graph=graph,
        transaction_node_index=0,
        transaction_timestamps=timestamps,
        hops=2,
    )

    nodes = {
        (
            node.node_type,
            node.node_index,
        )
        for node in result.nodes
    }

    assert (
        "transaction",
        1,
    ) not in nodes


def test_transaction_timestamp_is_timezone_aware():
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2025-01-01 10:00:00",
                "2025-01-01 09:00:00",
                "2025-01-01 11:00:00",
            ]
        )
    )

    with pytest.raises(GraphNeighborhoodError):
        extract_gnn_neighborhood(
            graph=make_graph(),
            transaction_node_index=0,
            transaction_timestamps=timestamps,
            hops=2,
        )


def test_invalid_transaction_index_rejected():
    with pytest.raises(GraphNeighborhoodError):
        extract_gnn_neighborhood(
            graph=make_graph(),
            transaction_node_index=99,
            transaction_timestamps=make_timestamps(),
        )


def test_negative_transaction_index_rejected():
    with pytest.raises(GraphNeighborhoodError):
        extract_gnn_neighborhood(
            graph=make_graph(),
            transaction_node_index=-1,
            transaction_timestamps=make_timestamps(),
        )


def test_negative_hops_rejected():
    with pytest.raises(GNNExplanationContractError):
        extract_gnn_neighborhood(
            graph=make_graph(),
            transaction_node_index=0,
            transaction_timestamps=make_timestamps(),
            hops=-1,
        )


def test_more_than_two_hops_rejected():
    with pytest.raises(GNNExplanationContractError):
        extract_gnn_neighborhood(
            graph=make_graph(),
            transaction_node_index=0,
            transaction_timestamps=make_timestamps(),
            hops=3,
        )


def test_node_hops_are_valid():
    result = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=2,
    )

    assert all(
        node.hop in {0, 1, 2}
        for node in result.nodes
    )


def test_edge_hops_are_valid():
    result = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=2,
    )

    assert all(
        edge.hop in {1, 2}
        for edge in result.edges
    )


def test_only_allowed_node_types_are_returned():
    result = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=2,
    )

    contract = GNNExplanationContract()

    assert set(result.node_types()).issubset(
        set(contract.allowed_node_types)
    )


def test_only_allowed_edge_types_are_returned():
    result = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=2,
    )

    contract = GNNExplanationContract()

    assert set(result.edge_types()).issubset(
        set(contract.allowed_edge_types)
    )


def test_nodes_are_deterministically_ordered():
    result_1 = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=2,
    )

    result_2 = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=2,
    )

    assert result_1.nodes == result_2.nodes
    assert result_1.edges == result_2.edges


def test_neighbor_nodes_excludes_target():
    result = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=2,
    )

    assert all(
        not (
            node.node_type == "transaction"
            and node.node_index == 0
        )
        for node in result.neighbor_nodes
    )


def test_supporting_edges_property():
    result = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=1,
    )

    assert (
        result.supporting_edges
        == result.edges
    )


def test_counts_are_consistent():
    result = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=2,
    )

    assert result.node_count == len(result.nodes)
    assert result.edge_count == len(result.edges)


def test_manual_valid_result():
    result = GNNNeighborhoodExplanation(
        transaction_node_index=0,
        transaction_timestamp=pd.Timestamp(
            "2025-01-01 10:00:00+00:00"
        ),
        nodes=(
            NeighborhoodNode(
                node_type="transaction",
                node_index=0,
                hop=0,
            ),
            NeighborhoodNode(
                node_type="merchant",
                node_index=0,
                hop=1,
            ),
        ),
        edges=(
            SupportingEdge(
                edge_type="transaction_merchant",
                source_type="transaction",
                source_index=0,
                target_type="merchant",
                target_index=0,
                hop=1,
            ),
        ),
    )

    validate_gnn_neighborhood(result)


def test_missing_target_rejected():
    result = GNNNeighborhoodExplanation(
        transaction_node_index=0,
        transaction_timestamp=pd.Timestamp(
            "2025-01-01 10:00:00+00:00"
        ),
        nodes=(
            NeighborhoodNode(
                node_type="merchant",
                node_index=0,
                hop=1,
            ),
        ),
        edges=(),
    )

    with pytest.raises(GraphNeighborhoodError):
        validate_gnn_neighborhood(result)


def test_duplicate_node_rejected():
    result = GNNNeighborhoodExplanation(
        transaction_node_index=0,
        transaction_timestamp=pd.Timestamp(
            "2025-01-01 10:00:00+00:00"
        ),
        nodes=(
            NeighborhoodNode(
                node_type="transaction",
                node_index=0,
                hop=0,
            ),
            NeighborhoodNode(
                node_type="transaction",
                node_index=0,
                hop=0,
            ),
        ),
        edges=(),
    )

    with pytest.raises(GraphNeighborhoodError):
        validate_gnn_neighborhood(result)


def test_unknown_node_type_rejected():
    result = GNNNeighborhoodExplanation(
        transaction_node_index=0,
        transaction_timestamp=pd.Timestamp(
            "2025-01-01 10:00:00+00:00"
        ),
        nodes=(
            NeighborhoodNode(
                node_type="unknown",
                node_index=0,
                hop=0,
            ),
        ),
        edges=(),
    )

    with pytest.raises(GNNExplanationContractError):
        validate_gnn_neighborhood(result)


def test_unknown_edge_type_rejected():
    result = GNNNeighborhoodExplanation(
        transaction_node_index=0,
        transaction_timestamp=pd.Timestamp(
            "2025-01-01 10:00:00+00:00"
        ),
        nodes=(
            NeighborhoodNode(
                node_type="transaction",
                node_index=0,
                hop=0,
            ),
            NeighborhoodNode(
                node_type="merchant",
                node_index=0,
                hop=1,
            ),
        ),
        edges=(
            SupportingEdge(
                edge_type="unknown_edge",
                source_type="transaction",
                source_index=0,
                target_type="merchant",
                target_index=0,
                hop=1,
            ),
        ),
    )

    with pytest.raises(GNNExplanationContractError):
        validate_gnn_neighborhood(result)


def test_edge_requires_existing_nodes():
    result = GNNNeighborhoodExplanation(
        transaction_node_index=0,
        transaction_timestamp=pd.Timestamp(
            "2025-01-01 10:00:00+00:00"
        ),
        nodes=(
            NeighborhoodNode(
                node_type="transaction",
                node_index=0,
                hop=0,
            ),
        ),
        edges=(
            SupportingEdge(
                edge_type="transaction_merchant",
                source_type="transaction",
                source_index=0,
                target_type="merchant",
                target_index=99,
                hop=1,
            ),
        ),
    )

    with pytest.raises(GraphNeighborhoodError):
        validate_gnn_neighborhood(result)


def test_zero_hop_returns_target_only():
    result = extract_gnn_neighborhood(
        graph=make_graph(),
        transaction_node_index=0,
        transaction_timestamps=make_timestamps(),
        hops=0,
    )

    assert result.nodes == (
        NeighborhoodNode(
            node_type="transaction",
            node_index=0,
            hop=0,
        ),
    )

    assert result.edges == ()