from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from fraud_intelligence.graph.contracts import (
    EDGE_TYPES,
    NODE_TYPES,
)
from fraud_intelligence.graph.edges import EdgeTable
from fraud_intelligence.graph.heterogeneous import (
    build_heterogeneous_graph,
)
from fraud_intelligence.graph.materialization import (
    GRAPH_ARTIFACT_VERSION,
    GraphMaterializationError,
    build_graph_metadata,
    load_graph_artifact,
    save_graph_artifact,
)
from fraud_intelligence.graph.nodes import NodeTable


def _build_graph():
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


def test_build_graph_metadata() -> None:
    graph = _build_graph()

    metadata = build_graph_metadata(graph)

    assert metadata["artifact"]["version"] == (
        GRAPH_ARTIFACT_VERSION
    )

    assert metadata["schema"]["node_types"] == list(
        NODE_TYPES
    )

    assert metadata["schema"]["edge_types"] == list(
        EDGE_TYPES
    )

    assert metadata["graph"]["total_nodes"] == 14
    assert metadata["graph"]["total_edges"] == 11


def test_metadata_contains_topology_hash() -> None:
    graph = _build_graph()

    metadata = build_graph_metadata(graph)

    topology_hash = metadata["topology"]["sha256"]

    assert isinstance(topology_hash, str)
    assert len(topology_hash) == 64


def test_topology_hash_is_deterministic() -> None:
    graph = _build_graph()

    first = build_graph_metadata(graph)
    second = build_graph_metadata(graph)

    assert (
        first["topology"]["sha256"]
        == second["topology"]["sha256"]
    )


def test_save_graph_artifact_creates_files(
    tmp_path,
) -> None:
    graph = _build_graph()

    artifact = save_graph_artifact(
        graph,
        tmp_path,
    )

    assert artifact.graph_path.exists()
    assert artifact.metadata_path.exists()


def test_saved_graph_is_heterodata(
    tmp_path,
) -> None:
    graph = _build_graph()

    artifact = save_graph_artifact(
        graph,
        tmp_path,
    )

    loaded = torch.load(
        artifact.graph_path,
        map_location="cpu",
        weights_only=False,
    )

    assert loaded.__class__.__name__ == "HeteroData"


def test_metadata_is_valid_json(
    tmp_path,
) -> None:
    graph = _build_graph()

    artifact = save_graph_artifact(
        graph,
        tmp_path,
    )

    metadata = json.loads(
        artifact.metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert metadata["artifact"]["name"] == (
        "fraud-intelligence-heterogeneous-graph"
    )


def test_load_graph_artifact_round_trip(
    tmp_path,
) -> None:
    graph = _build_graph()

    artifact = save_graph_artifact(
        graph,
        tmp_path,
    )

    loaded = load_graph_artifact(
        artifact.graph_path,
        artifact.metadata_path,
    )

    assert loaded.node_counts == graph.node_counts
    assert loaded.edge_counts == graph.edge_counts
    assert loaded.total_nodes == graph.total_nodes
    assert loaded.total_edges == graph.total_edges


def test_round_trip_preserves_node_types(
    tmp_path,
) -> None:
    graph = _build_graph()

    artifact = save_graph_artifact(
        graph,
        tmp_path,
    )

    loaded = load_graph_artifact(
        artifact.graph_path,
        artifact.metadata_path,
    )

    assert set(loaded.data.node_types) == set(
        graph.data.node_types
    )


def test_round_trip_preserves_edge_types(
    tmp_path,
) -> None:
    graph = _build_graph()

    artifact = save_graph_artifact(
        graph,
        tmp_path,
    )

    loaded = load_graph_artifact(
        artifact.graph_path,
        artifact.metadata_path,
    )

    assert set(loaded.data.edge_types) == set(
        graph.data.edge_types
    )


def test_missing_graph_artifact_is_rejected(
    tmp_path,
) -> None:
    metadata_path = tmp_path / "metadata.json"

    metadata_path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(
        GraphMaterializationError,
        match="Graph artifact not found",
    ):
        load_graph_artifact(
            tmp_path / "missing.pt",
            metadata_path,
        )


def test_missing_metadata_is_rejected(
    tmp_path,
) -> None:
    graph_path = tmp_path / "graph.pt"

    torch.save(
        _build_graph().data,
        graph_path,
    )

    with pytest.raises(
        GraphMaterializationError,
        match="Graph metadata not found",
    ):
        load_graph_artifact(
            graph_path,
            tmp_path / "missing.json",
        )


def test_invalid_metadata_json_is_rejected(
    tmp_path,
) -> None:
    graph_path = tmp_path / "graph.pt"
    metadata_path = tmp_path / "metadata.json"

    torch.save(
        _build_graph().data,
        graph_path,
    )

    metadata_path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(
        GraphMaterializationError,
        match="not valid JSON",
    ):
        load_graph_artifact(
            graph_path,
            metadata_path,
        )


def test_wrong_artifact_version_is_rejected(
    tmp_path,
) -> None:
    graph = _build_graph()

    artifact = save_graph_artifact(
        graph,
        tmp_path,
    )

    metadata = json.loads(
        artifact.metadata_path.read_text(
            encoding="utf-8"
        )
    )

    metadata["artifact"]["version"] = "999.0.0"

    artifact.metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        GraphMaterializationError,
        match="Unsupported graph artifact version",
    ):
        load_graph_artifact(
            artifact.graph_path,
            artifact.metadata_path,
        )


def test_modified_graph_fails_hash_validation(
    tmp_path,
) -> None:
    graph = _build_graph()

    artifact = save_graph_artifact(
        graph,
        tmp_path,
    )

    data = torch.load(
        artifact.graph_path,
        map_location="cpu",
        weights_only=False,
    )

    edge_type = data.edge_types[0]

    data[edge_type].edge_index = torch.tensor(
        [[1], [0]],
        dtype=torch.long,
    )

    torch.save(
        data,
        artifact.graph_path,
    )

    with pytest.raises(
        GraphMaterializationError,
        match="topology hash mismatch",
    ):
        load_graph_artifact(
            artifact.graph_path,
            artifact.metadata_path,
        )


def test_save_creates_nested_output_directory(
    tmp_path,
) -> None:
    graph = _build_graph()

    output_dir = (
        tmp_path
        / "data"
        / "graph"
    )

    artifact = save_graph_artifact(
        graph,
        output_dir,
    )

    assert artifact.graph_path.exists()
    assert artifact.metadata_path.exists()