from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import HeteroData

from fraud_intelligence.graph.contracts import (
    EDGE_TYPES,
    NODE_TYPES,
)
from fraud_intelligence.graph.heterogeneous import (
    HeterogeneousGraph,
    validate_heterogeneous_graph,
)
from fraud_intelligence.graph.statistics import (
    calculate_graph_statistics,
)


GRAPH_ARTIFACT_VERSION = "1.0.0"

GRAPH_FILENAME = "heterogeneous_graph.pt"
METADATA_FILENAME = "graph_metadata.json"


class GraphMaterializationError(ValueError):
    """Raised when graph artifact materialization fails."""


@dataclass(frozen=True)
class GraphArtifact:
    """Materialized graph artifact and metadata."""

    graph_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def _validate_output_directory(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return stable CPU byte representation of a tensor."""

    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .numpy()
        .tobytes()
    )


def _calculate_topology_hash(
    graph: HeterogeneousGraph,
) -> str:
    """
    Calculate a deterministic SHA-256 hash of graph topology.

    The hash includes:
    - canonical node types
    - node counts
    - canonical edge types
    - edge indices
    - edge counts

    It deliberately excludes fraud labels and arbitrary object metadata.
    """

    hasher = hashlib.sha256()

    for node_type in NODE_TYPES:
        hasher.update(
            node_type.encode("utf-8")
        )

        hasher.update(
            str(graph.node_counts[node_type]).encode(
                "utf-8"
            )
        )

    for edge_type in EDGE_TYPES:
        hasher.update(
            edge_type.encode("utf-8")
        )

        hasher.update(
            str(graph.edge_counts[edge_type]).encode(
                "utf-8"
            )
        )

        graph_edge_type = next(
            edge
            for edge in graph.data.edge_types
            if edge[1] == edge_type
        )

        edge_index = graph.data[
            graph_edge_type
        ].edge_index

        hasher.update(
            _tensor_bytes(edge_index)
        )

    return hasher.hexdigest()


def build_graph_metadata(
    graph: HeterogeneousGraph,
) -> dict[str, Any]:
    """Build metadata describing a graph artifact."""

    validate_heterogeneous_graph(graph)

    statistics = calculate_graph_statistics(graph)

    return {
        "artifact": {
            "name": "fraud-intelligence-heterogeneous-graph",
            "version": GRAPH_ARTIFACT_VERSION,
            "format": "torch-geometric-heterodata",
        },
        "schema": {
            "node_types": list(NODE_TYPES),
            "edge_types": list(EDGE_TYPES),
        },
        "graph": {
            "total_nodes": graph.total_nodes,
            "total_edges": graph.total_edges,
            "node_counts": dict(graph.node_counts),
            "edge_counts": dict(graph.edge_counts),
        },
        "topology": {
            "sha256": _calculate_topology_hash(graph),
        },
        "statistics": statistics.to_dict(),
    }


def save_graph_artifact(
    graph: HeterogeneousGraph,
    output_dir: Path,
) -> GraphArtifact:
    """
    Validate and save a heterogeneous graph artifact.

    The graph is saved as a PyTorch Geometric HeteroData object and
    accompanied by JSON metadata.
    """

    validate_heterogeneous_graph(graph)

    output_dir = _validate_output_directory(
        output_dir
    )

    graph_path = output_dir / GRAPH_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    metadata = build_graph_metadata(graph)

    torch.save(
        graph.data,
        graph_path,
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return GraphArtifact(
        graph_path=graph_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def load_graph_artifact(
    graph_path: Path,
    metadata_path: Path,
) -> HeterogeneousGraph:
    """Load and validate a materialized graph artifact."""

    graph_path = Path(graph_path)
    metadata_path = Path(metadata_path)

    if not graph_path.exists():
        raise GraphMaterializationError(
            f"Graph artifact not found: {graph_path}"
        )

    if not metadata_path.exists():
        raise GraphMaterializationError(
            f"Graph metadata not found: {metadata_path}"
        )

    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise GraphMaterializationError(
            "Graph metadata is not valid JSON."
        ) from exc

    expected_version = metadata.get(
        "artifact", {}
    ).get("version")

    if expected_version != GRAPH_ARTIFACT_VERSION:
        raise GraphMaterializationError(
            "Unsupported graph artifact version: "
            f"{expected_version}"
        )

    data = torch.load(
        graph_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(data, HeteroData):
        raise GraphMaterializationError(
            "Materialized object is not a HeteroData instance."
        )

    node_counts = {
        node_type: int(
            metadata["graph"]["node_counts"][node_type]
        )
        for node_type in NODE_TYPES
    }

    edge_counts = {
        edge_type: int(
            metadata["graph"]["edge_counts"][edge_type]
        )
        for edge_type in EDGE_TYPES
    }

    graph = HeterogeneousGraph(
        data=data,
        node_counts=node_counts,
        edge_counts=edge_counts,
    )

    validate_heterogeneous_graph(graph)

    actual_hash = _calculate_topology_hash(graph)

    expected_hash = metadata["topology"]["sha256"]

    if actual_hash != expected_hash:
        raise GraphMaterializationError(
            "Graph topology hash mismatch. "
            "The artifact may have been modified or corrupted."
        )

    return graph