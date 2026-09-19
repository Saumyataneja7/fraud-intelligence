from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fraud_intelligence.graph.contracts import (
    EDGE_TYPES,
    NODE_TYPES,
)
from fraud_intelligence.graph.materialization import (
    GRAPH_ARTIFACT_VERSION,
    METADATA_FILENAME,
)
from fraud_intelligence.graph.nodes import (
    NODE_SOURCE_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPH_DIR = (
    PROJECT_ROOT
    / "data"
    / "graph"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "graph"
)

METADATA_PATH = (
    GRAPH_DIR
    / METADATA_FILENAME
)

REPORT_PATH = (
    REPORT_DIR
    / "GRAPH_CONSTRUCTION_REPORT.md"
)

FREEZE_PATH = (
    REPORT_DIR
    / "PHASE_6_FREEZE.txt"
)


def _load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Graph metadata not found: {METADATA_PATH}"
        )

    return json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )


def _format_dict_table(
    values: dict[str, int],
    first_column: str,
    second_column: str,
) -> str:
    lines = [
        f"| {first_column} | {second_column} |",
        "|---|---:|",
    ]

    for key, value in values.items():
        lines.append(
            f"| `{key}` | {value:,} |"
        )

    return "\n".join(lines)


def _format_degree_statistics(
    records: list[dict],
) -> str:
    lines = [
        "| Node type | Count | Min | Max | Mean | Median | P95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in records:
        lines.append(
            "| `{node_type}` | {node_count:,} | "
            "{min_degree:.2f} | {max_degree:.2f} | "
            "{mean_degree:.2f} | {median_degree:.2f} | "
            "{p95_degree:.2f} |".format(**row)
        )

    return "\n".join(lines)


def build_report(metadata: dict) -> str:
    artifact = metadata["artifact"]
    schema = metadata["schema"]
    graph = metadata["graph"]
    topology = metadata["topology"]
    statistics = metadata["statistics"]

    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    node_source_rows = "\n".join(
        f"| `{node_type}` | "
        f"`{NODE_SOURCE_COLUMNS[node_type]}` |"
        for node_type in NODE_TYPES
    )

    node_counts_table = _format_dict_table(
        graph["node_counts"],
        "Node type",
        "Count",
    )

    edge_counts_table = _format_dict_table(
        graph["edge_counts"],
        "Edge type",
        "Count",
    )

    degree_table = _format_degree_statistics(
        statistics["degree_statistics"]
    )

    components = statistics[
        "connected_components"
    ]

    isolated = statistics[
        "isolated_node_counts"
    ]

    isolated_table = _format_dict_table(
        isolated,
        "Node type",
        "Isolated nodes",
    )

    return f"""# Graph Construction Report

## Phase

**Phase 6 — Graph Construction**

**Phase 6.10 — Graph Construction Report & Freeze**

Generated at: `{generated_at}`

---

## 1. Executive Summary

The Fraud Intelligence project now contains a validated,
heterogeneous fraud graph constructed from the synthetic
transaction dataset.

The graph represents:

- customers
- accounts
- cards
- transactions
- merchants
- devices
- IP addresses

The graph topology is constructed from relational entity
relationships and transaction relationships.

Fraud labels are not used to create graph topology.

---

## 2. Artifact

| Property | Value |
|---|---|
| Artifact name | `{artifact["name"]}` |
| Artifact version | `{artifact["version"]}` |
| Format | `{artifact["format"]}` |
| Graph file | `data/graph/heterogeneous_graph.pt` |
| Metadata file | `data/graph/{METADATA_FILENAME}` |
| Topology SHA-256 | `{topology["sha256"]}` |

---

## 3. Node Schema

The canonical node types are:

{", ".join(f"`{node}`" for node in schema["node_types"])}

### Source ID mapping

| Node type | Source identifier |
|---|---|
{node_source_rows}

### Node counts

{node_counts_table}

**Total nodes:** {graph["total_nodes"]:,}

---

## 4. Edge Schema

The graph contains the following canonical relationships:

{", ".join(f"`{edge}`" for edge in schema["edge_types"])}

### Edge counts

{edge_counts_table}

**Total edges:** {graph["total_edges"]:,}

---

## 5. Graph Construction

Graph construction follows the established pipeline:

```text
Relational source data
        ↓
Node tables
        ↓
Deterministic graph ID mapping
        ↓
Typed edge tables
        ↓
Heterogeneous PyG graph
        ↓
Structural validation
        ↓
Temporal controls
        ↓
Graph statistics
        ↓
Materialized graph artifact
'''

Node IDs are deterministic contiguous integer identifiers.

Edge endpoints reference graph node IDs rather than raw
business identifiers.

---

## 6. Structural Validation

The graph validation layer verifies:

-canonical node types
-canonical edge types
-node ID presence
-node ID dtype
-contiguous node IDs
-edge index shape
-edge index dtype
-endpoint ranges
-duplicate edges
-graph metadata consistency
-node and edge counts

The graph must pass structural validation before it is
materialized.

---

## 7. Temporal Controls

Temporal controls use strict point-in-time semantics.

For a cutoff timestamp T, only events satisfying:

event_timestamp < T

are considered historically available.

Events occurring:

event_timestamp == T

are excluded because the dataset does not provide an
explicit ordering mechanism for events sharing the same
timestamp.

Future events are rejected by the temporal validation layer.

This prevents future transaction information from being
introduced into historical graph context.

---

## 8. Graph Statistics
Degree statistics

{degree_table}

Isolated nodes

{isolated_table}

Connected components
Statistic	Value
Component count	{components["component_count"]:,}
Largest component size	{components["largest_component_size"]:,}
Largest component fraction	{components["largest_component_fraction"]:.6f}

Connected-component statistics are structural diagnostics.
They are not fraud-ring classifications.

---

## 9. Reproducibility

The graph artifact includes:

-explicit artifact version
-canonical node schema
-canonical edge schema
-node counts
-edge counts
-graph statistics
-deterministic topology hash

The graph can be loaded and structurally validated through
the graph materialization layer.

The topology hash provides an integrity check against
unexpected graph modification.

---

## 10. Leakage Controls

The graph construction design explicitly separates:

### Allowed

-entity relationships
-transaction relationships
-graph identifiers
-transaction timestamps
-historical relationships subject to point-in-time filtering

### Not used for topology

-'is_fraud'
-'fraud_scenario'
-future transactions
-same-timestamp transactions without event ordering

The `is_fraud` and `fraud_scenario` columns are retained only
as label metadata where required by the source transaction
representation. They are not used to create graph topology.

Fraud labels remain targets or post-hoc analysis attributes,
rather than graph-construction inputs.

---

## 11. Limitations

This graph is generated from a synthetic dataset designed for
Fraud Intelligence experimentation.

It is not a representation of real financial activity.

The current graph does not establish causal relationships
between entities.

Shared devices, IP addresses, merchants, or other entities
can have legitimate explanations.

Connected components therefore represent structural connectivity
rather than confirmed fraud rings.

Temporal filtering is available as an explicit control layer;
future Graph ML pipelines must use it whenever a
point-in-time training or inference graph is required.

---

## 12. Phase 6 Completion

Phases 6.1 through 6.9 establish:

-graph contracts
-node construction
-edge construction
-deterministic ID mapping
-heterogeneous graph construction
-structural validation
-temporal controls
-graph statistics
-graph artifact materialization

The graph-construction layer is therefore ready for the
next stage of the project.

---

## 13. Transition to Graph ML

The next stage can consume the validated graph artifact for
Graph ML experimentation.

Potential next-stage work includes:

-graph feature preparation
-node/edge feature design
-temporal graph dataset construction
-GraphSAGE
-graph-based fraud prediction
-graph embeddings
-graph-level investigation
-explainability

No Graph ML model is included in Phase 6.

Graph artifact version: {GRAPH_ARTIFACT_VERSION}

Status: Phase 6 graph construction complete.
"""

def build_freeze(metadata: dict) -> str:
    topology_hash = metadata["topology"]["sha256"]

    return f"""PHASE 6 FREEZE
    Project: Fraud Intelligence

Phase: 6 — Graph Construction
Version: {GRAPH_ARTIFACT_VERSION}

Status: FROZEN

Frozen components:

Graph contracts
Node tables
Edge tables
Graph ID mapping
Heterogeneous graph construction
Graph validation
Temporal graph controls
Graph statistics
Graph artifact materialization

Graph artifact:
data/graph/heterogeneous_graph.pt

Metadata:
data/graph/{METADATA_FILENAME}

Topology SHA-256:
{topology_hash}

The Phase 6 graph topology should not be changed
without intentionally creating a new graph artifact
version.

Next phase:
Graph ML
"""

def main() -> None:
    metadata = _load_metadata()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        build_report(metadata),
        encoding="utf-8",
    )

    FREEZE_PATH.write_text(
        build_freeze(metadata),
        encoding="utf-8",
    )

    print(
        f"Generated: {REPORT_PATH}"
    )
    print(
        f"Generated: {FREEZE_PATH}"
    )

if __name__ == "__main__":
    main()