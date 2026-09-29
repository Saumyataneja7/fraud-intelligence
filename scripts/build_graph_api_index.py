from __future__ import annotations

from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fraud_intelligence.graph.materialization import (  # noqa: E402
    GRAPH_ARTIFACT_VERSION,
    GRAPH_FILENAME,
    METADATA_FILENAME,
    load_graph_artifact,
)

GRAPH_DIR = PROJECT_ROOT / "data" / "graph"
GRAPH_PATH = GRAPH_DIR / GRAPH_FILENAME
METADATA_PATH = GRAPH_DIR / METADATA_FILENAME
INDEX_PATH = GRAPH_DIR / "graph_api_index.sqlite"

SCHEMA = """
PRAGMA journal_mode = DELETE;
PRAGMA synchronous = NORMAL;

CREATE TABLE graph_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE graph_nodes (
    node_type TEXT NOT NULL,
    node_id INTEGER NOT NULL,
    PRIMARY KEY (node_type, node_id)
);

CREATE TABLE graph_edges (
    relationship_type TEXT NOT NULL,
    source_node_type TEXT NOT NULL,
    source_node_id INTEGER NOT NULL,
    target_node_type TEXT NOT NULL,
    target_node_id INTEGER NOT NULL
);

CREATE INDEX idx_graph_edges_source
ON graph_edges(source_node_type, source_node_id);

CREATE INDEX idx_graph_edges_target
ON graph_edges(target_node_type, target_node_id);

CREATE INDEX idx_graph_edges_relationship
ON graph_edges(relationship_type);
"""


def main() -> None:
    if not GRAPH_PATH.exists():
        raise SystemExit(f"Missing Phase 6 graph artifact: {GRAPH_PATH}")
    if not METADATA_PATH.exists():
        raise SystemExit(f"Missing Phase 6 graph metadata: {METADATA_PATH}")

    print("Loading frozen Phase 6 graph artifact...")
    graph = load_graph_artifact(
        graph_path=GRAPH_PATH,
        metadata_path=METADATA_PATH,
    )

    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

    connection = sqlite3.connect(str(INDEX_PATH))
    try:
        connection.executescript(SCHEMA)

        metadata_rows = {
            "artifact_version": GRAPH_ARTIFACT_VERSION,
            "source_graph": str(GRAPH_PATH.relative_to(PROJECT_ROOT)),
            "source_metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
        }
        connection.executemany(
            "INSERT INTO graph_metadata(key, value) VALUES (?, ?)",
            metadata_rows.items(),
        )

        print("Writing graph nodes...")
        for node_type, count in graph.node_counts.items():
            connection.executemany(
                """
                INSERT INTO graph_nodes(node_type, node_id)
                VALUES (?, ?)
                """,
                ((node_type, node_id) for node_id in range(int(count))),
            )

        print("Writing graph edges...")
        edge_total = 0

        for edge_type in graph.data.edge_types:
            source_type, relationship_type, target_type = edge_type
            edge_index = graph.data[edge_type].edge_index

            source_ids = edge_index[0].detach().cpu().tolist()
            target_ids = edge_index[1].detach().cpu().tolist()

            connection.executemany(
                """
                INSERT INTO graph_edges(
                    relationship_type,
                    source_node_type,
                    source_node_id,
                    target_node_type,
                    target_node_id
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        str(relationship_type),
                        str(source_type),
                        int(source_id),
                        str(target_type),
                        int(target_id),
                    )
                    for source_id, target_id in zip(source_ids, target_ids)
                ),
            )

            edge_total += len(source_ids)
            print(
                f"  {source_type}:{relationship_type}:{target_type} "
                f"-> {len(source_ids):,} edges"
            )

        connection.commit()

        node_total = connection.execute(
            "SELECT COUNT(*) FROM graph_nodes"
        ).fetchone()[0]
        stored_edge_total = connection.execute(
            "SELECT COUNT(*) FROM graph_edges"
        ).fetchone()[0]

        if stored_edge_total != edge_total:
            raise RuntimeError(
                f"Edge count mismatch: expected {edge_total}, "
                f"stored {stored_edge_total}"
            )

        print()
        print("Graph API index created successfully.")
        print(f"  Path:  {INDEX_PATH}")
        print(f"  Nodes: {node_total:,}")
        print(f"  Edges: {stored_edge_total:,}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
