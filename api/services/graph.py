from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

from api.contracts.graph import GraphEdgeResponse, GraphNodeResponse, GraphResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_API_INDEX_PATH = PROJECT_ROOT / "data" / "graph" / "graph_api_index.sqlite"

SUPPORTED_NODE_TYPES = frozenset(
    {"customer", "account", "card", "transaction", "merchant", "device", "ip"}
)


class GraphInvestigationServiceError(RuntimeError):
    """Raised when graph investigation cannot be completed."""


@dataclass(frozen=True)
class GraphInvestigationService:
    index_path: Path = GRAPH_API_INDEX_PATH

    def _connect(self) -> sqlite3.Connection:
        if not self.index_path.exists():
            raise GraphInvestigationServiceError(
                f"Graph API index not found: {self.index_path}"
            )
        connection = sqlite3.connect(str(self.index_path))
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _validate_node_type(node_type: str) -> str:
        normalized = str(node_type).strip().lower()
        if normalized not in SUPPORTED_NODE_TYPES:
            raise GraphInvestigationServiceError(
                f"Unsupported graph node type: {node_type}"
            )
        return normalized

    @staticmethod
    def _validate_node_id(node_id: int) -> int:
        try:
            value = int(node_id)
        except (TypeError, ValueError) as exc:
            raise GraphInvestigationServiceError(
                f"Invalid graph node id: {node_id}"
            ) from exc
        if value < 0:
            raise GraphInvestigationServiceError(
                f"Invalid graph node id: {node_id}"
            )
        return value

    def investigate(self, *, node_type: str, node_id: int) -> GraphResponse:
        normalized_type = self._validate_node_type(node_type)
        normalized_id = self._validate_node_id(node_id)

        with self._connect() as connection:
            target_exists = connection.execute(
                """
                SELECT 1
                FROM graph_nodes
                WHERE node_type = ? AND node_id = ?
                """,
                (normalized_type, normalized_id),
            ).fetchone()

            if target_exists is None:
                raise GraphInvestigationServiceError(
                    f"Unknown graph node: {normalized_type}:{normalized_id}"
                )

            rows = connection.execute(
                """
                SELECT
                    relationship_type,
                    source_node_type,
                    source_node_id,
                    target_node_type,
                    target_node_id
                FROM graph_edges
                WHERE
                    (source_node_type = ? AND source_node_id = ?)
                    OR
                    (target_node_type = ? AND target_node_id = ?)
                ORDER BY
                    relationship_type,
                    source_node_type,
                    source_node_id,
                    target_node_type,
                    target_node_id
                """,
                (
                    normalized_type,
                    normalized_id,
                    normalized_type,
                    normalized_id,
                ),
            ).fetchall()

        edges = [
            GraphEdgeResponse(
                relationship_type=row["relationship_type"],
                source_node_type=row["source_node_type"],
                source_node_id=int(row["source_node_id"]),
                target_node_type=row["target_node_type"],
                target_node_id=int(row["target_node_id"]),
            )
            for row in rows
        ]

        node_keys = {(normalized_type, normalized_id)}
        for edge in edges:
            node_keys.add((edge.source_node_type, edge.source_node_id))
            node_keys.add((edge.target_node_type, edge.target_node_id))

        nodes = [
            GraphNodeResponse(node_type=kind, node_id=node_id_value)
            for kind, node_id_value in sorted(node_keys)
        ]

        return GraphResponse(
            target_node_type=normalized_type,
            target_node_id=normalized_id,
            nodes=nodes,
            edges=edges,
        )
