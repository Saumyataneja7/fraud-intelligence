#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "api" / "main.py"
SERVICE = ROOT / "api" / "services" / "graph.py"
BUILDER = ROOT / "scripts" / "build_graph_api_index.py"

if not (ROOT / "configs").is_dir():
    raise SystemExit("Run this script from the fraud-intelligence repository root.")

SERVICE.parent.mkdir(parents=True, exist_ok=True)
BUILDER.parent.mkdir(parents=True, exist_ok=True)

for path in (MAIN, SERVICE):
    if path.exists():
        backup = path.with_suffix(path.suffix + ".phase10_7_backup")
        shutil.copy2(path, backup)
        print(f"Backed up {path} -> {backup}")

SERVICE.write_text('from __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom pathlib import Path\nimport sqlite3\n\nfrom api.contracts.graph import GraphEdgeResponse, GraphNodeResponse, GraphResponse\n\n\nPROJECT_ROOT = Path(__file__).resolve().parents[2]\nGRAPH_API_INDEX_PATH = PROJECT_ROOT / "data" / "graph" / "graph_api_index.sqlite"\n\nSUPPORTED_NODE_TYPES = frozenset(\n    {"customer", "account", "card", "transaction", "merchant", "device", "ip"}\n)\n\n\nclass GraphInvestigationServiceError(RuntimeError):\n    """Raised when graph investigation cannot be completed."""\n\n\n@dataclass(frozen=True)\nclass GraphInvestigationService:\n    index_path: Path = GRAPH_API_INDEX_PATH\n\n    def _connect(self) -> sqlite3.Connection:\n        if not self.index_path.exists():\n            raise GraphInvestigationServiceError(\n                f"Graph API index not found: {self.index_path}"\n            )\n        connection = sqlite3.connect(str(self.index_path))\n        connection.row_factory = sqlite3.Row\n        return connection\n\n    @staticmethod\n    def _validate_node_type(node_type: str) -> str:\n        normalized = str(node_type).strip().lower()\n        if normalized not in SUPPORTED_NODE_TYPES:\n            raise GraphInvestigationServiceError(\n                f"Unsupported graph node type: {node_type}"\n            )\n        return normalized\n\n    @staticmethod\n    def _validate_node_id(node_id: int) -> int:\n        try:\n            value = int(node_id)\n        except (TypeError, ValueError) as exc:\n            raise GraphInvestigationServiceError(\n                f"Invalid graph node id: {node_id}"\n            ) from exc\n        if value < 0:\n            raise GraphInvestigationServiceError(\n                f"Invalid graph node id: {node_id}"\n            )\n        return value\n\n    def investigate(self, *, node_type: str, node_id: int) -> GraphResponse:\n        normalized_type = self._validate_node_type(node_type)\n        normalized_id = self._validate_node_id(node_id)\n\n        with self._connect() as connection:\n            target_exists = connection.execute(\n                """\n                SELECT 1\n                FROM graph_nodes\n                WHERE node_type = ? AND node_id = ?\n                """,\n                (normalized_type, normalized_id),\n            ).fetchone()\n\n            if target_exists is None:\n                raise GraphInvestigationServiceError(\n                    f"Unknown graph node: {normalized_type}:{normalized_id}"\n                )\n\n            rows = connection.execute(\n                """\n                SELECT\n                    relationship_type,\n                    source_node_type,\n                    source_node_id,\n                    target_node_type,\n                    target_node_id\n                FROM graph_edges\n                WHERE\n                    (source_node_type = ? AND source_node_id = ?)\n                    OR\n                    (target_node_type = ? AND target_node_id = ?)\n                ORDER BY\n                    relationship_type,\n                    source_node_type,\n                    source_node_id,\n                    target_node_type,\n                    target_node_id\n                """,\n                (\n                    normalized_type,\n                    normalized_id,\n                    normalized_type,\n                    normalized_id,\n                ),\n            ).fetchall()\n\n        edges = [\n            GraphEdgeResponse(\n                relationship_type=row["relationship_type"],\n                source_node_type=row["source_node_type"],\n                source_node_id=int(row["source_node_id"]),\n                target_node_type=row["target_node_type"],\n                target_node_id=int(row["target_node_id"]),\n            )\n            for row in rows\n        ]\n\n        node_keys = {(normalized_type, normalized_id)}\n        for edge in edges:\n            node_keys.add((edge.source_node_type, edge.source_node_id))\n            node_keys.add((edge.target_node_type, edge.target_node_id))\n\n        nodes = [\n            GraphNodeResponse(node_type=kind, node_id=node_id_value)\n            for kind, node_id_value in sorted(node_keys)\n        ]\n\n        return GraphResponse(\n            target_node_type=normalized_type,\n            target_node_id=normalized_id,\n            nodes=nodes,\n            edges=edges,\n        )\n', encoding="utf-8")
BUILDER.write_text('from __future__ import annotations\n\nfrom pathlib import Path\nimport sqlite3\nimport sys\n\nPROJECT_ROOT = Path(__file__).resolve().parents[1]\nsys.path.insert(0, str(PROJECT_ROOT))\n\nfrom fraud_intelligence.graph.materialization import (  # noqa: E402\n    GRAPH_ARTIFACT_VERSION,\n    GRAPH_FILENAME,\n    METADATA_FILENAME,\n    load_graph_artifact,\n)\n\nGRAPH_DIR = PROJECT_ROOT / "data" / "graph"\nGRAPH_PATH = GRAPH_DIR / GRAPH_FILENAME\nMETADATA_PATH = GRAPH_DIR / METADATA_FILENAME\nINDEX_PATH = GRAPH_DIR / "graph_api_index.sqlite"\n\nSCHEMA = """\nPRAGMA journal_mode = DELETE;\nPRAGMA synchronous = NORMAL;\n\nCREATE TABLE graph_metadata (\n    key TEXT PRIMARY KEY,\n    value TEXT NOT NULL\n);\n\nCREATE TABLE graph_nodes (\n    node_type TEXT NOT NULL,\n    node_id INTEGER NOT NULL,\n    PRIMARY KEY (node_type, node_id)\n);\n\nCREATE TABLE graph_edges (\n    relationship_type TEXT NOT NULL,\n    source_node_type TEXT NOT NULL,\n    source_node_id INTEGER NOT NULL,\n    target_node_type TEXT NOT NULL,\n    target_node_id INTEGER NOT NULL\n);\n\nCREATE INDEX idx_graph_edges_source\nON graph_edges(source_node_type, source_node_id);\n\nCREATE INDEX idx_graph_edges_target\nON graph_edges(target_node_type, target_node_id);\n\nCREATE INDEX idx_graph_edges_relationship\nON graph_edges(relationship_type);\n"""\n\n\ndef main() -> None:\n    if not GRAPH_PATH.exists():\n        raise SystemExit(f"Missing Phase 6 graph artifact: {GRAPH_PATH}")\n    if not METADATA_PATH.exists():\n        raise SystemExit(f"Missing Phase 6 graph metadata: {METADATA_PATH}")\n\n    print("Loading frozen Phase 6 graph artifact...")\n    graph = load_graph_artifact(\n        graph_path=GRAPH_PATH,\n        metadata_path=METADATA_PATH,\n    )\n\n    if INDEX_PATH.exists():\n        INDEX_PATH.unlink()\n\n    connection = sqlite3.connect(str(INDEX_PATH))\n    try:\n        connection.executescript(SCHEMA)\n\n        metadata_rows = {\n            "artifact_version": GRAPH_ARTIFACT_VERSION,\n            "source_graph": str(GRAPH_PATH.relative_to(PROJECT_ROOT)),\n            "source_metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)),\n        }\n        connection.executemany(\n            "INSERT INTO graph_metadata(key, value) VALUES (?, ?)",\n            metadata_rows.items(),\n        )\n\n        print("Writing graph nodes...")\n        for node_type, count in graph.node_counts.items():\n            connection.executemany(\n                """\n                INSERT INTO graph_nodes(node_type, node_id)\n                VALUES (?, ?)\n                """,\n                ((node_type, node_id) for node_id in range(int(count))),\n            )\n\n        print("Writing graph edges...")\n        edge_total = 0\n\n        for edge_type in graph.data.edge_types:\n            source_type, relationship_type, target_type = edge_type\n            edge_index = graph.data[edge_type].edge_index\n\n            source_ids = edge_index[0].detach().cpu().tolist()\n            target_ids = edge_index[1].detach().cpu().tolist()\n\n            connection.executemany(\n                """\n                INSERT INTO graph_edges(\n                    relationship_type,\n                    source_node_type,\n                    source_node_id,\n                    target_node_type,\n                    target_node_id\n                )\n                VALUES (?, ?, ?, ?, ?)\n                """,\n                (\n                    (\n                        str(relationship_type),\n                        str(source_type),\n                        int(source_id),\n                        str(target_type),\n                        int(target_id),\n                    )\n                    for source_id, target_id in zip(source_ids, target_ids)\n                ),\n            )\n\n            edge_total += len(source_ids)\n            print(\n                f"  {source_type}:{relationship_type}:{target_type} "\n                f"-> {len(source_ids):,} edges"\n            )\n\n        connection.commit()\n\n        node_total = connection.execute(\n            "SELECT COUNT(*) FROM graph_nodes"\n        ).fetchone()[0]\n        stored_edge_total = connection.execute(\n            "SELECT COUNT(*) FROM graph_edges"\n        ).fetchone()[0]\n\n        if stored_edge_total != edge_total:\n            raise RuntimeError(\n                f"Edge count mismatch: expected {edge_total}, "\n                f"stored {stored_edge_total}"\n            )\n\n        print()\n        print("Graph API index created successfully.")\n        print(f"  Path:  {INDEX_PATH}")\n        print(f"  Nodes: {node_total:,}")\n        print(f"  Edges: {stored_edge_total:,}")\n\n    finally:\n        connection.close()\n\n\nif __name__ == "__main__":\n    main()\n', encoding="utf-8")

main_text = MAIN.read_text(encoding="utf-8")

main_text = re.sub(
    r"from contextlib import asynccontextmanager\n\n",
    "",
    main_text,
)
main_text = re.sub(
    r"@asynccontextmanager\n"
    r"async def lifespan\(app: FastAPI\):\n"
    r"(?:    .*\n)+?"
    r"    app\.state\.graph_index = None\n\n",
    "",
    main_text,
)
main_text = re.sub(r",\n    lifespan=lifespan", "", main_text)

route_pattern = re.compile(
    r'@app\.get\(\n'
    r'    "/graph/\{node_id\}", response_model=GraphResponse, tags=\["graph"\]\n'
    r'\)\n'
    r'def investigate_graph\(node_id: int, node_type: str\) -> GraphResponse:\n'
    r'(?:    .*\n)+?'
    r'        raise HTTPException\(status_code=503, detail="Graph investigation service is unavailable\."\) from exc\n',
    re.MULTILINE,
)

NEW_ROUTE = '@app.get(\n    "/graph/{node_id}",\n    response_model=GraphResponse,\n    tags=["graph"],\n)\ndef investigate_graph(node_id: int, node_type: str) -> GraphResponse:\n    try:\n        return graph_investigation_service.investigate(\n            node_type=node_type,\n            node_id=node_id,\n        )\n    except GraphInvestigationServiceError as exc:\n        message = str(exc)\n        if message.startswith("Unknown graph node"):\n            raise HTTPException(status_code=404, detail=message) from exc\n        if message.startswith("Unsupported graph node type"):\n            raise HTTPException(status_code=404, detail=message) from exc\n        if message.startswith("Invalid graph node id"):\n            raise HTTPException(status_code=422, detail=message) from exc\n        raise HTTPException(\n            status_code=503,\n            detail="Graph investigation service is unavailable.",\n        ) from exc\n'

main_text, count = route_pattern.subn(NEW_ROUTE, main_text)
if count != 1:
    raise SystemExit(
        "Could not safely locate the existing /graph route in api/main.py."
    )

MAIN.write_text(main_text, encoding="utf-8")

print("Updated graph service and API route.")
print("Building the SQLite graph API index...")
subprocess.run(
    [sys.executable, str(BUILDER)],
    cwd=str(ROOT),
    check=True,
)

print()
print("Phase 10.7 graph fix applied.")
print("Run:")
print("  .venv/bin/python -m pytest tests/unit/test_api_graph.py -q")
print("  .venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000")