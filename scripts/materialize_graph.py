from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_intelligence.graph.edges import build_all_edge_tables
from fraud_intelligence.graph.heterogeneous import (
    build_heterogeneous_graph,
)
from fraud_intelligence.graph.materialization import (
    save_graph_artifact,
)
from fraud_intelligence.graph.nodes import (
    build_all_node_tables,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "synthetic"
GRAPH_DIR = PROJECT_ROOT / "data" / "graph"


def load_raw_data() -> dict[str, pd.DataFrame]:
    """Load the frozen synthetic graph source tables."""

    return {
        "customers": pd.read_parquet(
            RAW_DIR / "customers.parquet"
        ),
        "accounts": pd.read_parquet(
            RAW_DIR / "accounts.parquet"
        ),
        "cards": pd.read_parquet(
            RAW_DIR / "cards.parquet"
        ),
        "transactions": pd.read_parquet(
            RAW_DIR / "transactions.parquet"
        ),
        "merchants": pd.read_parquet(
            RAW_DIR / "merchants.parquet"
        ),
        "devices": pd.read_parquet(
            RAW_DIR / "devices.parquet"
        ),
        "ips": pd.read_parquet(
            RAW_DIR / "ips.parquet"
        ),
        "customer_devices": pd.read_parquet(
            RAW_DIR / "customer_devices.parquet"
        ),
        "customer_ips": pd.read_parquet(
            RAW_DIR / "customer_ips.parquet"
        ),
        "customer_merchants": pd.read_parquet(
            RAW_DIR / "customer_merchants.parquet"
        ),
    }


def main() -> None:
    print("Loading raw synthetic data...")
    raw = load_raw_data()

    print("Building node tables...")
    node_tables = build_all_node_tables(
        customers=raw["customers"],
        accounts=raw["accounts"],
        cards=raw["cards"],
        transactions=raw["transactions"],
        merchants=raw["merchants"],
        devices=raw["devices"],
        ips=raw["ips"],
    )

    print("Building edge tables...")
    edge_tables = build_all_edge_tables(
        transactions=raw["transactions"],
        customer_devices=raw["customer_devices"],
        customer_ips=raw["customer_ips"],
        customer_merchants=raw["customer_merchants"],
        customer_nodes=node_tables["customer"],
        account_nodes=node_tables["account"],
        card_nodes=node_tables["card"],
        transaction_nodes=node_tables["transaction"],
        merchant_nodes=node_tables["merchant"],
        device_nodes=node_tables["device"],
        ip_nodes=node_tables["ip"],
    )

    print("Constructing heterogeneous graph...")
    graph = build_heterogeneous_graph(
        node_tables=node_tables,
        edge_tables=edge_tables,
    )

    print("Materializing graph artifact...")
    artifact = save_graph_artifact(
        graph=graph,
        output_dir=GRAPH_DIR,
    )

    print()
    print("Graph materialization complete.")
    print(f"Graph:    {artifact.graph_path}")
    print(f"Metadata: {artifact.metadata_path}")
    print()
    print(
        f"Total nodes: {artifact.metadata['graph']['total_nodes']}"
    )
    print(
        f"Total edges: {artifact.metadata['graph']['total_edges']}"
    )
    print(
        "Topology SHA-256: "
        f"{artifact.metadata['topology']['sha256']}"
    )


if __name__ == "__main__":
    main()