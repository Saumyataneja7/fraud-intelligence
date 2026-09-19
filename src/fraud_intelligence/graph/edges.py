from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from fraud_intelligence.graph.contracts import (
    ACCOUNT,
    ACCOUNT_CARD,
    ACCOUNT_TRANSACTION,
    CARD,
    CARD_TRANSACTION,
    CUSTOMER,
    CUSTOMER_ACCOUNT,
    CUSTOMER_DEVICE,
    CUSTOMER_IP,
    CUSTOMER_MERCHANT,
    CUSTOMER_TRANSACTION,
    DEVICE,
    EDGE_TYPES,
    IP,
    MERCHANT,
    TRANSACTION,
    TRANSACTION_DEVICE,
    TRANSACTION_IP,
    TRANSACTION_MERCHANT,
)
from fraud_intelligence.graph.nodes import (
    NODE_ID_COLUMN,
    NodeTable,
)


class EdgeTableError(ValueError):
    """Raised when an edge table is invalid."""


SOURCE_NODE_COLUMN: Final[str] = "source_node_id"
TARGET_NODE_COLUMN: Final[str] = "target_node_id"


@dataclass(frozen=True)
class EdgeTable:
    """Validated typed edge table."""

    edge_type: str
    source_node_type: str
    target_node_type: str
    dataframe: pd.DataFrame

    @property
    def edge_count(self) -> int:
        return len(self.dataframe)


EDGE_ENDPOINTS: Final[
    dict[str, tuple[str, str]]
] = {
    CUSTOMER_ACCOUNT: (
        CUSTOMER,
        ACCOUNT,
    ),
    ACCOUNT_CARD: (
        ACCOUNT,
        CARD,
    ),
    CUSTOMER_TRANSACTION: (
        CUSTOMER,
        TRANSACTION,
    ),
    ACCOUNT_TRANSACTION: (
        ACCOUNT,
        TRANSACTION,
    ),
    CARD_TRANSACTION: (
        CARD,
        TRANSACTION,
    ),
    CUSTOMER_DEVICE: (
        CUSTOMER,
        DEVICE,
    ),
    CUSTOMER_IP: (
        CUSTOMER,
        IP,
    ),
    CUSTOMER_MERCHANT: (
        CUSTOMER,
        MERCHANT,
    ),
    TRANSACTION_MERCHANT: (
        TRANSACTION,
        MERCHANT,
    ),
    TRANSACTION_DEVICE: (
        TRANSACTION,
        DEVICE,
    ),
    TRANSACTION_IP: (
        TRANSACTION,
        IP,
    ),
}


def _validate_edge_type(
    edge_type: str,
) -> None:
    if edge_type not in EDGE_TYPES:
        raise EdgeTableError(
            f"Unsupported edge type: {edge_type}"
        )

    if edge_type not in EDGE_ENDPOINTS:
        raise EdgeTableError(
            f"No endpoint definition for edge type: "
            f"{edge_type}"
        )


def _validate_source_dataframe(
    dataframe: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> None:
    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise EdgeTableError(
            "Input must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise EdgeTableError(
            "Input DataFrame cannot be empty."
        )

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise EdgeTableError(
            "Input DataFrame is missing required "
            "columns: "
            + ", ".join(sorted(missing))
        )


def _build_id_to_node_map(
    node_table: NodeTable,
) -> dict[object, int]:
    dataframe = node_table.dataframe

    return dict(
        zip(
            dataframe[
                node_table.source_id_column
            ],
            dataframe[NODE_ID_COLUMN],
        )
    )


def _map_ids_to_node_ids(
    values: pd.Series,
    mapping: dict[object, int],
    column_name: str,
) -> pd.Series:
    mapped = values.map(mapping)

    if mapped.isna().any():
        missing_values = (
            values[mapped.isna()]
            .drop_duplicates()
            .tolist()
        )

        raise EdgeTableError(
            f"Unable to map {column_name} values "
            f"to graph node IDs: "
            f"{missing_values[:10]}"
        )

    return mapped.astype("int64")


def _build_edge_table(
    dataframe: pd.DataFrame,
    source_node_table: NodeTable,
    target_node_table: NodeTable,
    edge_type: str,
    source_id_column: str,
    target_id_column: str,
) -> EdgeTable:
    _validate_edge_type(edge_type)

    expected_source, expected_target = (
        EDGE_ENDPOINTS[edge_type]
    )

    if (
        source_node_table.node_type
        != expected_source
    ):
        raise EdgeTableError(
            f"Edge {edge_type} expects source node "
            f"type {expected_source}, got "
            f"{source_node_table.node_type}"
        )

    if (
        target_node_table.node_type
        != expected_target
    ):
        raise EdgeTableError(
            f"Edge {edge_type} expects target node "
            f"type {expected_target}, got "
            f"{target_node_table.node_type}"
        )

    _validate_source_dataframe(
        dataframe,
        (
            source_id_column,
            target_id_column,
        ),
    )

    source_map = _build_id_to_node_map(
        source_node_table
    )

    target_map = _build_id_to_node_map(
        target_node_table
    )

    edges = pd.DataFrame(
        {
            SOURCE_NODE_COLUMN: _map_ids_to_node_ids(
                dataframe[source_id_column],
                source_map,
                source_id_column,
            ),
            TARGET_NODE_COLUMN: _map_ids_to_node_ids(
                dataframe[target_id_column],
                target_map,
                target_id_column,
            ),
        }
    )

    # Multiple source records can describe the same
    # relationship. Graph topology should contain one
    # canonical edge for that relationship.
    edges = (
        edges
        .drop_duplicates()
        .sort_values(
            by=[
                SOURCE_NODE_COLUMN,
                TARGET_NODE_COLUMN,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    return EdgeTable(
        edge_type=edge_type,
        source_node_type=expected_source,
        target_node_type=expected_target,
        dataframe=edges,
    )


def build_customer_account_edges(
    relationships: pd.DataFrame,
    customer_nodes: NodeTable,
    account_nodes: NodeTable,
) -> EdgeTable:
    return _build_edge_table(
        relationships,
        customer_nodes,
        account_nodes,
        CUSTOMER_ACCOUNT,
        "customer_id",
        "account_id",
    )


def build_account_card_edges(
    relationships: pd.DataFrame,
    account_nodes: NodeTable,
    card_nodes: NodeTable,
) -> EdgeTable:
    return _build_edge_table(
        relationships,
        account_nodes,
        card_nodes,
        ACCOUNT_CARD,
        "account_id",
        "card_id",
    )


def build_customer_transaction_edges(
    transactions: pd.DataFrame,
    customer_nodes: NodeTable,
    transaction_nodes: NodeTable,
) -> EdgeTable:
    return _build_edge_table(
        transactions,
        customer_nodes,
        transaction_nodes,
        CUSTOMER_TRANSACTION,
        "customer_id",
        "transaction_id",
    )


def build_account_transaction_edges(
    transactions: pd.DataFrame,
    account_nodes: NodeTable,
    transaction_nodes: NodeTable,
) -> EdgeTable:
    return _build_edge_table(
        transactions,
        account_nodes,
        transaction_nodes,
        ACCOUNT_TRANSACTION,
        "account_id",
        "transaction_id",
    )


def build_card_transaction_edges(
    transactions: pd.DataFrame,
    card_nodes: NodeTable,
    transaction_nodes: NodeTable,
) -> EdgeTable:
    return _build_edge_table(
        transactions,
        card_nodes,
        transaction_nodes,
        CARD_TRANSACTION,
        "card_id",
        "transaction_id",
    )


def build_customer_device_edges(
    relationships: pd.DataFrame,
    customer_nodes: NodeTable,
    device_nodes: NodeTable,
) -> EdgeTable:
    return _build_edge_table(
        relationships,
        customer_nodes,
        device_nodes,
        CUSTOMER_DEVICE,
        "customer_id",
        "device_id",
    )


def build_customer_ip_edges(
    relationships: pd.DataFrame,
    customer_nodes: NodeTable,
    ip_nodes: NodeTable,
) -> EdgeTable:
    return _build_edge_table(
        relationships,
        customer_nodes,
        ip_nodes,
        CUSTOMER_IP,
        "customer_id",
        "ip_id",
    )


def build_customer_merchant_edges(
    relationships: pd.DataFrame,
    customer_nodes: NodeTable,
    merchant_nodes: NodeTable,
) -> EdgeTable:
    return _build_edge_table(
        relationships,
        customer_nodes,
        merchant_nodes,
        CUSTOMER_MERCHANT,
        "customer_id",
        "merchant_id",
    )


def build_transaction_merchant_edges(
    transactions: pd.DataFrame,
    transaction_nodes: NodeTable,
    merchant_nodes: NodeTable,
) -> EdgeTable:
    return _build_edge_table(
        transactions,
        transaction_nodes,
        merchant_nodes,
        TRANSACTION_MERCHANT,
        "transaction_id",
        "merchant_id",
    )


def build_transaction_device_edges(
    transactions: pd.DataFrame,
    transaction_nodes: NodeTable,
    device_nodes: NodeTable,
) -> EdgeTable:
    return _build_edge_table(
        transactions,
        transaction_nodes,
        device_nodes,
        TRANSACTION_DEVICE,
        "transaction_id",
        "device_id",
    )


def build_transaction_ip_edges(
    transactions: pd.DataFrame,
    transaction_nodes: NodeTable,
    ip_nodes: NodeTable,
) -> EdgeTable:
    return _build_edge_table(
        transactions,
        transaction_nodes,
        ip_nodes,
        TRANSACTION_IP,
        "transaction_id",
        "ip_id",
    )


def validate_edge_table(
    edge_table: EdgeTable,
) -> None:
    if not isinstance(
        edge_table,
        EdgeTable,
    ):
        raise EdgeTableError(
            "edge_table must be an EdgeTable."
        )

    _validate_edge_type(
        edge_table.edge_type
    )

    expected_source, expected_target = (
        EDGE_ENDPOINTS[edge_table.edge_type]
    )

    if (
        edge_table.source_node_type
        != expected_source
    ):
        raise EdgeTableError(
            "Incorrect source node type."
        )

    if (
        edge_table.target_node_type
        != expected_target
    ):
        raise EdgeTableError(
            "Incorrect target node type."
        )

    dataframe = edge_table.dataframe

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise EdgeTableError(
            "Edge dataframe must be a pandas DataFrame."
        )

    required = {
        SOURCE_NODE_COLUMN,
        TARGET_NODE_COLUMN,
    }

    missing = required - set(
        dataframe.columns
    )

    if missing:
        raise EdgeTableError(
            "Edge table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if dataframe[
        SOURCE_NODE_COLUMN
    ].isna().any():
        raise EdgeTableError(
            "Source node IDs cannot contain nulls."
        )

    if dataframe[
        TARGET_NODE_COLUMN
    ].isna().any():
        raise EdgeTableError(
            "Target node IDs cannot contain nulls."
        )

    if (
        dataframe[
            SOURCE_NODE_COLUMN
        ].dtype.kind
        not in "iu"
    ):
        raise EdgeTableError(
            "Source node IDs must be integers."
        )

    if (
        dataframe[
            TARGET_NODE_COLUMN
        ].dtype.kind
        not in "iu"
    ):
        raise EdgeTableError(
            "Target node IDs must be integers."
        )

    if dataframe.duplicated().any():
        raise EdgeTableError(
            "Edge table contains duplicate edges."
        )


def build_all_edge_tables(
    transactions: pd.DataFrame,
    customer_devices: pd.DataFrame,
    customer_ips: pd.DataFrame,
    customer_merchants: pd.DataFrame,
    customer_nodes: NodeTable,
    account_nodes: NodeTable,
    card_nodes: NodeTable,
    transaction_nodes: NodeTable,
    merchant_nodes: NodeTable,
    device_nodes: NodeTable,
    ip_nodes: NodeTable,
) -> dict[str, EdgeTable]:
    """Build all canonical graph edge tables."""

    edge_tables = {
        CUSTOMER_ACCOUNT: build_customer_account_edges(
            transactions[
                ["customer_id", "account_id"]
            ],
            customer_nodes,
            account_nodes,
        ),
        ACCOUNT_CARD: build_account_card_edges(
            transactions[
                ["account_id", "card_id"]
            ],
            account_nodes,
            card_nodes,
        ),
        CUSTOMER_TRANSACTION: (
            build_customer_transaction_edges(
                transactions,
                customer_nodes,
                transaction_nodes,
            )
        ),
        ACCOUNT_TRANSACTION: (
            build_account_transaction_edges(
                transactions,
                account_nodes,
                transaction_nodes,
            )
        ),
        CARD_TRANSACTION: (
            build_card_transaction_edges(
                transactions,
                card_nodes,
                transaction_nodes,
            )
        ),
        CUSTOMER_DEVICE: build_customer_device_edges(
            customer_devices,
            customer_nodes,
            device_nodes,
        ),
        CUSTOMER_IP: build_customer_ip_edges(
            customer_ips,
            customer_nodes,
            ip_nodes,
        ),
        CUSTOMER_MERCHANT: (
            build_customer_merchant_edges(
                customer_merchants,
                customer_nodes,
                merchant_nodes,
            )
        ),
        TRANSACTION_MERCHANT: (
            build_transaction_merchant_edges(
                transactions,
                transaction_nodes,
                merchant_nodes,
            )
        ),
        TRANSACTION_DEVICE: (
            build_transaction_device_edges(
                transactions,
                transaction_nodes,
                device_nodes,
            )
        ),
        TRANSACTION_IP: (
            build_transaction_ip_edges(
                transactions,
                transaction_nodes,
                ip_nodes,
            )
        ),
    }

    for edge_table in edge_tables.values():
        validate_edge_table(edge_table)

    return edge_tables