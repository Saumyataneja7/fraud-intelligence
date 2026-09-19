from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from fraud_intelligence.graph.contracts import (
    ACCOUNT,
    CARD,
    CUSTOMER,
    DEVICE,
    IP,
    MERCHANT,
    TRANSACTION,
)


class NodeTableError(ValueError):
    """Raised when a graph node table is invalid."""


NODE_ID_COLUMN: Final[str] = "node_id"


NODE_SOURCE_COLUMNS: Final[dict[str, str]] = {
    CUSTOMER: "customer_id",
    ACCOUNT: "account_id",
    CARD: "card_id",
    TRANSACTION: "transaction_id",
    MERCHANT: "merchant_id",
    DEVICE: "device_id",
    IP: "ip_id",
}


TRANSACTION_ATTRIBUTE_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp",
    "amount",
    "currency",
    "payment_method",
    "transaction_type",
    "is_fraud",
    "fraud_scenario",
)


@dataclass(frozen=True)
class NodeTable:
    """Validated node table for one graph node type."""

    node_type: str
    dataframe: pd.DataFrame
    source_id_column: str

    @property
    def node_count(self) -> int:
        return len(self.dataframe)


def _validate_source_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise NodeTableError(
            "Input must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise NodeTableError(
            "Input DataFrame cannot be empty."
        )


def _validate_node_type(
    node_type: str,
) -> None:
    if node_type not in NODE_SOURCE_COLUMNS:
        raise NodeTableError(
            f"Unsupported node type: {node_type}"
        )


def _validate_source_ids(
    dataframe: pd.DataFrame,
    source_id_column: str,
) -> None:
    if source_id_column not in dataframe.columns:
        raise NodeTableError(
            f"Missing required source ID column: "
            f"{source_id_column}"
        )

    ids = dataframe[source_id_column]

    if ids.isna().any():
        raise NodeTableError(
            f"Source ID column contains null values: "
            f"{source_id_column}"
        )

    if ids.duplicated().any():
        raise NodeTableError(
            f"Source ID column contains duplicates: "
            f"{source_id_column}"
        )


def _build_entity_node_table(
    dataframe: pd.DataFrame,
    node_type: str,
) -> NodeTable:
    _validate_source_dataframe(dataframe)
    _validate_node_type(node_type)

    source_id_column = NODE_SOURCE_COLUMNS[
        node_type
    ]

    _validate_source_ids(
        dataframe,
        source_id_column,
    )

    nodes = (
        dataframe[
            [source_id_column]
        ]
        .drop_duplicates()
        .sort_values(
            by=source_id_column,
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    nodes.insert(
        0,
        NODE_ID_COLUMN,
        range(len(nodes)),
    )

    return NodeTable(
        node_type=node_type,
        dataframe=nodes,
        source_id_column=source_id_column,
    )


def build_customer_nodes(
    customers: pd.DataFrame,
) -> NodeTable:
    """Build canonical customer nodes."""

    return _build_entity_node_table(
        customers,
        CUSTOMER,
    )


def build_account_nodes(
    accounts: pd.DataFrame,
) -> NodeTable:
    """Build canonical account nodes."""

    return _build_entity_node_table(
        accounts,
        ACCOUNT,
    )


def build_card_nodes(
    cards: pd.DataFrame,
) -> NodeTable:
    """Build canonical card nodes."""

    return _build_entity_node_table(
        cards,
        CARD,
    )


def build_merchant_nodes(
    merchants: pd.DataFrame,
) -> NodeTable:
    """Build canonical merchant nodes."""

    return _build_entity_node_table(
        merchants,
        MERCHANT,
    )


def build_device_nodes(
    devices: pd.DataFrame,
) -> NodeTable:
    """Build canonical device nodes."""

    return _build_entity_node_table(
        devices,
        DEVICE,
    )


def build_ip_nodes(
    ips: pd.DataFrame,
) -> NodeTable:
    """Build canonical IP nodes."""

    return _build_entity_node_table(
        ips,
        IP,
    )


def build_transaction_nodes(
    transactions: pd.DataFrame,
) -> NodeTable:
    """Build transaction nodes with transaction attributes."""

    _validate_source_dataframe(transactions)
    _validate_node_type(TRANSACTION)

    source_id_column = NODE_SOURCE_COLUMNS[
        TRANSACTION
    ]

    _validate_source_ids(
        transactions,
        source_id_column,
    )

    required_columns = (
        [source_id_column]
        + list(TRANSACTION_ATTRIBUTE_COLUMNS)
    )

    missing = [
        column
        for column in required_columns
        if column not in transactions.columns
    ]

    if missing:
        raise NodeTableError(
            "Transaction data is missing required "
            "columns: "
            + ", ".join(sorted(missing))
        )

    nodes = (
        transactions[
            required_columns
        ]
        .sort_values(
            by=[
                "timestamp",
                source_id_column,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    nodes.insert(
        0,
        NODE_ID_COLUMN,
        range(len(nodes)),
    )

    return NodeTable(
        node_type=TRANSACTION,
        dataframe=nodes,
        source_id_column=source_id_column,
    )


def validate_node_table(
    node_table: NodeTable,
) -> None:
    """Validate a constructed node table."""

    if not isinstance(
        node_table,
        NodeTable,
    ):
        raise NodeTableError(
            "node_table must be a NodeTable."
        )

    _validate_node_type(
        node_table.node_type
    )

    expected_source_id = NODE_SOURCE_COLUMNS[
        node_table.node_type
    ]

    if (
        node_table.source_id_column
        != expected_source_id
    ):
        raise NodeTableError(
            "NodeTable source ID column does not "
            "match its node type."
        )

    dataframe = node_table.dataframe

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise NodeTableError(
            "NodeTable dataframe must be a "
            "pandas DataFrame."
        )

    required = {
        NODE_ID_COLUMN,
        expected_source_id,
    }

    missing = required - set(
        dataframe.columns
    )

    if missing:
        raise NodeTableError(
            "Node table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if dataframe.empty:
        raise NodeTableError(
            "Node table cannot be empty."
        )

    node_ids = dataframe[
        NODE_ID_COLUMN
    ]

    if node_ids.isna().any():
        raise NodeTableError(
            "node_id cannot contain null values."
        )

    if node_ids.duplicated().any():
        raise NodeTableError(
            "node_id values must be unique."
        )

    expected_node_ids = pd.Series(
        range(len(dataframe)),
        index=dataframe.index,
        dtype=node_ids.dtype,
    )

    if not node_ids.reset_index(
        drop=True
    ).equals(
        expected_node_ids.reset_index(
            drop=True
        )
    ):
        raise NodeTableError(
            "node_id values must be contiguous "
            "and start at zero."
        )

    source_ids = dataframe[
        expected_source_id
    ]

    if source_ids.isna().any():
        raise NodeTableError(
            "Source IDs cannot contain null values."
        )

    if source_ids.duplicated().any():
        raise NodeTableError(
            "Source IDs must be unique within "
            "a node table."
        )


def build_all_node_tables(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    cards: pd.DataFrame,
    transactions: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: pd.DataFrame,
    ips: pd.DataFrame,
) -> dict[str, NodeTable]:
    """Build all canonical graph node tables."""

    node_tables = {
        CUSTOMER: build_customer_nodes(
            customers
        ),
        ACCOUNT: build_account_nodes(
            accounts
        ),
        CARD: build_card_nodes(
            cards
        ),
        TRANSACTION: build_transaction_nodes(
            transactions
        ),
        MERCHANT: build_merchant_nodes(
            merchants
        ),
        DEVICE: build_device_nodes(
            devices
        ),
        IP: build_ip_nodes(
            ips
        ),
    }

    for node_table in node_tables.values():
        validate_node_table(node_table)

    return node_tables


def validate_node_table_counts(
    node_tables: dict[str, NodeTable],
) -> None:
    """Ensure every expected node type exists."""

    expected = set(NODE_SOURCE_COLUMNS)
    actual = set(node_tables)

    if actual != expected:
        raise NodeTableError(
            "Node table types do not match the "
            "canonical graph contract."
        )

    for node_type, node_table in node_tables.items():
        if node_table.node_count <= 0:
            raise NodeTableError(
                f"Node table is empty: {node_type}"
            )