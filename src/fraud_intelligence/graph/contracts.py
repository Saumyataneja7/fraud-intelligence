from __future__ import annotations

from dataclasses import dataclass
from typing import Final


class GraphContractError(ValueError):
    """Raised when graph construction contracts are violated."""


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

CUSTOMER: Final[str] = "customer"
ACCOUNT: Final[str] = "account"
CARD: Final[str] = "card"
TRANSACTION: Final[str] = "transaction"
MERCHANT: Final[str] = "merchant"
DEVICE: Final[str] = "device"
IP: Final[str] = "ip"

NODE_TYPES: Final[tuple[str, ...]] = (
    CUSTOMER,
    ACCOUNT,
    CARD,
    TRANSACTION,
    MERCHANT,
    DEVICE,
    IP,
)


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------

CUSTOMER_ACCOUNT: Final[str] = "customer_account"
ACCOUNT_CARD: Final[str] = "account_card"

CUSTOMER_TRANSACTION: Final[str] = "customer_transaction"
ACCOUNT_TRANSACTION: Final[str] = "account_transaction"
CARD_TRANSACTION: Final[str] = "card_transaction"

CUSTOMER_DEVICE: Final[str] = "customer_device"
CUSTOMER_IP: Final[str] = "customer_ip"
CUSTOMER_MERCHANT: Final[str] = "customer_merchant"

TRANSACTION_MERCHANT: Final[str] = "transaction_merchant"
TRANSACTION_DEVICE: Final[str] = "transaction_device"
TRANSACTION_IP: Final[str] = "transaction_ip"


EDGE_TYPES: Final[tuple[str, ...]] = (
    CUSTOMER_ACCOUNT,
    ACCOUNT_CARD,
    CUSTOMER_TRANSACTION,
    ACCOUNT_TRANSACTION,
    CARD_TRANSACTION,
    CUSTOMER_DEVICE,
    CUSTOMER_IP,
    CUSTOMER_MERCHANT,
    TRANSACTION_MERCHANT,
    TRANSACTION_DEVICE,
    TRANSACTION_IP,
)


# ---------------------------------------------------------------------------
# Source columns
# ---------------------------------------------------------------------------

ENTITY_ID_COLUMNS: Final[tuple[str, ...]] = (
    "customer_id",
    "account_id",
    "card_id",
    "transaction_id",
    "merchant_id",
    "device_id",
    "ip_id",
)

TRANSACTION_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "transaction_id",
    "customer_id",
    "account_id",
    "card_id",
    "merchant_id",
    "device_id",
    "ip_id",
    "timestamp",
    "amount",
    "currency",
    "payment_method",
    "transaction_type",
    "is_fraud",
    "fraud_scenario",
)


@dataclass(frozen=True)
class NodeTypeContract:
    """Contract describing one graph node type."""

    node_type: str
    source_id_column: str

    def validate(self) -> None:
        if self.node_type not in NODE_TYPES:
            raise GraphContractError(
                f"Unsupported node type: {self.node_type}"
            )

        if not self.source_id_column:
            raise GraphContractError(
                "source_id_column cannot be empty."
            )


@dataclass(frozen=True)
class EdgeTypeContract:
    """Contract describing a typed graph relationship."""

    edge_type: str
    source_node_type: str
    target_node_type: str
    source_id_column: str
    target_id_column: str

    def validate(self) -> None:
        if self.edge_type not in EDGE_TYPES:
            raise GraphContractError(
                f"Unsupported edge type: {self.edge_type}"
            )

        if self.source_node_type not in NODE_TYPES:
            raise GraphContractError(
                f"Unsupported source node type: "
                f"{self.source_node_type}"
            )

        if self.target_node_type not in NODE_TYPES:
            raise GraphContractError(
                f"Unsupported target node type: "
                f"{self.target_node_type}"
            )

        if not self.source_id_column:
            raise GraphContractError(
                "source_id_column cannot be empty."
            )

        if not self.target_id_column:
            raise GraphContractError(
                "target_id_column cannot be empty."
            )


# ---------------------------------------------------------------------------
# Canonical node contracts
# ---------------------------------------------------------------------------

NODE_CONTRACTS: Final[tuple[NodeTypeContract, ...]] = (
    NodeTypeContract(
        node_type=CUSTOMER,
        source_id_column="customer_id",
    ),
    NodeTypeContract(
        node_type=ACCOUNT,
        source_id_column="account_id",
    ),
    NodeTypeContract(
        node_type=CARD,
        source_id_column="card_id",
    ),
    NodeTypeContract(
        node_type=TRANSACTION,
        source_id_column="transaction_id",
    ),
    NodeTypeContract(
        node_type=MERCHANT,
        source_id_column="merchant_id",
    ),
    NodeTypeContract(
        node_type=DEVICE,
        source_id_column="device_id",
    ),
    NodeTypeContract(
        node_type=IP,
        source_id_column="ip_id",
    ),
)


# ---------------------------------------------------------------------------
# Canonical edge contracts
# ---------------------------------------------------------------------------

EDGE_CONTRACTS: Final[tuple[EdgeTypeContract, ...]] = (
    EdgeTypeContract(
        edge_type=CUSTOMER_ACCOUNT,
        source_node_type=CUSTOMER,
        target_node_type=ACCOUNT,
        source_id_column="customer_id",
        target_id_column="account_id",
    ),
    EdgeTypeContract(
        edge_type=ACCOUNT_CARD,
        source_node_type=ACCOUNT,
        target_node_type=CARD,
        source_id_column="account_id",
        target_id_column="card_id",
    ),
    EdgeTypeContract(
        edge_type=CUSTOMER_TRANSACTION,
        source_node_type=CUSTOMER,
        target_node_type=TRANSACTION,
        source_id_column="customer_id",
        target_id_column="transaction_id",
    ),
    EdgeTypeContract(
        edge_type=ACCOUNT_TRANSACTION,
        source_node_type=ACCOUNT,
        target_node_type=TRANSACTION,
        source_id_column="account_id",
        target_id_column="transaction_id",
    ),
    EdgeTypeContract(
        edge_type=CARD_TRANSACTION,
        source_node_type=CARD,
        target_node_type=TRANSACTION,
        source_id_column="card_id",
        target_id_column="transaction_id",
    ),
    EdgeTypeContract(
        edge_type=CUSTOMER_DEVICE,
        source_node_type=CUSTOMER,
        target_node_type=DEVICE,
        source_id_column="customer_id",
        target_id_column="device_id",
    ),
    EdgeTypeContract(
        edge_type=CUSTOMER_IP,
        source_node_type=CUSTOMER,
        target_node_type=IP,
        source_id_column="customer_id",
        target_id_column="ip_id",
    ),
    EdgeTypeContract(
        edge_type=CUSTOMER_MERCHANT,
        source_node_type=CUSTOMER,
        target_node_type=MERCHANT,
        source_id_column="customer_id",
        target_id_column="merchant_id",
    ),
    EdgeTypeContract(
        edge_type=TRANSACTION_MERCHANT,
        source_node_type=TRANSACTION,
        target_node_type=MERCHANT,
        source_id_column="transaction_id",
        target_id_column="merchant_id",
    ),
    EdgeTypeContract(
        edge_type=TRANSACTION_DEVICE,
        source_node_type=TRANSACTION,
        target_node_type=DEVICE,
        source_id_column="transaction_id",
        target_id_column="device_id",
    ),
    EdgeTypeContract(
        edge_type=TRANSACTION_IP,
        source_node_type=TRANSACTION,
        target_node_type=IP,
        source_id_column="transaction_id",
        target_id_column="ip_id",
    ),
)


def validate_graph_contracts() -> None:
    """Validate all canonical node and edge contracts."""

    if len(NODE_TYPES) != len(set(NODE_TYPES)):
        raise GraphContractError(
            "NODE_TYPES contains duplicates."
        )

    if len(EDGE_TYPES) != len(set(EDGE_TYPES)):
        raise GraphContractError(
            "EDGE_TYPES contains duplicates."
        )

    for contract in NODE_CONTRACTS:
        contract.validate()

    for contract in EDGE_CONTRACTS:
        contract.validate()