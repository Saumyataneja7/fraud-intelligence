from __future__ import annotations

import pytest

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
    EDGE_CONTRACTS,
    EDGE_TYPES,
    EdgeTypeContract,
    GraphContractError,
    IP,
    MERCHANT,
    NODE_CONTRACTS,
    NODE_TYPES,
    TRANSACTION,
    TRANSACTION_DEVICE,
    TRANSACTION_IP,
    TRANSACTION_MERCHANT,
    NodeTypeContract,
    validate_graph_contracts,
)


def test_node_types_are_unique() -> None:
    assert len(NODE_TYPES) == len(set(NODE_TYPES))


def test_edge_types_are_unique() -> None:
    assert len(EDGE_TYPES) == len(set(EDGE_TYPES))


def test_all_node_contracts_validate() -> None:
    for contract in NODE_CONTRACTS:
        contract.validate()


def test_all_edge_contracts_validate() -> None:
    for contract in EDGE_CONTRACTS:
        contract.validate()


def test_global_graph_contract_validates() -> None:
    validate_graph_contracts()


def test_expected_node_types_exist() -> None:
    assert NODE_TYPES == (
        CUSTOMER,
        ACCOUNT,
        CARD,
        TRANSACTION,
        MERCHANT,
        DEVICE,
        IP,
    )


def test_expected_edge_types_exist() -> None:
    assert EDGE_TYPES == (
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


def test_invalid_node_type_raises() -> None:
    contract = NodeTypeContract(
        node_type="unknown",
        source_id_column="customer_id",
    )

    with pytest.raises(
        GraphContractError,
        match="Unsupported node type",
    ):
        contract.validate()


def test_empty_node_id_column_raises() -> None:
    contract = NodeTypeContract(
        node_type=CUSTOMER,
        source_id_column="",
    )

    with pytest.raises(
        GraphContractError,
        match="source_id_column",
    ):
        contract.validate()


def test_invalid_edge_type_raises() -> None:
    contract = EdgeTypeContract(
        edge_type="unknown",
        source_node_type=CUSTOMER,
        target_node_type=DEVICE,
        source_id_column="customer_id",
        target_id_column="device_id",
    )

    with pytest.raises(
        GraphContractError,
        match="Unsupported edge type",
    ):
        contract.validate()


def test_invalid_source_node_type_raises() -> None:
    contract = EdgeTypeContract(
        edge_type=CUSTOMER_DEVICE,
        source_node_type="unknown",
        target_node_type=DEVICE,
        source_id_column="customer_id",
        target_id_column="device_id",
    )

    with pytest.raises(
        GraphContractError,
        match="Unsupported source node type",
    ):
        contract.validate()


def test_invalid_target_node_type_raises() -> None:
    contract = EdgeTypeContract(
        edge_type=CUSTOMER_DEVICE,
        source_node_type=CUSTOMER,
        target_node_type="unknown",
        source_id_column="customer_id",
        target_id_column="device_id",
    )

    with pytest.raises(
        GraphContractError,
        match="Unsupported target node type",
    ):
        contract.validate()


def test_empty_source_id_column_raises() -> None:
    contract = EdgeTypeContract(
        edge_type=CUSTOMER_DEVICE,
        source_node_type=CUSTOMER,
        target_node_type=DEVICE,
        source_id_column="",
        target_id_column="device_id",
    )

    with pytest.raises(
        GraphContractError,
        match="source_id_column",
    ):
        contract.validate()


def test_empty_target_id_column_raises() -> None:
    contract = EdgeTypeContract(
        edge_type=CUSTOMER_DEVICE,
        source_node_type=CUSTOMER,
        target_node_type=DEVICE,
        source_id_column="customer_id",
        target_id_column="",
    )

    with pytest.raises(
        GraphContractError,
        match="target_id_column",
    ):
        contract.validate()