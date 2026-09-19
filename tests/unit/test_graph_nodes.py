from __future__ import annotations

import pandas as pd
import pytest

from fraud_intelligence.graph.nodes import (
    NODE_ID_COLUMN,
    NodeTable,
    NodeTableError,
    build_account_nodes,
    build_all_node_tables,
    build_card_nodes,
    build_customer_nodes,
    build_device_nodes,
    build_ip_nodes,
    build_merchant_nodes,
    build_transaction_nodes,
    validate_node_table,
    validate_node_table_counts,
)


def test_customer_nodes_are_deterministic() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": [
                "C003",
                "C001",
                "C002",
            ]
        }
    )

    result = build_customer_nodes(
        dataframe
    )

    assert result.dataframe[
        "customer_id"
    ].tolist() == [
        "C001",
        "C002",
        "C003",
    ]

    assert result.dataframe[
        NODE_ID_COLUMN
    ].tolist() == [0, 1, 2]


def test_account_nodes() -> None:
    dataframe = pd.DataFrame(
        {
            "account_id": [
                "A002",
                "A001",
            ]
        }
    )

    result = build_account_nodes(
        dataframe
    )

    assert result.node_count == 2
    assert result.dataframe[
        "account_id"
    ].tolist() == [
        "A001",
        "A002",
    ]


def test_card_nodes() -> None:
    dataframe = pd.DataFrame(
        {
            "card_id": [
                "CARD002",
                "CARD001",
            ]
        }
    )

    result = build_card_nodes(
        dataframe
    )

    assert result.node_count == 2


def test_merchant_nodes() -> None:
    dataframe = pd.DataFrame(
        {
            "merchant_id": [
                "M002",
                "M001",
            ]
        }
    )

    result = build_merchant_nodes(
        dataframe
    )

    assert result.node_count == 2


def test_device_nodes() -> None:
    dataframe = pd.DataFrame(
        {
            "device_id": [
                "D002",
                "D001",
            ]
        }
    )

    result = build_device_nodes(
        dataframe
    )

    assert result.node_count == 2


def test_ip_nodes() -> None:
    dataframe = pd.DataFrame(
        {
            "ip_id": [
                "IP002",
                "IP001",
            ]
        }
    )

    result = build_ip_nodes(
        dataframe
    )

    assert result.node_count == 2


def test_transaction_nodes_preserve_required_attributes() -> None:
    dataframe = pd.DataFrame(
        {
            "transaction_id": [
                "T002",
                "T001",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-02 10:00:00",
                    "2025-01-01 10:00:00",
                ],
                utc=True,
            ),
            "amount": [
                200.0,
                100.0,
            ],
            "currency": [
                "USD",
                "USD",
            ],
            "payment_method": [
                "card",
                "wallet",
            ],
            "transaction_type": [
                "purchase",
                "payment",
            ],
            "is_fraud": [
                1,
                0,
            ],
            "fraud_scenario": [
                "ACCOUNT_TAKEOVER",
                "NONE",
            ],
        }
    )

    result = build_transaction_nodes(
        dataframe
    )

    assert result.node_count == 2

    assert set(
        [
            "transaction_id",
            "timestamp",
            "amount",
            "currency",
            "payment_method",
            "transaction_type",
            "is_fraud",
            "fraud_scenario",
        ]
    ).issubset(
        result.dataframe.columns
    )

    assert result.dataframe[
        "transaction_id"
    ].tolist() == [
        "T001",
        "T002",
    ]

    assert result.dataframe[
        NODE_ID_COLUMN
    ].tolist() == [0, 1]


def test_duplicate_source_ids_raise() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C001",
            ]
        }
    )

    with pytest.raises(
        NodeTableError,
        match="duplicates",
    ):
        build_customer_nodes(
            dataframe
        )


def test_null_source_ids_raise() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                None,
            ]
        }
    )

    with pytest.raises(
        NodeTableError,
        match="null",
    ):
        build_customer_nodes(
            dataframe
        )


def test_empty_dataframe_raises() -> None:
    dataframe = pd.DataFrame(
        columns=["customer_id"]
    )

    with pytest.raises(
        NodeTableError,
        match="empty",
    ):
        build_customer_nodes(
            dataframe
        )


def test_missing_source_column_raises() -> None:
    dataframe = pd.DataFrame(
        {
            "wrong_column": [
                "C001"
            ]
        }
    )

    with pytest.raises(
        NodeTableError,
        match="Missing required source ID",
    ):
        build_customer_nodes(
            dataframe
        )


def test_invalid_node_type_cannot_validate() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": [
                "C001"
            ]
        }
    )

    result = build_customer_nodes(
        dataframe
    )

    invalid = NodeTable(
        node_type="invalid",
        dataframe=result.dataframe,
        source_id_column="customer_id",
    )

    with pytest.raises(
        NodeTableError,
        match="Unsupported node type",
    ):
        validate_node_table(
            invalid
        )


def test_non_contiguous_node_ids_raise() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
            ]
        }
    )

    result = build_customer_nodes(
        dataframe
    )

    invalid_dataframe = result.dataframe.copy()

    invalid_dataframe[
        NODE_ID_COLUMN
    ] = [0, 2]

    invalid = NodeTable(
        node_type=result.node_type,
        dataframe=invalid_dataframe,
        source_id_column=result.source_id_column,
    )

    with pytest.raises(
        NodeTableError,
        match="contiguous",
    ):
        validate_node_table(
            invalid
        )


def test_all_node_tables_are_built() -> None:
    customers = pd.DataFrame(
        {
            "customer_id": ["C001"]
        }
    )

    accounts = pd.DataFrame(
        {
            "account_id": ["A001"]
        }
    )

    cards = pd.DataFrame(
        {
            "card_id": ["CARD001"]
        }
    )

    transactions = pd.DataFrame(
        {
            "transaction_id": ["T001"],
            "timestamp": pd.to_datetime(
                ["2025-01-01 10:00:00"],
                utc=True,
            ),
            "amount": [100.0],
            "currency": ["USD"],
            "payment_method": ["card"],
            "transaction_type": ["purchase"],
            "is_fraud": [0],
            "fraud_scenario": ["NONE"],
        }
    )

    merchants = pd.DataFrame(
        {
            "merchant_id": ["M001"]
        }
    )

    devices = pd.DataFrame(
        {
            "device_id": ["D001"]
        }
    )

    ips = pd.DataFrame(
        {
            "ip_id": ["IP001"]
        }
    )

    node_tables = build_all_node_tables(
        customers=customers,
        accounts=accounts,
        cards=cards,
        transactions=transactions,
        merchants=merchants,
        devices=devices,
        ips=ips,
    )

    assert set(
        node_tables
    ) == {
        "customer",
        "account",
        "card",
        "transaction",
        "merchant",
        "device",
        "ip",
    }

    assert all(
        table.node_count == 1
        for table in node_tables.values()
    )


def test_node_table_counts_validate() -> None:
    customers = pd.DataFrame(
        {
            "customer_id": ["C001"]
        }
    )

    accounts = pd.DataFrame(
        {
            "account_id": ["A001"]
        }
    )

    cards = pd.DataFrame(
        {
            "card_id": ["CARD001"]
        }
    )

    transactions = pd.DataFrame(
        {
            "transaction_id": ["T001"],
            "timestamp": pd.to_datetime(
                ["2025-01-01 10:00:00"],
                utc=True,
            ),
            "amount": [100.0],
            "currency": ["USD"],
            "payment_method": ["card"],
            "transaction_type": ["purchase"],
            "is_fraud": [0],
            "fraud_scenario": ["NONE"],
        }
    )

    merchants = pd.DataFrame(
        {
            "merchant_id": ["M001"]
        }
    )

    devices = pd.DataFrame(
        {
            "device_id": ["D001"]
        }
    )

    ips = pd.DataFrame(
        {
            "ip_id": ["IP001"]
        }
    )

    node_tables = build_all_node_tables(
        customers,
        accounts,
        cards,
        transactions,
        merchants,
        devices,
        ips,
    )

    validate_node_table_counts(
        node_tables
    )