from __future__ import annotations

import pandas as pd
import pytest

from fraud_intelligence.graph.contracts import (
    ACCOUNT,
    ACCOUNT_CARD,
    CARD,
    CUSTOMER,
    CUSTOMER_ACCOUNT,
    CUSTOMER_DEVICE,
    DEVICE,
)
from fraud_intelligence.graph.edges import (
    EdgeTable,
    EdgeTableError,
    SOURCE_NODE_COLUMN,
    TARGET_NODE_COLUMN,
    build_account_card_edges,
    build_customer_account_edges,
    build_customer_device_edges,
    validate_edge_table,
)
from fraud_intelligence.graph.nodes import (
    build_account_nodes,
    build_card_nodes,
    build_customer_nodes,
    build_device_nodes,
)


def test_customer_account_edges_are_mapped() -> None:
    customers = build_customer_nodes(
        pd.DataFrame(
            {
                "customer_id": [
                    "C002",
                    "C001",
                ]
            }
        )
    )

    accounts = build_account_nodes(
        pd.DataFrame(
            {
                "account_id": [
                    "A002",
                    "A001",
                ]
            }
        )
    )

    relationships = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
            ],
            "account_id": [
                "A001",
                "A002",
            ],
        }
    )

    result = build_customer_account_edges(
        relationships,
        customers,
        accounts,
    )

    assert result.edge_type == CUSTOMER_ACCOUNT
    assert (
        result.source_node_type
        == CUSTOMER
    )
    assert (
        result.target_node_type
        == ACCOUNT
    )

    assert result.dataframe[
        SOURCE_NODE_COLUMN
    ].tolist() == [0, 1]

    assert result.dataframe[
        TARGET_NODE_COLUMN
    ].tolist() == [0, 1]


def test_duplicate_relationships_are_deduplicated() -> None:
    customers = build_customer_nodes(
        pd.DataFrame(
            {
                "customer_id": [
                    "C001"
                ]
            }
        )
    )

    accounts = build_account_nodes(
        pd.DataFrame(
            {
                "account_id": [
                    "A001"
                ]
            }
        )
    )

    relationships = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C001",
                "C001",
            ],
            "account_id": [
                "A001",
                "A001",
                "A001",
            ],
        }
    )

    result = build_customer_account_edges(
        relationships,
        customers,
        accounts,
    )

    assert result.edge_count == 1


def test_account_card_edges() -> None:
    accounts = build_account_nodes(
        pd.DataFrame(
            {
                "account_id": [
                    "A001"
                ]
            }
        )
    )

    cards = build_card_nodes(
        pd.DataFrame(
            {
                "card_id": [
                    "CARD001"
                ]
            }
        )
    )

    relationships = pd.DataFrame(
        {
            "account_id": [
                "A001"
            ],
            "card_id": [
                "CARD001"
            ],
        }
    )

    result = build_account_card_edges(
        relationships,
        accounts,
        cards,
    )

    assert result.edge_type == ACCOUNT_CARD
    assert result.edge_count == 1


def test_customer_device_edges() -> None:
    customers = build_customer_nodes(
        pd.DataFrame(
            {
                "customer_id": [
                    "C001"
                ]
            }
        )
    )

    devices = build_device_nodes(
        pd.DataFrame(
            {
                "device_id": [
                    "D001"
                ]
            }
        )
    )

    relationships = pd.DataFrame(
        {
            "customer_id": [
                "C001"
            ],
            "device_id": [
                "D001"
            ],
        }
    )

    result = build_customer_device_edges(
        relationships,
        customers,
        devices,
    )

    assert result.edge_type == CUSTOMER_DEVICE
    assert result.edge_count == 1


def test_missing_source_id_raises() -> None:
    customers = build_customer_nodes(
        pd.DataFrame(
            {
                "customer_id": [
                    "C001"
                ]
            }
        )
    )

    accounts = build_account_nodes(
        pd.DataFrame(
            {
                "account_id": [
                    "A001"
                ]
            }
        )
    )

    relationships = pd.DataFrame(
        {
            "customer_id": [
                "C001"
            ]
        }
    )

    with pytest.raises(
        EdgeTableError,
        match="missing required",
    ):
        build_customer_account_edges(
            relationships,
            customers,
            accounts,
        )


def test_unknown_entity_id_raises() -> None:
    customers = build_customer_nodes(
        pd.DataFrame(
            {
                "customer_id": [
                    "C001"
                ]
            }
        )
    )

    accounts = build_account_nodes(
        pd.DataFrame(
            {
                "account_id": [
                    "A001"
                ]
            }
        )
    )

    relationships = pd.DataFrame(
        {
            "customer_id": [
                "C999"
            ],
            "account_id": [
                "A001"
            ],
        }
    )

    with pytest.raises(
        EdgeTableError,
        match="Unable to map",
    ):
        build_customer_account_edges(
            relationships,
            customers,
            accounts,
        )


def test_invalid_edge_type_raises() -> None:
    with pytest.raises(
        EdgeTableError,
        match="Unsupported edge type",
    ):
        validate_edge_table(
            EdgeTable(
                edge_type="unknown",
                source_node_type=CUSTOMER,
                target_node_type=DEVICE,
                dataframe=pd.DataFrame(
                    {
                        SOURCE_NODE_COLUMN: [0],
                        TARGET_NODE_COLUMN: [0],
                    }
                ),
            )
        )


def test_duplicate_edges_cannot_exist_after_validation() -> None:
    edge_table = EdgeTable(
        edge_type=CUSTOMER_ACCOUNT,
        source_node_type=CUSTOMER,
        target_node_type=ACCOUNT,
        dataframe=pd.DataFrame(
            {
                SOURCE_NODE_COLUMN: [
                    0,
                    0,
                ],
                TARGET_NODE_COLUMN: [
                    0,
                    0,
                ],
            }
        ),
    )

    with pytest.raises(
        EdgeTableError,
        match="duplicate",
    ):
        validate_edge_table(
            edge_table
        )


def test_edge_ids_are_integer() -> None:
    edge_table = EdgeTable(
        edge_type=CUSTOMER_ACCOUNT,
        source_node_type=CUSTOMER,
        target_node_type=ACCOUNT,
        dataframe=pd.DataFrame(
            {
                SOURCE_NODE_COLUMN: [
                    "0"
                ],
                TARGET_NODE_COLUMN: [
                    0
                ],
            }
        ),
    )

    with pytest.raises(
        EdgeTableError,
        match="integers",
    ):
        validate_edge_table(
            edge_table
        )


def test_null_edge_ids_raise() -> None:
    edge_table = EdgeTable(
        edge_type=CUSTOMER_ACCOUNT,
        source_node_type=CUSTOMER,
        target_node_type=ACCOUNT,
        dataframe=pd.DataFrame(
            {
                SOURCE_NODE_COLUMN: [
                    None
                ],
                TARGET_NODE_COLUMN: [
                    0
                ],
            }
        ),
    )

    with pytest.raises(
        EdgeTableError,
        match="nulls",
    ):
        validate_edge_table(
            edge_table
        )