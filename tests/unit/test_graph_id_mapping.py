from __future__ import annotations

import pandas as pd
import pytest

from fraud_intelligence.graph.id_mapping import (
    GRAPH_ID_COLUMN,
    SOURCE_ID_COLUMN,
    GraphIDMapping,
    GraphIDMappingError,
    build_all_graph_id_mappings,
    build_graph_id_mapping,
    validate_graph_id_mapping,
)
from fraud_intelligence.graph.nodes import (
    build_account_nodes,
    build_customer_nodes,
)


def test_build_mapping_from_node_table() -> None:
    nodes = build_customer_nodes(
        pd.DataFrame(
            {
                "customer_id": [
                    "C002",
                    "C001",
                    "C003",
                ]
            }
        )
    )

    mapping = build_graph_id_mapping(nodes)

    assert mapping.node_type == "customer"
    assert mapping.source_id_column == "customer_id"
    assert mapping.node_count == 3

    assert mapping.mapping[
        SOURCE_ID_COLUMN
    ].tolist() == [
        "C001",
        "C002",
        "C003",
    ]

    assert mapping.mapping[
        GRAPH_ID_COLUMN
    ].tolist() == [
        0,
        1,
        2,
    ]


def test_source_to_graph_mapping() -> None:
    nodes = build_customer_nodes(
        pd.DataFrame(
            {
                "customer_id": [
                    "C001",
                    "C002",
                    "C003",
                ]
            }
        )
    )

    mapping = build_graph_id_mapping(nodes)

    result = mapping.source_to_graph(
        pd.Series(
            [
                "C003",
                "C001",
                "C002",
            ]
        )
    )

    assert result.tolist() == [
        2,
        0,
        1,
    ]


def test_graph_to_source_mapping() -> None:
    nodes = build_customer_nodes(
        pd.DataFrame(
            {
                "customer_id": [
                    "C001",
                    "C002",
                    "C003",
                ]
            }
        )
    )

    mapping = build_graph_id_mapping(nodes)

    result = mapping.graph_to_source(
        pd.Series(
            [
                2,
                0,
                1,
            ]
        )
    )

    assert result.tolist() == [
        "C003",
        "C001",
        "C002",
    ]


def test_round_trip_source_to_graph_to_source() -> None:
    nodes = build_customer_nodes(
        pd.DataFrame(
            {
                "customer_id": [
                    "C001",
                    "C002",
                    "C003",
                ]
            }
        )
    )

    mapping = build_graph_id_mapping(nodes)

    source_ids = pd.Series(
        [
            "C003",
            "C001",
        ]
    )

    graph_ids = mapping.source_to_graph(
        source_ids
    )

    restored = mapping.graph_to_source(
        graph_ids
    )

    assert restored.tolist() == (
        source_ids.tolist()
    )


def test_unknown_source_id_raises() -> None:
    nodes = build_customer_nodes(
        pd.DataFrame(
            {
                "customer_id": [
                    "C001",
                ]
            }
        )
    )

    mapping = build_graph_id_mapping(nodes)

    with pytest.raises(
        GraphIDMappingError,
        match="Unknown customer source IDs",
    ):
        mapping.source_to_graph(
            pd.Series(["C999"])
        )


def test_unknown_graph_id_raises() -> None:
    nodes = build_customer_nodes(
        pd.DataFrame(
            {
                "customer_id": [
                    "C001",
                ]
            }
        )
    )

    mapping = build_graph_id_mapping(nodes)

    with pytest.raises(
        GraphIDMappingError,
        match="Unknown customer graph IDs",
    ):
        mapping.graph_to_source(
            pd.Series([999])
        )


def test_duplicate_source_ids_are_rejected() -> None:
    mapping = GraphIDMapping(
        node_type="customer",
        source_id_column="customer_id",
        mapping=pd.DataFrame(
            {
                SOURCE_ID_COLUMN: [
                    "C001",
                    "C001",
                ],
                GRAPH_ID_COLUMN: [
                    0,
                    1,
                ],
            }
        ),
    )

    with pytest.raises(
        GraphIDMappingError,
        match="Source IDs must be unique",
    ):
        validate_graph_id_mapping(
            mapping
        )


def test_duplicate_graph_ids_are_rejected() -> None:
    mapping = GraphIDMapping(
        node_type="customer",
        source_id_column="customer_id",
        mapping=pd.DataFrame(
            {
                SOURCE_ID_COLUMN: [
                    "C001",
                    "C002",
                ],
                GRAPH_ID_COLUMN: [
                    0,
                    0,
                ],
            }
        ),
    )

    with pytest.raises(
        GraphIDMappingError,
        match="Graph IDs must be unique",
    ):
        validate_graph_id_mapping(
            mapping
        )


def test_graph_ids_must_be_zero_based() -> None:
    mapping = GraphIDMapping(
        node_type="customer",
        source_id_column="customer_id",
        mapping=pd.DataFrame(
            {
                SOURCE_ID_COLUMN: [
                    "C001",
                    "C002",
                ],
                GRAPH_ID_COLUMN: [
                    1,
                    2,
                ],
            }
        ),
    )

    with pytest.raises(
        GraphIDMappingError,
        match="contiguous and zero-based",
    ):
        validate_graph_id_mapping(
            mapping
        )


def test_all_mappings_are_created() -> None:
    customers = build_customer_nodes(
        pd.DataFrame(
            {
                "customer_id": [
                    "C001",
                    "C002",
                ]
            }
        )
    )

    accounts = build_account_nodes(
        pd.DataFrame(
            {
                "account_id": [
                    "A001",
                    "A002",
                ]
            }
        )
    )

    mappings = build_all_graph_id_mappings(
        {
            "customer": customers,
            "account": accounts,
        }
    )

    assert set(mappings) == {
        "customer",
        "account",
    }

    assert (
        mappings["customer"].node_count
        == 2
    )

    assert (
        mappings["account"].node_count
        == 2
    )