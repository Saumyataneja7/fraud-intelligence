from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from fraud_intelligence.graph.nodes import (
    NODE_ID_COLUMN,
    NODE_SOURCE_COLUMNS,
    NodeTable,
)


class GraphIDMappingError(ValueError):
    """Raised when graph ID mapping is invalid."""


SOURCE_ID_COLUMN: Final[str] = "source_id"
GRAPH_ID_COLUMN: Final[str] = "node_id"


@dataclass(frozen=True)
class GraphIDMapping:
    """Bidirectional mapping between business IDs and graph node IDs."""

    node_type: str
    source_id_column: str
    mapping: pd.DataFrame

    @property
    def node_count(self) -> int:
        return len(self.mapping)

    def source_to_graph(
        self,
        source_ids: pd.Series,
    ) -> pd.Series:
        """Map business/entity IDs to graph node IDs."""

        lookup = self.mapping.set_index(
            SOURCE_ID_COLUMN
        )[GRAPH_ID_COLUMN]

        result = source_ids.map(lookup)

        if result.isna().any():
            missing = (
                source_ids[result.isna()]
                .drop_duplicates()
                .tolist()
            )

            raise GraphIDMappingError(
                f"Unknown {self.node_type} source IDs: "
                f"{missing[:10]}"
            )

        return result.astype("int64")

    def graph_to_source(
        self,
        graph_ids: pd.Series,
    ) -> pd.Series:
        """Map graph node IDs back to business/entity IDs."""

        lookup = self.mapping.set_index(
            GRAPH_ID_COLUMN
        )[SOURCE_ID_COLUMN]

        result = graph_ids.map(lookup)

        if result.isna().any():
            missing = (
                graph_ids[result.isna()]
                .drop_duplicates()
                .tolist()
            )

            raise GraphIDMappingError(
                f"Unknown {self.node_type} graph IDs: "
                f"{missing[:10]}"
            )

        return result


def build_graph_id_mapping(
    node_table: NodeTable,
) -> GraphIDMapping:
    """
    Build a deterministic business-ID ↔ graph-ID mapping
    from a validated NodeTable.
    """

    if not isinstance(
        node_table,
        NodeTable,
    ):
        raise GraphIDMappingError(
            "node_table must be a NodeTable."
        )

    if node_table.node_type not in NODE_SOURCE_COLUMNS:
        raise GraphIDMappingError(
            f"Unsupported node type: "
            f"{node_table.node_type}"
        )

    dataframe = node_table.dataframe

    source_column = node_table.source_id_column

    required_columns = {
        source_column,
        NODE_ID_COLUMN,
    }

    missing = required_columns - set(
        dataframe.columns
    )

    if missing:
        raise GraphIDMappingError(
            "Node table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    mapping = (
        dataframe[
            [
                source_column,
                NODE_ID_COLUMN,
            ]
        ]
        .rename(
            columns={
                source_column: SOURCE_ID_COLUMN,
                NODE_ID_COLUMN: GRAPH_ID_COLUMN,
            }
        )
        .copy()
    )

    # Deterministic ordering.
    mapping = (
        mapping
        .sort_values(
            GRAPH_ID_COLUMN,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    result = GraphIDMapping(
        node_type=node_table.node_type,
        source_id_column=source_column,
        mapping=mapping,
    )

    validate_graph_id_mapping(result)

    return result


def validate_graph_id_mapping(
    graph_mapping: GraphIDMapping,
) -> None:
    """Validate mapping integrity."""

    if not isinstance(
        graph_mapping,
        GraphIDMapping,
    ):
        raise GraphIDMappingError(
            "graph_mapping must be a GraphIDMapping."
        )

    mapping = graph_mapping.mapping

    required_columns = {
        SOURCE_ID_COLUMN,
        GRAPH_ID_COLUMN,
    }

    missing = required_columns - set(
        mapping.columns
    )

    if missing:
        raise GraphIDMappingError(
            "Mapping is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if mapping.empty:
        raise GraphIDMappingError(
            "Graph ID mapping cannot be empty."
        )

    if mapping[
        SOURCE_ID_COLUMN
    ].isna().any():
        raise GraphIDMappingError(
            "Source IDs cannot contain nulls."
        )

    if mapping[
        GRAPH_ID_COLUMN
    ].isna().any():
        raise GraphIDMappingError(
            "Graph IDs cannot contain nulls."
        )

    if mapping[
        SOURCE_ID_COLUMN
    ].duplicated().any():
        raise GraphIDMappingError(
            "Source IDs must be unique."
        )

    if mapping[
        GRAPH_ID_COLUMN
    ].duplicated().any():
        raise GraphIDMappingError(
            "Graph IDs must be unique."
        )

    graph_ids = mapping[
        GRAPH_ID_COLUMN
    ]

    if graph_ids.dtype.kind not in "iu":
        raise GraphIDMappingError(
            "Graph IDs must be integers."
        )

    expected_ids = pd.Series(
        range(len(mapping)),
        dtype="int64",
    )

    actual_ids = (
        graph_ids
        .reset_index(drop=True)
        .astype("int64")
    )

    if not actual_ids.equals(expected_ids):
        raise GraphIDMappingError(
            "Graph IDs must be contiguous and "
            "zero-based."
        )


def build_all_graph_id_mappings(
    node_tables: dict[str, NodeTable],
) -> dict[str, GraphIDMapping]:
    """Build mappings for every graph node type."""

    if not node_tables:
        raise GraphIDMappingError(
            "node_tables cannot be empty."
        )

    mappings: dict[
        str,
        GraphIDMapping,
    ] = {}

    for node_type, node_table in node_tables.items():
        if node_type != node_table.node_type:
            raise GraphIDMappingError(
                "Node table dictionary key does not "
                "match NodeTable.node_type: "
                f"{node_type}"
            )

        mappings[node_type] = (
            build_graph_id_mapping(
                node_table
            )
        )

    return mappings


def validate_all_graph_id_mappings(
    mappings: dict[str, GraphIDMapping],
) -> None:
    """Validate all graph ID mappings."""

    if not mappings:
        raise GraphIDMappingError(
            "mappings cannot be empty."
        )

    for node_type, mapping in mappings.items():
        if node_type != mapping.node_type:
            raise GraphIDMappingError(
                "Mapping dictionary key does not "
                "match mapping.node_type: "
                f"{node_type}"
            )

        validate_graph_id_mapping(mapping)