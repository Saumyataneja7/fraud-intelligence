from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
import torch

from fraud_intelligence.features.contracts import FeatureContract
from fraud_intelligence.graph.contracts import TRANSACTION
from fraud_intelligence.graph.nodes import NODE_ID_COLUMN


TRANSACTION_ID_COLUMN: Final[str] = "transaction_id"
TIMESTAMP_COLUMN: Final[str] = "timestamp"

TARGET_COLUMN: Final[str] = "is_fraud"
SCENARIO_COLUMN: Final[str] = "fraud_scenario"

EXPECTED_NODE_ID_COLUMN: Final[str] = NODE_ID_COLUMN


class GraphFeaturePreparationError(ValueError):
    """Raised when graph ML feature preparation fails."""


@dataclass(frozen=True)
class GraphFeatureSet:
    """
    Prepared transaction-node features for graph ML.

    The feature matrix contains only approved numerical model features.
    Target and identifier columns are retained separately for downstream
    label construction and traceability.
    """

    node_type: str
    node_ids: np.ndarray
    transaction_ids: np.ndarray
    timestamps: pd.Series
    features: torch.Tensor
    feature_columns: tuple[str, ...]

    @property
    def node_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.features.shape[1])


def _validate_feature_dataset(
    feature_dataset: pd.DataFrame,
) -> None:
    if not isinstance(feature_dataset, pd.DataFrame):
        raise GraphFeaturePreparationError(
            "feature_dataset must be a pandas DataFrame."
        )

    if feature_dataset.empty:
        raise GraphFeaturePreparationError(
            "feature_dataset must not be empty."
        )

    required_columns = {
        TRANSACTION_ID_COLUMN,
        TIMESTAMP_COLUMN,
    }

    missing = required_columns.difference(feature_dataset.columns)

    if missing:
        raise GraphFeaturePreparationError(
            f"Missing required columns: {sorted(missing)}"
        )

    if feature_dataset[TRANSACTION_ID_COLUMN].isna().any():
        raise GraphFeaturePreparationError(
            "transaction_id must not contain null values."
        )

    if feature_dataset[TRANSACTION_ID_COLUMN].duplicated().any():
        raise GraphFeaturePreparationError(
            "transaction_id must be unique."
        )

    if feature_dataset[TIMESTAMP_COLUMN].isna().any():
        raise GraphFeaturePreparationError(
            "timestamp must not contain null values."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        feature_dataset[TIMESTAMP_COLUMN]
    ):
        raise GraphFeaturePreparationError(
            "timestamp must be a datetime column."
        )


def _validate_graph_transaction_nodes(
    transaction_nodes: pd.DataFrame,
) -> None:
    if not isinstance(transaction_nodes, pd.DataFrame):
        raise GraphFeaturePreparationError(
            "transaction_nodes must be a pandas DataFrame."
        )

    if transaction_nodes.empty:
        raise GraphFeaturePreparationError(
            "transaction_nodes must not be empty."
        )

    required_columns = {
        EXPECTED_NODE_ID_COLUMN,
        TRANSACTION_ID_COLUMN,
    }

    missing = required_columns.difference(transaction_nodes.columns)

    if missing:
        raise GraphFeaturePreparationError(
            f"Missing transaction-node columns: {sorted(missing)}"
        )

    if transaction_nodes[EXPECTED_NODE_ID_COLUMN].isna().any():
        raise GraphFeaturePreparationError(
            "node_id must not contain null values."
        )

    if transaction_nodes[TRANSACTION_ID_COLUMN].isna().any():
        raise GraphFeaturePreparationError(
            "transaction_id must not contain null values."
        )

    if transaction_nodes[TRANSACTION_ID_COLUMN].duplicated().any():
        raise GraphFeaturePreparationError(
            "transaction-node transaction_id values must be unique."
        )

    if transaction_nodes[EXPECTED_NODE_ID_COLUMN].duplicated().any():
        raise GraphFeaturePreparationError(
            "transaction-node node_id values must be unique."
        )

    if not pd.api.types.is_integer_dtype(
        transaction_nodes[EXPECTED_NODE_ID_COLUMN]
    ):
        raise GraphFeaturePreparationError(
            "transaction-node node_id must be integer typed."
        )


def _validate_feature_columns(
    feature_columns: list[str] | tuple[str, ...],
) -> None:
    if not feature_columns:
        raise GraphFeaturePreparationError(
            "At least one feature column is required."
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise GraphFeaturePreparationError(
            "Feature columns must not contain duplicates."
        )

    FeatureContract().validate_feature_columns(feature_columns)

    forbidden = {
        TARGET_COLUMN,
        SCENARIO_COLUMN,
        TRANSACTION_ID_COLUMN,
        TIMESTAMP_COLUMN,
        EXPECTED_NODE_ID_COLUMN,
    }

    invalid = forbidden.intersection(feature_columns)

    if invalid:
        raise GraphFeaturePreparationError(
            f"Forbidden graph ML feature columns: {sorted(invalid)}"
        )


def get_graph_feature_columns(
    feature_dataset: pd.DataFrame,
) -> tuple[str, ...]:
    """
    Return the approved Phase 5 model features.

    This deliberately reuses the existing feature contract rather than
    maintaining a second independent feature list.
    """

    _validate_feature_dataset(feature_dataset)

    feature_columns = [
        column
        for column in feature_dataset.columns
        if column not in {
            TRANSACTION_ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            SCENARIO_COLUMN,
        }
    ]

    _validate_feature_columns(feature_columns)

    return tuple(feature_columns)


def prepare_transaction_node_features(
    feature_dataset: pd.DataFrame,
    transaction_nodes: pd.DataFrame,
) -> GraphFeatureSet:
    """
    Align engineered transaction features with transaction graph nodes.
    """

    _validate_feature_dataset(feature_dataset)
    _validate_graph_transaction_nodes(transaction_nodes)

    feature_columns = get_graph_feature_columns(feature_dataset)

    # ------------------------------------------------------------------
    # Validate exact transaction coverage before alignment.
    # ------------------------------------------------------------------

    feature_transaction_ids = set(
        feature_dataset[TRANSACTION_ID_COLUMN].tolist()
    )

    node_transaction_ids = set(
        transaction_nodes[TRANSACTION_ID_COLUMN].tolist()
    )

    if feature_transaction_ids != node_transaction_ids:
        missing_from_nodes = feature_transaction_ids - node_transaction_ids
        missing_from_features = node_transaction_ids - feature_transaction_ids

        raise GraphFeaturePreparationError(
            "Feature/node transaction coverage mismatch. "
            f"Transactions missing from graph nodes: "
            f"{sorted(missing_from_nodes)}; "
            f"Transactions missing from feature dataset: "
            f"{sorted(missing_from_features)}"
        )

    # ------------------------------------------------------------------
    # Align features to graph transaction nodes.
    # ------------------------------------------------------------------

    merged = transaction_nodes[
        [
            EXPECTED_NODE_ID_COLUMN,
            TRANSACTION_ID_COLUMN,
        ]
    ].merge(
        feature_dataset[
            [
                TRANSACTION_ID_COLUMN,
                TIMESTAMP_COLUMN,
                *feature_columns,
            ]
        ],
        on=TRANSACTION_ID_COLUMN,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(merged) != len(transaction_nodes):
        raise GraphFeaturePreparationError(
            "Feature/node alignment changed the transaction-node count."
        )

    # ------------------------------------------------------------------
    # Validate feature completeness.
    # ------------------------------------------------------------------

    if merged[list(feature_columns)].isna().any().any():
        missing_features = [
            column
            for column in feature_columns
            if merged[column].isna().any()
        ]

        raise GraphFeaturePreparationError(
            "Missing feature values after transaction-node alignment: "
            f"{missing_features}"
        )

    # ------------------------------------------------------------------
    # Preserve the canonical graph node ordering.
    # ------------------------------------------------------------------

    merged = merged.sort_values(
        EXPECTED_NODE_ID_COLUMN,
        kind="mergesort",
    ).reset_index(drop=True)

    feature_array = merged.loc[
        :,
        list(feature_columns),
    ].to_numpy(
        dtype=np.float32,
    )

    if not np.isfinite(feature_array).all():
        raise GraphFeaturePreparationError(
            "Graph features must contain only finite values."
        )

    return GraphFeatureSet(
        node_type=TRANSACTION,
        node_ids=merged[EXPECTED_NODE_ID_COLUMN].to_numpy(
            dtype=np.int64,
        ),
        transaction_ids=merged[TRANSACTION_ID_COLUMN].to_numpy(),
        timestamps=merged[TIMESTAMP_COLUMN].copy(),
        features=torch.from_numpy(feature_array),
        feature_columns=feature_columns,
    )


def validate_transaction_node_features(
    feature_set: GraphFeatureSet,
    expected_feature_columns: tuple[str, ...] | None = None,
) -> None:
    """
    Validate the prepared transaction-node feature matrix.
    """

    if not isinstance(feature_set, GraphFeatureSet):
        raise GraphFeaturePreparationError(
            "feature_set must be a GraphFeatureSet."
        )

    if feature_set.node_type != TRANSACTION:
        raise GraphFeaturePreparationError(
            f"Expected node type '{TRANSACTION}', "
            f"got '{feature_set.node_type}'."
        )

    if feature_set.features.ndim != 2:
        raise GraphFeaturePreparationError(
            "features must be a 2-dimensional tensor."
        )

    if feature_set.features.dtype != torch.float32:
        raise GraphFeaturePreparationError(
            "features must use torch.float32."
        )

    if feature_set.features.shape[0] != len(feature_set.node_ids):
        raise GraphFeaturePreparationError(
            "Feature rows must match node_ids."
        )

    if feature_set.features.shape[0] != len(feature_set.transaction_ids):
        raise GraphFeaturePreparationError(
            "Feature rows must match transaction_ids."
        )

    if feature_set.features.shape[0] != len(feature_set.timestamps):
        raise GraphFeaturePreparationError(
            "Feature rows must match timestamps."
        )

    if not torch.isfinite(feature_set.features).all():
        raise GraphFeaturePreparationError(
            "Graph features must contain only finite values."
        )

    if len(feature_set.feature_columns) != feature_set.feature_count:
        raise GraphFeaturePreparationError(
            "feature_columns count does not match feature matrix width."
        )

    _validate_feature_columns(feature_set.feature_columns)

    if expected_feature_columns is not None:
        if feature_set.feature_columns != expected_feature_columns:
            raise GraphFeaturePreparationError(
                "Prepared feature columns do not match the expected "
                "feature contract."
            )


def align_features_to_split(
    feature_set: GraphFeatureSet,
    split_transactions: pd.DataFrame,
) -> GraphFeatureSet:
    """
    Select transaction-node features belonging to one temporal split.

    Original graph node IDs are preserved. They are never renumbered.
    """

    if not isinstance(split_transactions, pd.DataFrame):
        raise GraphFeaturePreparationError(
            "split_transactions must be a pandas DataFrame."
        )

    required = {
        TRANSACTION_ID_COLUMN,
        TIMESTAMP_COLUMN,
    }

    missing = required.difference(split_transactions.columns)

    if missing:
        raise GraphFeaturePreparationError(
            f"Missing split columns: {sorted(missing)}"
        )

    split_ids = set(
        split_transactions[TRANSACTION_ID_COLUMN].tolist()
    )

    mask = np.array(
        [
            transaction_id in split_ids
            for transaction_id in feature_set.transaction_ids
        ],
        dtype=bool,
    )

    selected_indices = np.flatnonzero(mask)

    selected = GraphFeatureSet(
        node_type=feature_set.node_type,
        node_ids=feature_set.node_ids[selected_indices],
        transaction_ids=feature_set.transaction_ids[selected_indices],
        timestamps=feature_set.timestamps.iloc[selected_indices].copy(),
        features=feature_set.features[selected_indices],
        feature_columns=feature_set.feature_columns,
    )

    validate_transaction_node_features(
        selected,
        expected_feature_columns=feature_set.feature_columns,
    )

    expected_ids = set(
        split_transactions[TRANSACTION_ID_COLUMN].tolist()
    )

    actual_ids = set(selected.transaction_ids.tolist())

    if actual_ids != expected_ids:
        missing_ids = expected_ids - actual_ids
        extra_ids = actual_ids - expected_ids

        raise GraphFeaturePreparationError(
            "Split/feature alignment mismatch. "
            f"Missing IDs: {sorted(missing_ids)}; "
            f"Extra IDs: {sorted(extra_ids)}"
        )

    return selected