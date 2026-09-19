from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
import torch

from fraud_intelligence.graph.contracts import TRANSACTION
from fraud_intelligence.graph.nodes import NODE_ID_COLUMN


TRANSACTION_ID_COLUMN: Final[str] = "transaction_id"
TIMESTAMP_COLUMN: Final[str] = "timestamp"
TARGET_COLUMN: Final[str] = "is_fraud"
SCENARIO_COLUMN: Final[str] = "fraud_scenario"


class GraphLabelPreparationError(ValueError):
    """Raised when transaction graph label preparation fails."""


@dataclass(frozen=True)
class TransactionNodeLabels:
    """
    Prepared labels aligned with transaction graph nodes.

    `labels` contains only the binary fraud target. Fraud scenario is
    retained separately as post-hoc diagnostic metadata.
    """

    node_type: str
    node_ids: np.ndarray
    transaction_ids: np.ndarray
    timestamps: pd.Series
    labels: torch.Tensor
    fraud_scenarios: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.labels.shape[0])

    @property
    def positive_count(self) -> int:
        return int(self.labels.sum().item())

    @property
    def negative_count(self) -> int:
        return self.node_count - self.positive_count

    @property
    def fraud_rate(self) -> float:
        if self.node_count == 0:
            return 0.0

        return self.positive_count / self.node_count


def _validate_feature_dataset(
    feature_dataset: pd.DataFrame,
) -> None:
    if not isinstance(feature_dataset, pd.DataFrame):
        raise GraphLabelPreparationError(
            "feature_dataset must be a pandas DataFrame."
        )

    if feature_dataset.empty:
        raise GraphLabelPreparationError(
            "feature_dataset must not be empty."
        )

    required_columns = {
        TRANSACTION_ID_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        SCENARIO_COLUMN,
    }

    missing = required_columns.difference(feature_dataset.columns)

    if missing:
        raise GraphLabelPreparationError(
            f"Missing required columns: {sorted(missing)}"
        )

    if feature_dataset[TRANSACTION_ID_COLUMN].isna().any():
        raise GraphLabelPreparationError(
            "transaction_id must not contain null values."
        )

    if feature_dataset[TRANSACTION_ID_COLUMN].duplicated().any():
        raise GraphLabelPreparationError(
            "transaction_id must be unique."
        )

    if feature_dataset[TIMESTAMP_COLUMN].isna().any():
        raise GraphLabelPreparationError(
            "timestamp must not contain null values."
        )

    if not pd.api.types.is_datetime64_any_dtype(
        feature_dataset[TIMESTAMP_COLUMN]
    ):
        raise GraphLabelPreparationError(
            "timestamp must be a datetime column."
        )


def _validate_transaction_nodes(
    transaction_nodes: pd.DataFrame,
) -> None:
    if not isinstance(transaction_nodes, pd.DataFrame):
        raise GraphLabelPreparationError(
            "transaction_nodes must be a pandas DataFrame."
        )

    if transaction_nodes.empty:
        raise GraphLabelPreparationError(
            "transaction_nodes must not be empty."
        )

    required_columns = {
        NODE_ID_COLUMN,
        TRANSACTION_ID_COLUMN,
    }

    missing = required_columns.difference(transaction_nodes.columns)

    if missing:
        raise GraphLabelPreparationError(
            f"Missing transaction-node columns: {sorted(missing)}"
        )

    if transaction_nodes[NODE_ID_COLUMN].isna().any():
        raise GraphLabelPreparationError(
            "node_id must not contain null values."
        )

    if transaction_nodes[TRANSACTION_ID_COLUMN].isna().any():
        raise GraphLabelPreparationError(
            "transaction_id must not contain null values."
        )

    if transaction_nodes[NODE_ID_COLUMN].duplicated().any():
        raise GraphLabelPreparationError(
            "node_id must be unique."
        )

    if transaction_nodes[TRANSACTION_ID_COLUMN].duplicated().any():
        raise GraphLabelPreparationError(
            "transaction-node transaction_id values must be unique."
        )

    if not pd.api.types.is_integer_dtype(
        transaction_nodes[NODE_ID_COLUMN]
    ):
        raise GraphLabelPreparationError(
            "node_id must be integer typed."
        )


def _validate_binary_labels(
    labels: pd.Series,
) -> None:
    if labels.isna().any():
        raise GraphLabelPreparationError(
            "is_fraud must not contain null values."
        )

    values = set(labels.unique().tolist())

    if not values.issubset({0, 1}):
        raise GraphLabelPreparationError(
            "is_fraud must contain only binary values {0, 1}. "
            f"Found: {sorted(values)}"
        )


def prepare_transaction_node_labels(
    feature_dataset: pd.DataFrame,
    transaction_nodes: pd.DataFrame,
) -> TransactionNodeLabels:
    """
    Align transaction fraud labels with graph transaction nodes.
    """

    _validate_feature_dataset(feature_dataset)
    _validate_transaction_nodes(transaction_nodes)

    _validate_binary_labels(
        feature_dataset[TARGET_COLUMN]
    )

    feature_transaction_ids = set(
        feature_dataset[TRANSACTION_ID_COLUMN].tolist()
    )

    node_transaction_ids = set(
        transaction_nodes[TRANSACTION_ID_COLUMN].tolist()
    )

    if feature_transaction_ids != node_transaction_ids:
        missing_from_nodes = feature_transaction_ids - node_transaction_ids
        missing_from_features = node_transaction_ids - feature_transaction_ids

        raise GraphLabelPreparationError(
            "Label/node transaction coverage mismatch. "
            f"Transactions missing from graph nodes: "
            f"{sorted(missing_from_nodes)}; "
            f"Transactions missing from feature dataset: "
            f"{sorted(missing_from_features)}"
        )

    merged = transaction_nodes[
        [
            NODE_ID_COLUMN,
            TRANSACTION_ID_COLUMN,
        ]
    ].merge(
        feature_dataset[
            [
                TRANSACTION_ID_COLUMN,
                TIMESTAMP_COLUMN,
                TARGET_COLUMN,
                SCENARIO_COLUMN,
            ]
        ],
        on=TRANSACTION_ID_COLUMN,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(merged) != len(transaction_nodes):
        raise GraphLabelPreparationError(
            "Label/node alignment changed the transaction-node count."
        )

    if merged[TARGET_COLUMN].isna().any():
        raise GraphLabelPreparationError(
            "Missing fraud labels after transaction-node alignment."
        )

    merged = merged.sort_values(
        NODE_ID_COLUMN,
        kind="mergesort",
    ).reset_index(drop=True)

    label_array = merged[TARGET_COLUMN].to_numpy(
        dtype=np.int64,
    )

    if not np.isin(label_array, [0, 1]).all():
        raise GraphLabelPreparationError(
            "Aligned fraud labels must contain only 0 and 1."
        )

    return TransactionNodeLabels(
        node_type=TRANSACTION,
        node_ids=merged[NODE_ID_COLUMN].to_numpy(
            dtype=np.int64,
        ),
        transaction_ids=merged[TRANSACTION_ID_COLUMN].to_numpy(),
        timestamps=merged[TIMESTAMP_COLUMN].copy(),
        labels=torch.from_numpy(label_array).to(
            dtype=torch.long,
        ),
        fraud_scenarios=merged[SCENARIO_COLUMN].to_numpy(),
    )


def validate_transaction_node_labels(
    label_set: TransactionNodeLabels,
) -> None:
    """
    Validate a prepared transaction-node label set.
    """

    if not isinstance(
        label_set,
        TransactionNodeLabels,
    ):
        raise GraphLabelPreparationError(
            "label_set must be a TransactionNodeLabels instance."
        )

    if label_set.node_type != TRANSACTION:
        raise GraphLabelPreparationError(
            f"Expected node type '{TRANSACTION}', "
            f"got '{label_set.node_type}'."
        )

    if label_set.labels.ndim != 1:
        raise GraphLabelPreparationError(
            "labels must be a 1-dimensional tensor."
        )

    if label_set.labels.dtype != torch.long:
        raise GraphLabelPreparationError(
            "labels must use torch.long."
        )

    if label_set.labels.shape[0] != len(label_set.node_ids):
        raise GraphLabelPreparationError(
            "Label count must match node_ids."
        )

    if label_set.labels.shape[0] != len(
        label_set.transaction_ids
    ):
        raise GraphLabelPreparationError(
            "Label count must match transaction_ids."
        )

    if label_set.labels.shape[0] != len(
        label_set.timestamps
    ):
        raise GraphLabelPreparationError(
            "Label count must match timestamps."
        )

    if label_set.labels.shape[0] != len(
        label_set.fraud_scenarios
    ):
        raise GraphLabelPreparationError(
            "Label count must match fraud_scenarios."
        )

    unique_labels = set(
        label_set.labels.tolist()
    )

    if not unique_labels.issubset({0, 1}):
        raise GraphLabelPreparationError(
            "labels must contain only 0 and 1."
        )

    if len(np.unique(label_set.node_ids)) != len(
        label_set.node_ids
    ):
        raise GraphLabelPreparationError(
            "node_ids must be unique."
        )

    if len(np.unique(label_set.transaction_ids)) != len(
        label_set.transaction_ids
    ):
        raise GraphLabelPreparationError(
            "transaction_ids must be unique."
        )


def align_labels_to_split(
    label_set: TransactionNodeLabels,
    split_transactions: pd.DataFrame,
) -> TransactionNodeLabels:
    """
    Select transaction-node labels belonging to one temporal split.

    Original graph node IDs are preserved.
    """

    if not isinstance(split_transactions, pd.DataFrame):
        raise GraphLabelPreparationError(
            "split_transactions must be a pandas DataFrame."
        )

    required_columns = {
        TRANSACTION_ID_COLUMN,
        TIMESTAMP_COLUMN,
    }

    missing = required_columns.difference(
        split_transactions.columns
    )

    if missing:
        raise GraphLabelPreparationError(
            f"Missing split columns: {sorted(missing)}"
        )

    if split_transactions[
        TRANSACTION_ID_COLUMN
    ].duplicated().any():
        raise GraphLabelPreparationError(
            "split transaction_id values must be unique."
        )

    split_ids = set(
        split_transactions[TRANSACTION_ID_COLUMN].tolist()
    )

    selected_indices = np.flatnonzero(
        np.isin(
            label_set.transaction_ids,
            list(split_ids),
        )
    )

    selected = TransactionNodeLabels(
        node_type=label_set.node_type,
        node_ids=label_set.node_ids[selected_indices],
        transaction_ids=label_set.transaction_ids[
            selected_indices
        ],
        timestamps=label_set.timestamps.iloc[
            selected_indices
        ].copy(),
        labels=label_set.labels[selected_indices],
        fraud_scenarios=label_set.fraud_scenarios[
            selected_indices
        ],
    )

    validate_transaction_node_labels(selected)

    expected_ids = set(
        split_transactions[TRANSACTION_ID_COLUMN].tolist()
    )

    actual_ids = set(
        selected.transaction_ids.tolist()
    )

    if actual_ids != expected_ids:
        missing_ids = expected_ids - actual_ids
        extra_ids = actual_ids - expected_ids

        raise GraphLabelPreparationError(
            "Split/label alignment mismatch. "
            f"Missing IDs: {sorted(missing_ids)}; "
            f"Extra IDs: {sorted(extra_ids)}"
        )

    return selected