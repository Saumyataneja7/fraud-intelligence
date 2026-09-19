from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

from fraud_intelligence.graph.contracts import (
    NODE_TYPES,
    TRANSACTION,
)


class GraphMLContractError(ValueError):
    """Raised when the Graph ML data contract is violated."""


GRAPH_ML_TASK: Final[str] = "transaction_fraud_classification"

TARGET_NODE_TYPE: Final[str] = TRANSACTION
TARGET_COLUMN: Final[str] = "is_fraud"

FORBIDDEN_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "is_fraud",
    "fraud_scenario",
    "node_id",
    "transaction_id",
    "customer_id",
    "account_id",
    "card_id",
    "merchant_id",
    "device_id",
    "ip_id",
    "timestamp",
)

TEMPORAL_CUTOFF_RULE: Final[str] = (
    "Only information strictly earlier than the prediction "
    "timestamp may be used for historical graph context."
)

SAME_TIMESTAMP_RULE: Final[str] = (
    "Events at the same timestamp are excluded from historical "
    "context because no explicit event ordering is available."
)


@dataclass(frozen=True)
class GraphMLDataContract:
    """Canonical contract for the Phase 7 Graph ML dataset."""

    task: str = GRAPH_ML_TASK
    target_node_type: str = TARGET_NODE_TYPE
    target_column: str = TARGET_COLUMN
    node_types: tuple[str, ...] = NODE_TYPES
    forbidden_feature_columns: tuple[str, ...] = (
        FORBIDDEN_FEATURE_COLUMNS
    )
    temporal_cutoff_rule: str = TEMPORAL_CUTOFF_RULE
    same_timestamp_rule: str = SAME_TIMESTAMP_RULE

    def validate_node_types(
        self,
        node_types: Iterable[str],
    ) -> None:
        """Validate the graph contains exactly the canonical nodes."""

        actual = tuple(node_types)

        if set(actual) != set(self.node_types):
            missing = set(self.node_types) - set(actual)
            unknown = set(actual) - set(self.node_types)

            messages: list[str] = []

            if missing:
                messages.append(
                    "missing node types: "
                    + ", ".join(sorted(missing))
                )

            if unknown:
                messages.append(
                    "unknown node types: "
                    + ", ".join(sorted(unknown))
                )

            raise GraphMLContractError(
                "Invalid Graph ML node schema: "
                + "; ".join(messages)
            )

    def validate_target_node_type(
        self,
        node_type: str,
    ) -> None:
        """Validate the prediction node type."""

        if node_type != self.target_node_type:
            raise GraphMLContractError(
                "Graph ML target node type must be "
                f"{self.target_node_type!r}, got {node_type!r}."
            )

    def validate_target_column(
        self,
        columns: Iterable[str],
    ) -> None:
        """Validate the fraud target exists."""

        columns = set(columns)

        if self.target_column not in columns:
            raise GraphMLContractError(
                "Missing required Graph ML target column: "
                f"{self.target_column}"
            )

    def validate_feature_columns(
        self,
        feature_columns: Iterable[str],
    ) -> None:
        """Reject targets, identifiers, timestamps, and forbidden fields."""

        features = tuple(feature_columns)

        duplicates = {
            column
            for column in features
            if features.count(column) > 1
        }

        if duplicates:
            raise GraphMLContractError(
                "Feature columns contain duplicates: "
                + ", ".join(sorted(duplicates))
            )

        forbidden = (
            set(features)
            & set(self.forbidden_feature_columns)
        )

        if forbidden:
            raise GraphMLContractError(
                "Forbidden Graph ML feature columns detected: "
                + ", ".join(sorted(forbidden))
            )

    def validate_target_not_in_features(
        self,
        feature_columns: Iterable[str],
    ) -> None:
        """Explicitly prevent target leakage."""

        features = set(feature_columns)

        if self.target_column in features:
            raise GraphMLContractError(
                f"Target column {self.target_column!r} "
                "cannot be used as a Graph ML feature."
            )

    def validate_temporal_requirements(
        self,
        timestamp_column: str,
    ) -> None:
        """Require a timestamp for temporal graph controls."""

        if not timestamp_column:
            raise GraphMLContractError(
                "Graph ML requires a transaction timestamp "
                "for temporal leakage controls."
            )

    def validate_all(
        self,
        *,
        node_types: Iterable[str],
        target_node_type: str,
        target_columns: Iterable[str],
        feature_columns: Iterable[str],
        timestamp_column: str,
    ) -> None:
        """Run the complete Graph ML contract validation."""

        self.validate_node_types(node_types)

        self.validate_target_node_type(
            target_node_type
        )

        self.validate_target_column(
            target_columns
        )

        self.validate_feature_columns(
            feature_columns
        )

        self.validate_target_not_in_features(
            feature_columns
        )

        self.validate_temporal_requirements(
            timestamp_column
        )