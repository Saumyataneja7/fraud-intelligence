from __future__ import annotations

from dataclasses import dataclass
from typing import Final

class ExplainabilityContractError(ValueError):
    """Raised when explainability contract validation fails."""


@dataclass(frozen=True)
class ExplainabilityDataContract:
    """Canonical contract for transaction-level explanations."""

    TASK: str = "transaction_fraud_explanation"
    TARGET_NODE_TYPE: str = "transaction"
    TARGET_COLUMN: str = "is_fraud"

    REQUIRED_TRANSACTION_COLUMNS: tuple[str, ...] = (
        "transaction_id",
        "timestamp",
    )

    FORBIDDEN_EXPLANATION_COLUMNS: tuple[str, ...] = (
        "is_fraud",
        "fraud_scenario",
    )

    FORBIDDEN_FUTURE_CONTEXT: bool = True

    EXPLANATION_COMPONENTS: tuple[str, ...] = (
        "prediction",
        "feature_attribution",
        "graph_context",
        "evidence",
    )

    def validate_task(
        self,
        task: str,
    ) -> None:
        if task != self.TASK:
            raise ExplainabilityContractError(
                f"Expected task '{self.TASK}', "
                f"received '{task}'."
            )

    def validate_target_node_type(
        self,
        node_type: str,
    ) -> None:
        if node_type != self.TARGET_NODE_TYPE:
            raise ExplainabilityContractError(
                "Explainability target node type must be "
                f"'{self.TARGET_NODE_TYPE}'."
            )

    def validate_target_column(
        self,
        target_column: str,
    ) -> None:
        if target_column != self.TARGET_COLUMN:
            raise ExplainabilityContractError(
                "Explainability target column must be "
                f"'{self.TARGET_COLUMN}'."
            )

    def validate_transaction_columns(
        self,
        columns: list[str] | tuple[str, ...],
    ) -> None:
        available = set(columns)

        missing = [
            column
            for column in self.REQUIRED_TRANSACTION_COLUMNS
            if column not in available
        ]

        if missing:
            raise ExplainabilityContractError(
                "Missing required transaction columns: "
                f"{missing}."
            )

    def validate_explanation_columns(
        self,
        columns: list[str] | tuple[str, ...],
    ) -> None:
        forbidden = set(self.FORBIDDEN_EXPLANATION_COLUMNS)
        violations = [
            column
            for column in columns
            if column in forbidden
        ]

        if violations:
            raise ExplainabilityContractError(
                "Explanation evidence cannot directly expose "
                f"target-derived columns: {violations}."
            )

    def validate_future_context(
        self,
        uses_future_context: bool,
    ) -> None:
        if (
            self.FORBIDDEN_FUTURE_CONTEXT
            and uses_future_context
        ):
            raise ExplainabilityContractError(
                "Explanations cannot use future transaction "
                "or graph context."
            )

    def validate_explanation_components(
        self,
        components: list[str] | tuple[str, ...],
    ) -> None:
        unknown = [
            component
            for component in components
            if component not in self.EXPLANATION_COMPONENTS
        ]

        if unknown:
            raise ExplainabilityContractError(
                f"Unknown explanation components: {unknown}."
            )

    def validate_all(
        self,
        *,
        task: str,
        node_type: str,
        target_column: str,
        transaction_columns: list[str] | tuple[str, ...],
        explanation_columns: list[str] | tuple[str, ...],
        uses_future_context: bool,
        components: list[str] | tuple[str, ...],
    ) -> None:
        self.validate_task(task)
        self.validate_target_node_type(node_type)
        self.validate_target_column(target_column)
        self.validate_transaction_columns(
            transaction_columns
        )
        self.validate_explanation_columns(
            explanation_columns
        )
        self.validate_future_context(
            uses_future_context
        )
        self.validate_explanation_components(
            components
        )


# ---------------------------------------------------------------------------
# GNN / Graph Explanation Contract
# ---------------------------------------------------------------------------

GRAPH_EXPLANATION_TASK: Final[str] = (
    "transaction_graph_explanation"
)

GRAPH_TARGET_NODE_TYPE: Final[str] = "transaction"

GRAPH_TARGET_COLUMN: Final[str] = "is_fraud"

GRAPH_EXPLANATION_COMPONENTS: Final[tuple[str, ...]] = (
    "target_node",
    "neighbor_nodes",
    "supporting_edges",
    "node_importance",
    "edge_importance",
)

GRAPH_FORBIDDEN_CONTEXT: Final[tuple[str, ...]] = (
    "is_fraud",
    "fraud_scenario",
)

GRAPH_FORBIDDEN_NODE_METADATA: Final[tuple[str, ...]] = (
    "transaction_id",
    "customer_id",
    "account_id",
    "card_id",
    "merchant_id",
    "device_id",
    "ip_id",
)

GRAPH_TEMPORAL_RULE: Final[str] = (
    "Only graph context available strictly before the target "
    "transaction timestamp may be used."
)

GRAPH_ALLOWED_NODE_TYPES: Final[tuple[str, ...]] = (
    "customer",
    "account",
    "card",
    "transaction",
    "merchant",
    "device",
    "ip",
)

GRAPH_ALLOWED_EDGE_TYPES: Final[tuple[str, ...]] = (
    "customer_account",
    "account_card",
    "customer_transaction",
    "account_transaction",
    "card_transaction",
    "customer_device",
    "customer_ip",
    "customer_merchant",
    "transaction_merchant",
    "transaction_device",
    "transaction_ip",
)


class GNNExplanationContractError(ValueError):
    """Raised when a graph explanation violates its contract."""


@dataclass(frozen=True)
class GNNExplanationContract:
    """
    Contract governing explanations produced from graph-based models.

    This contract defines:
    - the explanation task
    - the target node
    - allowed graph entities
    - allowed graph relationships
    - forbidden target/label information
    - temporal leakage requirements
    - required explanation components
    """

    task: str = GRAPH_EXPLANATION_TASK
    target_node_type: str = GRAPH_TARGET_NODE_TYPE
    target_column: str = GRAPH_TARGET_COLUMN

    allowed_node_types: tuple[str, ...] = (
        GRAPH_ALLOWED_NODE_TYPES
    )

    allowed_edge_types: tuple[str, ...] = (
        GRAPH_ALLOWED_EDGE_TYPES
    )

    forbidden_context: tuple[str, ...] = (
        GRAPH_FORBIDDEN_CONTEXT
    )

    forbidden_node_metadata: tuple[str, ...] = (
        GRAPH_FORBIDDEN_NODE_METADATA
    )

    temporal_rule: str = GRAPH_TEMPORAL_RULE

    required_components: tuple[str, ...] = (
        GRAPH_EXPLANATION_COMPONENTS
    )

    max_hops: int = 2

    def validate_task(
        self,
        task: str,
    ) -> None:
        if task != self.task:
            raise GNNExplanationContractError(
                f"Invalid explanation task: {task!r}. "
                f"Expected {self.task!r}."
            )

    def validate_target_node_type(
        self,
        node_type: str,
    ) -> None:
        if node_type != self.target_node_type:
            raise GNNExplanationContractError(
                f"Invalid target node type: {node_type!r}. "
                f"Expected {self.target_node_type!r}."
            )

    def validate_target_column(
        self,
        column: str,
    ) -> None:
        if column != self.target_column:
            raise GNNExplanationContractError(
                f"Invalid target column: {column!r}. "
                f"Expected {self.target_column!r}."
            )

    def validate_node_types(
        self,
        node_types: tuple[str, ...],
    ) -> None:
        unknown = sorted(
            set(node_types)
            - set(self.allowed_node_types)
        )

        if unknown:
            raise GNNExplanationContractError(
                "Unknown graph node types: "
                f"{unknown}."
            )

    def validate_edge_types(
        self,
        edge_types: tuple[str, ...],
    ) -> None:
        unknown = sorted(
            set(edge_types)
            - set(self.allowed_edge_types)
        )

        if unknown:
            raise GNNExplanationContractError(
                "Unknown graph edge types: "
                f"{unknown}."
            )

    def validate_forbidden_context(
        self,
        columns: tuple[str, ...],
    ) -> None:
        forbidden = sorted(
            set(columns)
            & set(self.forbidden_context)
        )

        if forbidden:
            raise GNNExplanationContractError(
                "Graph explanation contains forbidden "
                f"label/context columns: {forbidden}."
            )

    def validate_node_metadata(
        self,
        columns: tuple[str, ...],
    ) -> None:
        forbidden = sorted(
            set(columns)
            & set(self.forbidden_node_metadata)
        )

        if forbidden:
            raise GNNExplanationContractError(
                "Graph explanation contains forbidden "
                f"node metadata: {forbidden}."
            )

    def validate_temporal_context(
        self,
        *,
        target_timestamp: object,
        context_timestamps: tuple[object, ...],
    ) -> None:
        """
        Validate the strict temporal rule.

        The contract requires every graph-context event to occur
        strictly before the target transaction.

        Timestamp comparison is intentionally delegated to the
        caller so this contract remains independent of the
        project's timestamp implementation.
        """
        for timestamp in context_timestamps:
            if timestamp >= target_timestamp:
                raise GNNExplanationContractError(
                    "Graph explanation contains same-time or "
                    "future context. Context timestamps must "
                    "be strictly earlier than the target "
                    "timestamp."
                )

    def validate_hops(
        self,
        hops: int,
    ) -> None:
        if not isinstance(hops, int):
            raise GNNExplanationContractError(
                "hops must be an integer."
            )

        if hops < 0:
            raise GNNExplanationContractError(
                "hops cannot be negative."
            )

        if hops > self.max_hops:
            raise GNNExplanationContractError(
                f"hops cannot exceed {self.max_hops}."
            )

    def validate_components(
        self,
        components: tuple[str, ...],
    ) -> None:
        unknown = sorted(
            set(components)
            - set(self.required_components)
        )

        if unknown:
            raise GNNExplanationContractError(
                "Unknown explanation components: "
                f"{unknown}."
            )

    def validate_all(
        self,
        *,
        task: str,
        target_node_type: str,
        target_column: str,
        node_types: tuple[str, ...],
        edge_types: tuple[str, ...],
        context_columns: tuple[str, ...],
        node_metadata_columns: tuple[str, ...],
        components: tuple[str, ...],
        hops: int,
    ) -> None:
        self.validate_task(task)
        self.validate_target_node_type(
            target_node_type
        )
        self.validate_target_column(
            target_column
        )
        self.validate_node_types(node_types)
        self.validate_edge_types(edge_types)
        self.validate_forbidden_context(
            context_columns
        )
        self.validate_node_metadata(
            node_metadata_columns
        )
        self.validate_components(components)
        self.validate_hops(hops)