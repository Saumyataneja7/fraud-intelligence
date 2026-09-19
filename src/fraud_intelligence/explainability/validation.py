from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fraud_intelligence.explainability.contracts import (
    ExplainabilityDataContract,
    GNNExplanationContract,
)
from fraud_intelligence.explainability.human_readable import (
    HumanReadableExplanation,
    validate_human_readable_explanation,
)
from fraud_intelligence.explainability.transaction import (
    TransactionExplanation,
    validate_transaction_explanation,
)


class ExplanationValidationError(ValueError):
    """Raised when explanation validation fails."""


@dataclass(frozen=True)
class ExplanationLeakageReport:
    """Result of explanation leakage validation."""

    transaction_id: str
    target_timestamp: pd.Timestamp
    forbidden_features_found: tuple[str, ...]
    future_transaction_indices: tuple[int, ...]
    same_timestamp_transaction_indices: tuple[int, ...]
    target_transaction_in_context: bool

    @property
    def has_leakage(self) -> bool:
        return (
            bool(self.forbidden_features_found)
            or bool(self.future_transaction_indices)
            or bool(self.same_timestamp_transaction_indices)
            or self.target_transaction_in_context
        )


FORBIDDEN_EXPLANATION_FIELDS = frozenset(
    {
        "is_fraud",
        "fraud_scenario",
        "transaction_id",
        "customer_id",
        "account_id",
        "card_id",
        "merchant_id",
        "device_id",
        "ip_id",
        "node_id",
        "timestamp",
    }
)


def _validate_target_timestamp(
    timestamp: pd.Timestamp,
) -> None:
    if not isinstance(timestamp, pd.Timestamp):
        raise ExplanationValidationError(
            "target timestamp must be a pandas Timestamp."
        )

    if timestamp.tzinfo is None:
        raise ExplanationValidationError(
            "target timestamp must be timezone-aware."
        )


def validate_explanation_features(
    feature_names: tuple[str, ...] | list[str],
) -> None:
    """
    Ensure explanation features cannot contain targets,
    identifiers, or temporal fields.
    """
    forbidden = sorted(
        set(feature_names).intersection(
            FORBIDDEN_EXPLANATION_FIELDS
        )
    )

    if forbidden:
        raise ExplanationValidationError(
            "Forbidden explanation features detected: "
            + ", ".join(forbidden)
        )


def validate_graph_temporal_context(
    *,
    target_timestamp: pd.Timestamp,
    graph_transaction_timestamps: dict[int, pd.Timestamp],
    target_transaction_node_index: int,
) -> ExplanationLeakageReport:
    """
    Validate that graph transaction context is strictly historical.

    Allowed:
        transaction_timestamp < target_timestamp

    Forbidden:
        transaction_timestamp == target_timestamp
        transaction_timestamp > target_timestamp
        target transaction itself appearing as context
    """
    _validate_target_timestamp(target_timestamp)

    future_indices: list[int] = []
    same_timestamp_indices: list[int] = []

    target_transaction_in_context = (
        target_transaction_node_index
        in graph_transaction_timestamps
    )

    for node_index, timestamp in graph_transaction_timestamps.items():
        if not isinstance(timestamp, pd.Timestamp):
            raise ExplanationValidationError(
                "Graph transaction timestamps must be pandas Timestamps."
            )

        if timestamp.tzinfo is None:
            raise ExplanationValidationError(
                "Graph transaction timestamps must be timezone-aware."
            )

        if node_index == target_transaction_node_index:
            continue

        if timestamp > target_timestamp:
            future_indices.append(node_index)
        elif timestamp == target_timestamp:
            same_timestamp_indices.append(node_index)

    report = ExplanationLeakageReport(
        transaction_id="",
        target_timestamp=target_timestamp,
        forbidden_features_found=(),
        future_transaction_indices=tuple(
            sorted(future_indices)
        ),
        same_timestamp_transaction_indices=tuple(
            sorted(same_timestamp_indices)
        ),
        target_transaction_in_context=target_transaction_in_context,
    )

    if report.has_leakage:
        raise ExplanationValidationError(
            "Graph explanation contains future, same-timestamp, "
            "or target-transaction context."
        )

    return report


def validate_transaction_explanation_leakage(
    explanation: TransactionExplanation,
) -> ExplanationLeakageReport:
    """
    Validate leakage controls for a TransactionExplanation.

    Graph neighborhood extraction is expected to have already applied
    the strict historical filtering from Phase 8.5.
    """
    if not isinstance(
        explanation,
        TransactionExplanation,
    ):
        raise ExplanationValidationError(
            "explanation must be a TransactionExplanation."
        )

    validate_transaction_explanation(explanation)

    validate_explanation_features(
        tuple(
            feature
            for feature in (
                explanation.classical_attribution.to_dataframe()
                .get("feature", [])
                .tolist()
                if explanation.classical_attribution is not None
                else []
            )
        )
    )

    if explanation.shap_explanation is not None:
        validate_explanation_features(
            tuple(
                item.feature
                for item
                in explanation.shap_explanation.feature_attributions
            )
        )

    if explanation.gnn_importance is not None:
        validate_explanation_features(
            tuple(
                item.feature
                for item
                in explanation.gnn_importance.feature_importances
            )
        )

    target_timestamp = explanation.timestamp

    future_indices: list[int] = []
    same_timestamp_indices: list[int] = []
    target_in_context = False

    if explanation.graph_neighborhood is not None:
        neighborhood = explanation.graph_neighborhood

        if (
            neighborhood.transaction_timestamp
            != target_timestamp
        ):
            raise ExplanationValidationError(
                "Graph neighborhood timestamp does not match "
                "target transaction timestamp."
            )

        target_node_count = sum(
            node.node_type == "transaction"
            and node.node_index
            == neighborhood.transaction_node_index
            for node in neighborhood.nodes
        )

        if target_node_count != 1:
            raise ExplanationValidationError(
                "Graph neighborhood must contain exactly one "
                "target transaction node."
            )

    if future_indices:
        raise ExplanationValidationError(
            "Future transaction context detected in explanation."
        )

    if same_timestamp_indices:
        raise ExplanationValidationError(
            "Same-timestamp transaction context detected "
            "in explanation."
        )

    report = ExplanationLeakageReport(
        transaction_id=explanation.transaction_id,
        target_timestamp=target_timestamp,
        forbidden_features_found=(),
        future_transaction_indices=tuple(
            sorted(future_indices)
        ),
        same_timestamp_transaction_indices=tuple(
            sorted(same_timestamp_indices)
        ),
        target_transaction_in_context=target_in_context,
    )

    return report


def validate_human_readable_leakage(
    explanation: HumanReadableExplanation,
) -> None:
    """
    Validate that the human-readable layer does not introduce
    prohibited fields or leakage-sensitive claims.
    """
    validate_human_readable_explanation(explanation)

    forbidden_tokens = (
        "is_fraud",
        "fraud_scenario",
        "customer_id",
        "account_id",
        "card_id",
        "device_id",
        "ip_id",
        "merchant_id",
        "node_id",
    )

    text_fields = [
        explanation.summary,
        *(
            reason.text
            for reason in explanation.reasons
        ),
        *(
            finding.text
            for finding in explanation.graph_findings
        ),
        *explanation.supporting_evidence,
        *explanation.limitations,
    ]

    for text in text_fields:
        lowered = text.lower()

        for token in forbidden_tokens:
            if token.lower() in lowered:
                raise ExplanationValidationError(
                    "Human-readable explanation contains "
                    f"forbidden field: {token}"
                )


def validate_complete_explanation(
    *,
    transaction_explanation: TransactionExplanation,
    human_readable_explanation: HumanReadableExplanation
    | None = None,
    contract: ExplainabilityDataContract | None = None,
    graph_contract: GNNExplanationContract | None = None,
) -> ExplanationLeakageReport:
    """
    Run the complete Phase 8.9 validation suite.

    No model predictions or explanations are recalculated.
    """
    try:
        validate_transaction_explanation(
            transaction_explanation,
            contract=contract,
            graph_contract=graph_contract,
        )
    except Exception as exc:
        raise ExplanationValidationError(
            f"Transaction explanation validation failed: {exc}"
        ) from exc

    report = validate_transaction_explanation_leakage(
        transaction_explanation
    )

    if human_readable_explanation is not None:
        if (
            human_readable_explanation.transaction_id
            != transaction_explanation.transaction_id
        ):
            raise ExplanationValidationError(
                "Human-readable explanation transaction_id "
                "does not match transaction explanation."
            )

        validate_human_readable_leakage(
            human_readable_explanation
        )

    return report