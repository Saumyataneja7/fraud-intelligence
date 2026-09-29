from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


class FraudIntelligenceContractError(ValueError):
    """Raised when fraud intelligence contract validation fails."""


@dataclass(frozen=True)
class FraudIntelligenceDataContract:
    """
    Contract for investigation-oriented fraud intelligence.

    This contract defines what a fraud intelligence response may
    contain and what information must remain excluded from
    predictive/investigative evidence.
    """

    TASK: ClassVar[str] = "fraud_intelligence_investigation"

    TARGET_TRANSACTION_NODE_TYPE: ClassVar[str] = "transaction"

    ALLOWED_ENTITY_TYPES: ClassVar[tuple[str, ...]] = (
        "customer",
        "account",
        "card",
        "transaction",
        "merchant",
        "device",
        "ip",
    )

    ALLOWED_RELATION_TYPES: ClassVar[tuple[str, ...]] = (
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

    INTELLIGENCE_COMPONENTS: ClassVar[tuple[str, ...]] = (
        "transaction_intelligence",
        "entity_intelligence",
        "network_intelligence",
        "fraud_ring",
        "evidence",
    )

    FORBIDDEN_PREDICTIVE_FIELDS: ClassVar[tuple[str, ...]] = (
        "is_fraud",
        "fraud_scenario",
    )

    FORBIDDEN_FUTURE_CONTEXT: ClassVar[bool] = True

    TEMPORAL_RULE: ClassVar[str] = (
        "Investigation context used for predictive or temporal "
        "analysis must not include information from the future "
        "relative to the target transaction."
    )

    def validate_task(
        self,
        task: str,
    ) -> None:
        if task != self.TASK:
            raise FraudIntelligenceContractError(
                f"Invalid intelligence task: {task!r}. "
                f"Expected {self.TASK!r}."
            )

    def validate_transaction_node_type(
        self,
        node_type: str,
    ) -> None:
        if node_type != self.TARGET_TRANSACTION_NODE_TYPE:
            raise FraudIntelligenceContractError(
                "Fraud intelligence target node type must be "
                f"{self.TARGET_TRANSACTION_NODE_TYPE!r}."
            )

    def validate_entity_type(
        self,
        entity_type: str,
    ) -> None:
        if entity_type not in self.ALLOWED_ENTITY_TYPES:
            raise FraudIntelligenceContractError(
                f"Unsupported entity type: {entity_type!r}."
            )

    def validate_relation_type(
        self,
        relation_type: str,
    ) -> None:
        if relation_type not in self.ALLOWED_RELATION_TYPES:
            raise FraudIntelligenceContractError(
                f"Unsupported relation type: {relation_type!r}."
            )

    def validate_intelligence_component(
        self,
        component: str,
    ) -> None:
        if component not in self.INTELLIGENCE_COMPONENTS:
            raise FraudIntelligenceContractError(
                f"Unsupported intelligence component: "
                f"{component!r}."
            )

    def validate_predictive_fields(
        self,
        fields: tuple[str, ...] | list[str],
    ) -> None:
        forbidden = sorted(
            set(fields).intersection(
                self.FORBIDDEN_PREDICTIVE_FIELDS
            )
        )

        if forbidden:
            raise FraudIntelligenceContractError(
                "Forbidden predictive fields detected: "
                + ", ".join(forbidden)
            )

    def validate_future_context_rule(
        self,
        future_context_allowed: bool,
    ) -> None:
        if (
            self.FORBIDDEN_FUTURE_CONTEXT
            and future_context_allowed
        ):
            raise FraudIntelligenceContractError(
                "Future context cannot be used for "
                "fraud intelligence analysis."
            )

    def validate_all(
        self,
        *,
        task: str,
        target_node_type: str,
        entity_types: tuple[str, ...] | list[str],
        relation_types: tuple[str, ...] | list[str],
        intelligence_components: tuple[str, ...] | list[str],
        predictive_fields: tuple[str, ...] | list[str],
        future_context_allowed: bool,
    ) -> None:
        self.validate_task(task)
        self.validate_transaction_node_type(
            target_node_type
        )

        for entity_type in entity_types:
            self.validate_entity_type(entity_type)

        for relation_type in relation_types:
            self.validate_relation_type(relation_type)

        for component in intelligence_components:
            self.validate_intelligence_component(component)

        self.validate_predictive_fields(
            predictive_fields
        )

        self.validate_future_context_rule(
            future_context_allowed
        )