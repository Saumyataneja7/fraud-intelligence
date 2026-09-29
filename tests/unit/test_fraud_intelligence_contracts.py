from __future__ import annotations

import pytest

from fraud_intelligence.intelligence.contracts import (
    FraudIntelligenceContractError,
    FraudIntelligenceDataContract,
)


@pytest.fixture
def contract() -> FraudIntelligenceDataContract:
    return FraudIntelligenceDataContract()


def test_task_constant(
    contract: FraudIntelligenceDataContract,
) -> None:
    assert (
        contract.TASK
        == "fraud_intelligence_investigation"
    )


def test_transaction_is_target_node(
    contract: FraudIntelligenceDataContract,
) -> None:
    assert (
        contract.TARGET_TRANSACTION_NODE_TYPE
        == "transaction"
    )


def test_allowed_entity_types(
    contract: FraudIntelligenceDataContract,
) -> None:
    assert contract.ALLOWED_ENTITY_TYPES == (
        "customer",
        "account",
        "card",
        "transaction",
        "merchant",
        "device",
        "ip",
    )


def test_allowed_relation_types(
    contract: FraudIntelligenceDataContract,
) -> None:
    assert len(contract.ALLOWED_RELATION_TYPES) == 11

    assert (
        "transaction_device"
        in contract.ALLOWED_RELATION_TYPES
    )

    assert (
        "transaction_ip"
        in contract.ALLOWED_RELATION_TYPES
    )


def test_intelligence_components(
    contract: FraudIntelligenceDataContract,
) -> None:
    assert contract.INTELLIGENCE_COMPONENTS == (
        "transaction_intelligence",
        "entity_intelligence",
        "network_intelligence",
        "fraud_ring",
        "evidence",
    )


def test_forbidden_predictive_fields(
    contract: FraudIntelligenceDataContract,
) -> None:
    assert contract.FORBIDDEN_PREDICTIVE_FIELDS == (
        "is_fraud",
        "fraud_scenario",
    )


def test_future_context_is_forbidden(
    contract: FraudIntelligenceDataContract,
) -> None:
    assert contract.FORBIDDEN_FUTURE_CONTEXT is True


def test_valid_task(
    contract: FraudIntelligenceDataContract,
) -> None:
    contract.validate_task(
        "fraud_intelligence_investigation"
    )


def test_invalid_task_rejected(
    contract: FraudIntelligenceDataContract,
) -> None:
    with pytest.raises(
        FraudIntelligenceContractError,
        match="Invalid intelligence task",
    ):
        contract.validate_task("fraud_prediction")


def test_valid_transaction_node_type(
    contract: FraudIntelligenceDataContract,
) -> None:
    contract.validate_transaction_node_type(
        "transaction"
    )


def test_invalid_transaction_node_type_rejected(
    contract: FraudIntelligenceDataContract,
) -> None:
    with pytest.raises(
        FraudIntelligenceContractError,
        match="target node type",
    ):
        contract.validate_transaction_node_type(
            "customer"
        )


@pytest.mark.parametrize(
    "entity_type",
    [
        "customer",
        "account",
        "card",
        "transaction",
        "merchant",
        "device",
        "ip",
    ],
)
def test_valid_entity_types(
    contract: FraudIntelligenceDataContract,
    entity_type: str,
) -> None:
    contract.validate_entity_type(entity_type)


def test_invalid_entity_type_rejected(
    contract: FraudIntelligenceDataContract,
) -> None:
    with pytest.raises(
        FraudIntelligenceContractError,
        match="Unsupported entity type",
    ):
        contract.validate_entity_type("email")


@pytest.mark.parametrize(
    "relation_type",
    [
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
    ],
)
def test_valid_relation_types(
    contract: FraudIntelligenceDataContract,
    relation_type: str,
) -> None:
    contract.validate_relation_type(relation_type)


def test_invalid_relation_type_rejected(
    contract: FraudIntelligenceDataContract,
) -> None:
    with pytest.raises(
        FraudIntelligenceContractError,
        match="Unsupported relation type",
    ):
        contract.validate_relation_type(
            "customer_email"
        )


@pytest.mark.parametrize(
    "component",
    [
        "transaction_intelligence",
        "entity_intelligence",
        "network_intelligence",
        "fraud_ring",
        "evidence",
    ],
)
def test_valid_intelligence_components(
    contract: FraudIntelligenceDataContract,
    component: str,
) -> None:
    contract.validate_intelligence_component(
        component
    )


def test_invalid_intelligence_component_rejected(
    contract: FraudIntelligenceDataContract,
) -> None:
    with pytest.raises(
        FraudIntelligenceContractError,
        match="Unsupported intelligence component",
    ):
        contract.validate_intelligence_component(
            "prediction"
        )


def test_allowed_predictive_fields_pass(
    contract: FraudIntelligenceDataContract,
) -> None:
    contract.validate_predictive_fields(
        (
            "amount",
            "customer_txn_count_1m",
            "device_txn_count_before",
        )
    )


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "is_fraud",
        "fraud_scenario",
    ],
)
def test_forbidden_predictive_fields_rejected(
    contract: FraudIntelligenceDataContract,
    forbidden_field: str,
) -> None:
    with pytest.raises(
        FraudIntelligenceContractError,
        match="Forbidden predictive fields",
    ):
        contract.validate_predictive_fields(
            (
                "amount",
                forbidden_field,
            )
        )


def test_future_context_false_passes(
    contract: FraudIntelligenceDataContract,
) -> None:
    contract.validate_future_context_rule(False)


def test_future_context_true_rejected(
    contract: FraudIntelligenceDataContract,
) -> None:
    with pytest.raises(
        FraudIntelligenceContractError,
        match="Future context",
    ):
        contract.validate_future_context_rule(True)


def test_validate_all_accepts_valid_contract(
    contract: FraudIntelligenceDataContract,
) -> None:
    contract.validate_all(
        task="fraud_intelligence_investigation",
        target_node_type="transaction",
        entity_types=(
            "customer",
            "account",
            "transaction",
        ),
        relation_types=(
            "customer_transaction",
            "account_transaction",
        ),
        intelligence_components=(
            "transaction_intelligence",
            "network_intelligence",
            "evidence",
        ),
        predictive_fields=(
            "amount",
            "customer_txn_count_1m",
        ),
        future_context_allowed=False,
    )


def test_validate_all_rejects_invalid_contract(
    contract: FraudIntelligenceDataContract,
) -> None:
    with pytest.raises(
        FraudIntelligenceContractError
    ):
        contract.validate_all(
            task="fraud_intelligence_investigation",
            target_node_type="transaction",
            entity_types=("customer",),
            relation_types=("customer_transaction",),
            intelligence_components=("evidence",),
            predictive_fields=("is_fraud",),
            future_context_allowed=False,
        )


def test_temporal_rule_is_documented(
    contract: FraudIntelligenceDataContract,
) -> None:
    assert "future" in contract.TEMPORAL_RULE.lower()
    assert "target transaction" in (
        contract.TEMPORAL_RULE.lower()
    )