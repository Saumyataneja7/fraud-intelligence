from __future__ import annotations

import pytest

from fraud_intelligence.graph.contracts import (
    NODE_TYPES,
    TRANSACTION,
)
from fraud_intelligence.graph_ml.contracts import (
    FORBIDDEN_FEATURE_COLUMNS,
    GRAPH_ML_TASK,
    SAME_TIMESTAMP_RULE,
    TARGET_COLUMN,
    TARGET_NODE_TYPE,
    TEMPORAL_CUTOFF_RULE,
    GraphMLContractError,
    GraphMLDataContract,
)


@pytest.fixture
def contract() -> GraphMLDataContract:
    return GraphMLDataContract()


def test_contract_uses_expected_task(
    contract: GraphMLDataContract,
) -> None:
    assert contract.task == GRAPH_ML_TASK
    assert contract.task == (
        "transaction_fraud_classification"
    )


def test_contract_contains_all_canonical_node_types(
    contract: GraphMLDataContract,
) -> None:
    assert contract.node_types == NODE_TYPES
    assert len(contract.node_types) == 7
    assert set(contract.node_types) == set(
        (
            "customer",
            "account",
            "card",
            "transaction",
            "merchant",
            "device",
            "ip",
        )
    )


def test_transaction_is_prediction_node(
    contract: GraphMLDataContract,
) -> None:
    assert contract.target_node_type == TARGET_NODE_TYPE
    assert contract.target_node_type == TRANSACTION


def test_is_fraud_is_target(
    contract: GraphMLDataContract,
) -> None:
    assert contract.target_column == TARGET_COLUMN
    assert contract.target_column == "is_fraud"


def test_valid_node_schema_is_accepted(
    contract: GraphMLDataContract,
) -> None:
    contract.validate_node_types(NODE_TYPES)


def test_missing_node_type_is_rejected(
    contract: GraphMLDataContract,
) -> None:
    node_types = tuple(
        node
        for node in NODE_TYPES
        if node != "device"
    )

    with pytest.raises(
        GraphMLContractError,
        match="missing node types",
    ):
        contract.validate_node_types(node_types)


def test_unknown_node_type_is_rejected(
    contract: GraphMLDataContract,
) -> None:
    node_types = (
        *NODE_TYPES,
        "unknown",
    )

    with pytest.raises(
        GraphMLContractError,
        match="unknown node types",
    ):
        contract.validate_node_types(node_types)


def test_non_transaction_target_node_is_rejected(
    contract: GraphMLDataContract,
) -> None:
    with pytest.raises(
        GraphMLContractError,
        match="target node type",
    ):
        contract.validate_target_node_type(
            "customer"
        )


def test_missing_target_is_rejected(
    contract: GraphMLDataContract,
) -> None:
    with pytest.raises(
        GraphMLContractError,
        match="Missing required Graph ML target",
    ):
        contract.validate_target_column(
            ["amount", "timestamp"]
        )


def test_is_fraud_cannot_be_feature(
    contract: GraphMLDataContract,
) -> None:
    with pytest.raises(
        GraphMLContractError,
        match="is_fraud",
    ):
        contract.validate_feature_columns(
            ["amount", "is_fraud"]
        )


def test_fraud_scenario_cannot_be_feature(
    contract: GraphMLDataContract,
) -> None:
    with pytest.raises(
        GraphMLContractError,
        match="fraud_scenario",
    ):
        contract.validate_feature_columns(
            ["amount", "fraud_scenario"]
        )


@pytest.mark.parametrize(
    "identifier",
    [
        "node_id",
        "transaction_id",
        "customer_id",
        "account_id",
        "card_id",
        "merchant_id",
        "device_id",
        "ip_id",
    ],
)
def test_graph_identifiers_cannot_be_features(
    contract: GraphMLDataContract,
    identifier: str,
) -> None:
    with pytest.raises(
        GraphMLContractError,
        match=identifier,
    ):
        contract.validate_feature_columns(
            ["amount", identifier]
        )


def test_timestamp_cannot_be_feature(
    contract: GraphMLDataContract,
) -> None:
    with pytest.raises(
        GraphMLContractError,
        match="timestamp",
    ):
        contract.validate_feature_columns(
            ["amount", "timestamp"]
        )


def test_duplicate_features_are_rejected(
    contract: GraphMLDataContract,
) -> None:
    with pytest.raises(
        GraphMLContractError,
        match="duplicates",
    ):
        contract.validate_feature_columns(
            ["amount", "amount"]
        )


def test_valid_features_are_accepted(
    contract: GraphMLDataContract,
) -> None:
    contract.validate_feature_columns(
        [
            "amount_vs_customer_mean",
            "customer_txn_count_5m",
            "is_new_device",
        ]
    )


def test_target_not_in_features(
    contract: GraphMLDataContract,
) -> None:
    contract.validate_target_not_in_features(
        ["amount", "customer_txn_count_5m"]
    )


def test_target_leakage_is_explicitly_rejected(
    contract: GraphMLDataContract,
) -> None:
    with pytest.raises(
        GraphMLContractError,
        match="Target column",
    ):
        contract.validate_target_not_in_features(
            ["amount", "is_fraud"]
        )


def test_timestamp_is_required_for_temporal_controls(
    contract: GraphMLDataContract,
) -> None:
    contract.validate_temporal_requirements(
        "timestamp"
    )


def test_missing_timestamp_is_rejected(
    contract: GraphMLDataContract,
) -> None:
    with pytest.raises(
        GraphMLContractError,
        match="timestamp",
    ):
        contract.validate_temporal_requirements("")


def test_temporal_cutoff_rule_is_strict(
    contract: GraphMLDataContract,
) -> None:
    assert "strictly earlier" in (
        contract.temporal_cutoff_rule
    )

    assert contract.temporal_cutoff_rule == (
        TEMPORAL_CUTOFF_RULE
    )


def test_same_timestamp_rule_is_documented(
    contract: GraphMLDataContract,
) -> None:
    assert "same timestamp" in (
        contract.same_timestamp_rule.lower()
    )

    assert contract.same_timestamp_rule == (
        SAME_TIMESTAMP_RULE
    )


def test_forbidden_feature_contract_is_deterministic(
    contract: GraphMLDataContract,
) -> None:
    assert contract.forbidden_feature_columns == (
        FORBIDDEN_FEATURE_COLUMNS
    )

    assert contract.forbidden_feature_columns == (
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


def test_complete_contract_validation(
    contract: GraphMLDataContract,
) -> None:
    contract.validate_all(
        node_types=NODE_TYPES,
        target_node_type=TRANSACTION,
        target_columns=[
            "is_fraud",
            "fraud_scenario",
        ],
        feature_columns=[
            "amount",
            "amount_vs_customer_mean",
            "customer_txn_count_5m",
            "is_new_device",
        ],
        timestamp_column="timestamp",
    )


def test_complete_contract_rejects_target_leakage(
    contract: GraphMLDataContract,
) -> None:
    with pytest.raises(
        GraphMLContractError,
    ):
        contract.validate_all(
            node_types=NODE_TYPES,
            target_node_type=TRANSACTION,
            target_columns=["is_fraud"],
            feature_columns=[
                "amount",
                "is_fraud",
            ],
            timestamp_column="timestamp",
        )