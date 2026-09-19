import pytest

from fraud_intelligence.explainability.contracts import (
    ExplainabilityContractError,
    ExplainabilityDataContract,
)


@pytest.fixture
def contract():
    return ExplainabilityDataContract()


def test_contract_is_deterministic():
    first = ExplainabilityDataContract()
    second = ExplainabilityDataContract()

    assert first == second


def test_task():
    contract = ExplainabilityDataContract()

    contract.validate_task(
        "transaction_fraud_explanation"
    )


def test_invalid_task_rejected():
    contract = ExplainabilityDataContract()

    with pytest.raises(ExplainabilityContractError):
        contract.validate_task(
            "customer_fraud_explanation"
        )


def test_target_node_type():
    contract = ExplainabilityDataContract()

    contract.validate_target_node_type(
        "transaction"
    )


def test_invalid_target_node_type_rejected():
    contract = ExplainabilityDataContract()

    with pytest.raises(ExplainabilityContractError):
        contract.validate_target_node_type(
            "customer"
        )


def test_target_column():
    contract = ExplainabilityDataContract()

    contract.validate_target_column(
        "is_fraud"
    )


def test_invalid_target_column_rejected():
    contract = ExplainabilityDataContract()

    with pytest.raises(ExplainabilityContractError):
        contract.validate_target_column(
            "fraud_scenario"
        )


def test_required_transaction_columns():
    contract = ExplainabilityDataContract()

    contract.validate_transaction_columns(
        [
            "transaction_id",
            "timestamp",
            "amount",
        ]
    )


def test_missing_transaction_id_rejected():
    contract = ExplainabilityDataContract()

    with pytest.raises(ExplainabilityContractError):
        contract.validate_transaction_columns(
            ["timestamp"]
        )


def test_missing_timestamp_rejected():
    contract = ExplainabilityDataContract()

    with pytest.raises(ExplainabilityContractError):
        contract.validate_transaction_columns(
            ["transaction_id"]
        )


def test_target_columns_forbidden_from_explanation():
    contract = ExplainabilityDataContract()

    with pytest.raises(ExplainabilityContractError):
        contract.validate_explanation_columns(
            [
                "amount",
                "is_fraud",
            ]
        )


def test_fraud_scenario_forbidden_from_explanation():
    contract = ExplainabilityDataContract()

    with pytest.raises(ExplainabilityContractError):
        contract.validate_explanation_columns(
            [
                "amount",
                "fraud_scenario",
            ]
        )


def test_valid_explanation_columns():
    contract = ExplainabilityDataContract()

    contract.validate_explanation_columns(
        [
            "amount",
            "customer_txn_count_5m",
            "is_new_device",
        ]
    )


def test_future_context_rejected():
    contract = ExplainabilityDataContract()

    with pytest.raises(ExplainabilityContractError):
        contract.validate_future_context(True)


def test_historical_context_allowed():
    contract = ExplainabilityDataContract()

    contract.validate_future_context(False)


def test_valid_explanation_components():
    contract = ExplainabilityDataContract()

    contract.validate_explanation_components(
        [
            "prediction",
            "feature_attribution",
            "graph_context",
            "evidence",
        ]
    )


def test_unknown_explanation_component_rejected():
    contract = ExplainabilityDataContract()

    with pytest.raises(ExplainabilityContractError):
        contract.validate_explanation_components(
            [
                "prediction",
                "unknown_component",
            ]
        )


def test_full_contract_validation():
    contract = ExplainabilityDataContract()

    contract.validate_all(
        task="transaction_fraud_explanation",
        node_type="transaction",
        target_column="is_fraud",
        transaction_columns=[
            "transaction_id",
            "timestamp",
            "amount",
        ],
        explanation_columns=[
            "amount",
            "customer_txn_count_5m",
            "is_new_device",
        ],
        uses_future_context=False,
        components=[
            "prediction",
            "feature_attribution",
            "graph_context",
            "evidence",
        ],
    )


def test_full_contract_rejects_future_context():
    contract = ExplainabilityDataContract()

    with pytest.raises(ExplainabilityContractError):
        contract.validate_all(
            task="transaction_fraud_explanation",
            node_type="transaction",
            target_column="is_fraud",
            transaction_columns=[
                "transaction_id",
                "timestamp",
            ],
            explanation_columns=[
                "amount",
            ],
            uses_future_context=True,
            components=[
                "prediction",
                "feature_attribution",
            ],
        )


def test_contract_constants():
    contract = ExplainabilityDataContract()

    assert contract.TARGET_NODE_TYPE == "transaction"
    assert contract.TARGET_COLUMN == "is_fraud"

    assert contract.FORBIDDEN_EXPLANATION_COLUMNS == (
        "is_fraud",
        "fraud_scenario",
    )

    assert contract.EXPLANATION_COMPONENTS == (
        "prediction",
        "feature_attribution",
        "graph_context",
        "evidence",
    )