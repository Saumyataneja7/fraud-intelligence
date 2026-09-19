import pytest

from fraud_intelligence.explainability.contracts import (
    GRAPH_ALLOWED_EDGE_TYPES,
    GRAPH_ALLOWED_NODE_TYPES,
    GRAPH_EXPLANATION_COMPONENTS,
    GRAPH_EXPLANATION_TASK,
    GRAPH_TARGET_COLUMN,
    GRAPH_TARGET_NODE_TYPE,
    GNNExplanationContract,
    GNNExplanationContractError,
)


@pytest.fixture
def contract() -> GNNExplanationContract:
    return GNNExplanationContract()


def test_default_task(contract):
    assert contract.task == GRAPH_EXPLANATION_TASK


def test_default_target_node(contract):
    assert (
        contract.target_node_type
        == GRAPH_TARGET_NODE_TYPE
    )


def test_default_target_column(contract):
    assert (
        contract.target_column
        == GRAPH_TARGET_COLUMN
    )


def test_allowed_node_types(contract):
    assert contract.allowed_node_types == (
        GRAPH_ALLOWED_NODE_TYPES
    )


def test_allowed_edge_types(contract):
    assert contract.allowed_edge_types == (
        GRAPH_ALLOWED_EDGE_TYPES
    )


def test_required_components(contract):
    assert contract.required_components == (
        GRAPH_EXPLANATION_COMPONENTS
    )


def test_default_max_hops(contract):
    assert contract.max_hops == 2


def test_valid_task(contract):
    contract.validate_task(
        GRAPH_EXPLANATION_TASK
    )


def test_invalid_task(contract):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_task(
            "wrong_task"
        )


def test_valid_target_node_type(contract):
    contract.validate_target_node_type(
        "transaction"
    )


def test_invalid_target_node_type(contract):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_target_node_type(
            "customer"
        )


def test_valid_target_column(contract):
    contract.validate_target_column(
        "is_fraud"
    )


def test_invalid_target_column(contract):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_target_column(
            "fraud_scenario"
        )


def test_valid_node_types(contract):
    contract.validate_node_types(
        (
            "transaction",
            "device",
            "ip",
            "merchant",
        )
    )


def test_unknown_node_type_rejected(contract):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_node_types(
            (
                "transaction",
                "unknown_node",
            )
        )


def test_valid_edge_types(contract):
    contract.validate_edge_types(
        (
            "transaction_device",
            "transaction_ip",
            "transaction_merchant",
        )
    )


def test_unknown_edge_type_rejected(contract):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_edge_types(
            (
                "transaction_device",
                "unknown_edge",
            )
        )


def test_fraud_label_forbidden(contract):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_forbidden_context(
            ("is_fraud",)
        )


def test_fraud_scenario_forbidden(contract):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_forbidden_context(
            ("fraud_scenario",)
        )


def test_normal_context_allowed(contract):
    contract.validate_forbidden_context(
        (
            "amount",
            "customer_txn_count_5m",
            "is_new_device",
        )
    )


@pytest.mark.parametrize(
    "column",
    [
        "transaction_id",
        "customer_id",
        "account_id",
        "card_id",
        "merchant_id",
        "device_id",
        "ip_id",
    ],
)
def test_identifier_metadata_forbidden(
    contract,
    column,
):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_node_metadata(
            (column,)
        )


def test_normal_node_metadata_allowed(contract):
    contract.validate_node_metadata(
        (
            "node_degree",
            "feature_value",
        )
    )


def test_strictly_past_context_allowed(contract):
    contract.validate_temporal_context(
        target_timestamp=10,
        context_timestamps=(
            1,
            5,
            9,
        ),
    )


def test_same_timestamp_context_rejected(contract):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_temporal_context(
            target_timestamp=10,
            context_timestamps=(
                5,
                10,
            ),
        )


def test_future_context_rejected(contract):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_temporal_context(
            target_timestamp=10,
            context_timestamps=(
                5,
                11,
            ),
        )


def test_zero_hops_allowed(contract):
    contract.validate_hops(0)


def test_two_hops_allowed(contract):
    contract.validate_hops(2)


def test_negative_hops_rejected(contract):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_hops(-1)


def test_more_than_two_hops_rejected(contract):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_hops(3)


def test_non_integer_hops_rejected(contract):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_hops(1.5)


def test_valid_components(contract):
    contract.validate_components(
        (
            "target_node",
            "neighbor_nodes",
            "supporting_edges",
        )
    )


def test_unknown_component_rejected(contract):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_components(
            (
                "target_node",
                "unknown_component",
            )
        )


def test_validate_all_success(contract):
    contract.validate_all(
        task=GRAPH_EXPLANATION_TASK,
        target_node_type="transaction",
        target_column="is_fraud",
        node_types=(
            "transaction",
            "device",
            "ip",
        ),
        edge_types=(
            "transaction_device",
            "transaction_ip",
        ),
        context_columns=(
            "amount",
            "is_new_device",
        ),
        node_metadata_columns=(
            "node_degree",
        ),
        components=(
            "target_node",
            "neighbor_nodes",
            "supporting_edges",
        ),
        hops=2,
    )


def test_validate_all_rejects_forbidden_label(
    contract,
):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_all(
            task=GRAPH_EXPLANATION_TASK,
            target_node_type="transaction",
            target_column="is_fraud",
            node_types=("transaction",),
            edge_types=("transaction_device",),
            context_columns=("is_fraud",),
            node_metadata_columns=(),
            components=("target_node",),
            hops=1,
        )


def test_validate_all_rejects_invalid_node(
    contract,
):
    with pytest.raises(
        GNNExplanationContractError
    ):
        contract.validate_all(
            task=GRAPH_EXPLANATION_TASK,
            target_node_type="transaction",
            target_column="is_fraud",
            node_types=(
                "transaction",
                "fake_node",
            ),
            edge_types=("transaction_device",),
            context_columns=(),
            node_metadata_columns=(),
            components=("target_node",),
            hops=1,
        )