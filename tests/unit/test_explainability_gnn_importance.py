import numpy as np
import pandas as pd
import pytest

from fraud_intelligence.explainability.contracts import (
    GNNExplanationContract,
    GNNExplanationContractError,
)
from fraud_intelligence.explainability.gnn_importance import (
    GNNFeatureImportance,
    GNNImportanceError,
    GNNImportanceResult,
    GNNNodeImportance,
    build_gnn_importance_result,
    calculate_feature_perturbation_importance,
    calculate_node_perturbation_importance,
    validate_gnn_importance_result,
)
from fraud_intelligence.explainability.graph_neighborhood import (
    GNNNeighborhoodExplanation,
    NeighborhoodNode,
    SupportingEdge,
)


FEATURES = (
    "amount",
    "customer_txn_count_5m",
    "is_new_device",
)


def make_neighborhood() -> GNNNeighborhoodExplanation:
    return GNNNeighborhoodExplanation(
        transaction_node_index=0,
        transaction_timestamp=pd.Timestamp(
            "2025-01-01 10:00:00+00:00"
        ),
        nodes=(
            NeighborhoodNode(
                node_type="transaction",
                node_index=0,
                hop=0,
            ),
            NeighborhoodNode(
                node_type="customer",
                node_index=0,
                hop=1,
            ),
            NeighborhoodNode(
                node_type="device",
                node_index=0,
                hop=1,
            ),
            NeighborhoodNode(
                node_type="ip",
                node_index=0,
                hop=2,
            ),
        ),
        edges=(
            SupportingEdge(
                edge_type="customer_transaction",
                source_type="customer",
                source_index=0,
                target_type="transaction",
                target_index=0,
                hop=1,
            ),
            SupportingEdge(
                edge_type="customer_device",
                source_type="customer",
                source_index=0,
                target_type="device",
                target_index=0,
                hop=1,
            ),
            SupportingEdge(
                edge_type="customer_ip",
                source_type="customer",
                source_index=0,
                target_type="ip",
                target_index=0,
                hop=2,
            ),
        ),
    )


def make_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                100.0,
                5.0,
                1.0,
            ]
        ],
        columns=FEATURES,
    )


def test_feature_importance_dataclass():
    result = GNNFeatureImportance(
        feature="amount",
        importance=0.5,
        rank=1,
    )

    assert result.feature == "amount"
    assert result.importance == 0.5
    assert result.rank == 1


def test_node_importance_dataclass():
    result = GNNNodeImportance(
        node_type="device",
        node_index=4,
        importance=0.7,
        rank=1,
    )

    assert result.node_type == "device"
    assert result.node_index == 4
    assert result.importance == 0.7
    assert result.rank == 1


def test_result_properties():
    result = GNNImportanceResult(
        transaction_node_index=10,
        baseline_score=0.8,
        feature_importances=(
            GNNFeatureImportance(
                feature="amount",
                importance=0.5,
                rank=1,
            ),
        ),
        node_importances=(
            GNNNodeImportance(
                node_type="device",
                node_index=2,
                importance=0.4,
                rank=1,
            ),
        ),
    )

    assert result.feature_count == 1
    assert result.node_count == 1
    assert result.top_feature is not None
    assert result.top_feature.feature == "amount"
    assert result.top_node is not None
    assert result.top_node.node_type == "device"


def test_feature_perturbation_importance():
    X = make_features()

    def predict_score(frame: pd.DataFrame) -> float:
        return min(
            1.0,
            float(
                0.5
                + frame.loc[0, "amount"] / 1000
                + frame.loc[0, "customer_txn_count_5m"]
                / 100
                + frame.loc[0, "is_new_device"] * 0.2
            ),
        )

    result = calculate_feature_perturbation_importance(
        X=X,
        feature_columns=FEATURES,
        predict_score=predict_score,
    )

    assert len(result) == len(FEATURES)

    assert result[0].feature == "is_new_device"
    assert result[0].importance > 0


def test_feature_importances_are_ranked():
    X = make_features()

    def predict_score(frame: pd.DataFrame) -> float:
        return min(
            1.0,
            float(
                frame.loc[0, "amount"] / 100
                + frame.loc[0, "customer_txn_count_5m"]
                / 10
                + frame.loc[0, "is_new_device"]
            )
            / 10,
        )

    result = calculate_feature_perturbation_importance(
        X=X,
        feature_columns=FEATURES,
        predict_score=predict_score,
    )

    ranks = [
        item.rank
        for item in result
    ]

    assert ranks == [1, 2, 3]


def test_feature_importances_are_non_negative():
    X = make_features()

    result = calculate_feature_perturbation_importance(
        X=X,
        feature_columns=FEATURES,
        predict_score=lambda frame: 0.5,
    )

    assert all(
        item.importance >= 0
        for item in result
    )


def test_constant_model_produces_zero_importance():
    X = make_features()

    result = calculate_feature_perturbation_importance(
        X=X,
        feature_columns=FEATURES,
        predict_score=lambda frame: 0.5,
    )

    assert all(
        item.importance == 0
        for item in result
    )


def test_missing_features_rejected():
    X = make_features()
    X.loc[0, "amount"] = np.nan

    with pytest.raises(GNNImportanceError):
        calculate_feature_perturbation_importance(
            X=X,
            feature_columns=FEATURES,
            predict_score=lambda frame: 0.5,
        )


def test_infinite_features_rejected():
    X = make_features()
    X.loc[0, "amount"] = np.inf

    with pytest.raises(GNNImportanceError):
        calculate_feature_perturbation_importance(
            X=X,
            feature_columns=FEATURES,
            predict_score=lambda frame: 0.5,
        )


def test_multiple_rows_rejected():
    X = pd.concat(
        [make_features(), make_features()],
        ignore_index=True,
    )

    with pytest.raises(GNNImportanceError):
        calculate_feature_perturbation_importance(
            X=X,
            feature_columns=FEATURES,
            predict_score=lambda frame: 0.5,
        )


def test_feature_order_mismatch_rejected():
    X = make_features()[
        [
            "is_new_device",
            "amount",
            "customer_txn_count_5m",
        ]
    ]

    with pytest.raises(GNNImportanceError):
        calculate_feature_perturbation_importance(
            X=X,
            feature_columns=FEATURES,
            predict_score=lambda frame: 0.5,
        )


@pytest.mark.parametrize(
    "forbidden",
    [
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
    ],
)
def test_forbidden_features_rejected(forbidden):
    X = make_features()

    columns = list(FEATURES)
    columns[0] = forbidden

    X.columns = columns

    with pytest.raises(
        (
            GNNImportanceError,
            GNNExplanationContractError,
        )
    ):
        calculate_feature_perturbation_importance(
            X=X,
            feature_columns=tuple(columns),
            predict_score=lambda frame: 0.5,
        )


def test_node_perturbation_importance():
    neighborhood = make_neighborhood()

    scores = {
        ("customer", 0): 0.6,
        ("device", 0): 0.3,
        ("ip", 0): 0.7,
    }

    result = calculate_node_perturbation_importance(
        neighborhood=neighborhood,
        baseline_score=0.9,
        perturb_node=lambda node_type, node_index: scores[
            (node_type, node_index)
        ],
    )

    assert len(result) == 3

    assert result[0].node_type == "device"
    assert result[0].importance == pytest.approx(0.6)

    assert result[1].node_type == "customer"
    assert result[1].importance == pytest.approx(0.3)

    assert result[2].node_type == "ip"
    assert result[2].importance == pytest.approx(0.2)


def test_target_transaction_is_not_ablated():
    neighborhood = make_neighborhood()

    called = []

    def perturb(
        node_type,
        node_index,
    ):
        called.append(
            (node_type, node_index)
        )
        return 0.5

    calculate_node_perturbation_importance(
        neighborhood=neighborhood,
        baseline_score=0.8,
        perturb_node=perturb,
    )

    assert (
        "transaction",
        0,
    ) not in called


def test_node_importance_non_negative():
    neighborhood = make_neighborhood()

    result = calculate_node_perturbation_importance(
        neighborhood=neighborhood,
        baseline_score=0.8,
        perturb_node=lambda node_type, node_index: 0.8,
    )

    assert all(
        item.importance >= 0
        for item in result
    )


def test_node_importance_deterministic():
    neighborhood = make_neighborhood()

    def perturb(node_type, node_index):
        return {
            ("customer", 0): 0.5,
            ("device", 0): 0.5,
            ("ip", 0): 0.5,
        }[
            (node_type, node_index)
        ]

    first = calculate_node_perturbation_importance(
        neighborhood=neighborhood,
        baseline_score=0.8,
        perturb_node=perturb,
    )

    second = calculate_node_perturbation_importance(
        neighborhood=neighborhood,
        baseline_score=0.8,
        perturb_node=perturb,
    )

    assert first == second


def test_build_result():
    neighborhood = make_neighborhood()

    features = (
        GNNFeatureImportance(
            feature="amount",
            importance=0.4,
            rank=1,
        ),
    )

    nodes = (
        GNNNodeImportance(
            node_type="device",
            node_index=0,
            importance=0.3,
            rank=1,
        ),
    )

    result = build_gnn_importance_result(
        neighborhood=neighborhood,
        baseline_score=0.8,
        feature_importances=features,
        node_importances=nodes,
    )

    assert result.transaction_node_index == 0
    assert result.baseline_score == 0.8


def test_invalid_baseline_rejected():
    neighborhood = make_neighborhood()

    with pytest.raises(GNNImportanceError):
        build_gnn_importance_result(
            neighborhood=neighborhood,
            baseline_score=1.5,
            feature_importances=(),
            node_importances=(),
        )


def test_invalid_feature_importance_rejected():
    result = GNNImportanceResult(
        transaction_node_index=0,
        baseline_score=0.8,
        feature_importances=(
            GNNFeatureImportance(
                feature="amount",
                importance=-0.1,
                rank=1,
            ),
        ),
        node_importances=(),
    )

    with pytest.raises(GNNImportanceError):
        validate_gnn_importance_result(result)


def test_duplicate_feature_rejected():
    result = GNNImportanceResult(
        transaction_node_index=0,
        baseline_score=0.8,
        feature_importances=(
            GNNFeatureImportance(
                feature="amount",
                importance=0.1,
                rank=1,
            ),
            GNNFeatureImportance(
                feature="amount",
                importance=0.2,
                rank=2,
            ),
        ),
        node_importances=(),
    )

    with pytest.raises(GNNImportanceError):
        validate_gnn_importance_result(result)


def test_duplicate_node_rejected():
    result = GNNImportanceResult(
        transaction_node_index=0,
        baseline_score=0.8,
        feature_importances=(),
        node_importances=(
            GNNNodeImportance(
                node_type="device",
                node_index=1,
                importance=0.1,
                rank=1,
            ),
            GNNNodeImportance(
                node_type="device",
                node_index=1,
                importance=0.2,
                rank=2,
            ),
        ),
    )

    with pytest.raises(GNNImportanceError):
        validate_gnn_importance_result(result)


def test_invalid_feature_rank_rejected():
    result = GNNImportanceResult(
        transaction_node_index=0,
        baseline_score=0.8,
        feature_importances=(
            GNNFeatureImportance(
                feature="amount",
                importance=0.1,
                rank=2,
            ),
        ),
        node_importances=(),
    )

    with pytest.raises(GNNImportanceError):
        validate_gnn_importance_result(result)


def test_invalid_node_rank_rejected():
    result = GNNImportanceResult(
        transaction_node_index=0,
        baseline_score=0.8,
        feature_importances=(),
        node_importances=(
            GNNNodeImportance(
                node_type="device",
                node_index=1,
                importance=0.1,
                rank=2,
            ),
        ),
    )

    with pytest.raises(GNNImportanceError):
        validate_gnn_importance_result(result)


def test_unknown_node_type_rejected():
    result = GNNImportanceResult(
        transaction_node_index=0,
        baseline_score=0.8,
        feature_importances=(),
        node_importances=(
            GNNNodeImportance(
                node_type="unknown",
                node_index=1,
                importance=0.1,
                rank=1,
            ),
        ),
    )

    with pytest.raises(
        GNNExplanationContractError
    ):
        validate_gnn_importance_result(result)


def test_nan_importance_rejected():
    result = GNNImportanceResult(
        transaction_node_index=0,
        baseline_score=0.8,
        feature_importances=(
            GNNFeatureImportance(
                feature="amount",
                importance=np.nan,
                rank=1,
            ),
        ),
        node_importances=(),
    )

    with pytest.raises(GNNImportanceError):
        validate_gnn_importance_result(result)


def test_inf_node_importance_rejected():
    result = GNNImportanceResult(
        transaction_node_index=0,
        baseline_score=0.8,
        feature_importances=(),
        node_importances=(
            GNNNodeImportance(
                node_type="device",
                node_index=1,
                importance=np.inf,
                rank=1,
            ),
        ),
    )

    with pytest.raises(GNNImportanceError):
        validate_gnn_importance_result(result)