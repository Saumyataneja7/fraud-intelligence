from __future__ import annotations

import pandas as pd
import pytest

from fraud_intelligence.explainability.classical import (
    ClassicalAttributionResult,
    FeatureAttribution,
)
from fraud_intelligence.explainability.gnn_importance import (
    GNNFeatureImportance,
    GNNImportanceResult,
    GNNNodeImportance,
)
from fraud_intelligence.explainability.graph_neighborhood import (
    GNNNeighborhoodExplanation,
    NeighborhoodNode,
)
from fraud_intelligence.explainability.human_readable import (
    ExplanationReason,
    GraphFinding,
    HumanReadableExplanation,
    HumanReadableExplanationError,
    build_human_readable_explanation,
    validate_human_readable_explanation,
)
from fraud_intelligence.explainability.shap import (
    SHAPExplanation,
    SHAPFeatureAttribution,
)
from fraud_intelligence.explainability.transaction import (
    PredictionEvidence,
    TransactionExplanation,
)


TIMESTAMP = pd.Timestamp(
    "2025-01-01 12:00:00+00:00"
)


def _prediction(
    prediction: int = 1,
) -> PredictionEvidence:
    return PredictionEvidence(
        transaction_id="txn_001",
        probability=0.91,
        prediction=prediction,
        threshold=0.50,
        model_name="XGBoost",
    )


def _classical() -> ClassicalAttributionResult:
    return ClassicalAttributionResult(
        model_name="XGBoost",
        model_family="classical_ml",
        feature_attributions=(
            FeatureAttribution(
                feature="amount_deviation_from_customer_mean",
                attribution=0.80,
                absolute_attribution=0.80,
                rank=1,
            ),
        ),
    )


def _shap() -> SHAPExplanation:
    return SHAPExplanation(
        model_name="XGBoost",
        feature_attributions=(
            SHAPFeatureAttribution(
                feature="customer_txn_count_1m",
                shap_value=0.60,
                absolute_shap_value=0.60,
                rank=1,
            ),
        ),
        base_value=0.10,
        model_output=0.91,
    )


def _neighborhood() -> GNNNeighborhoodExplanation:
    return GNNNeighborhoodExplanation(
        transaction_node_index=10,
        transaction_timestamp=TIMESTAMP,
        nodes=(
            NeighborhoodNode(
                node_type="transaction",
                node_index=10,
                hop=0,
            ),
            NeighborhoodNode(
                node_type="device",
                node_index=5,
                hop=1,
            ),
        ),
        edges=(),
    )


def _gnn_importance() -> GNNImportanceResult:
    return GNNImportanceResult(
        transaction_node_index=10,
        baseline_score=0.90,
        feature_importances=(
            GNNFeatureImportance(
                feature="customer_txn_count_1m",
                importance=0.40,
                rank=1,
            ),
        ),
        node_importances=(
            GNNNodeImportance(
                node_type="device",
                node_index=5,
                importance=0.70,
                rank=1,
            ),
        ),
    )


def _explanation(
    *,
    prediction: int = 1,
    classical: ClassicalAttributionResult | None = None,
    shap: SHAPExplanation | None = None,
    neighborhood: GNNNeighborhoodExplanation | None = None,
    importance: GNNImportanceResult | None = None,
) -> TransactionExplanation:
    return TransactionExplanation(
        transaction_id="txn_001",
        timestamp=TIMESTAMP,
        prediction=_prediction(prediction),
        classical_attribution=classical,
        shap_explanation=shap,
        graph_neighborhood=neighborhood,
        gnn_importance=importance,
    )


def test_suspicious_prediction_summary() -> None:
    result = build_human_readable_explanation(
        _explanation(
            classical=_classical()
        )
    )

    assert result.transaction_id == "txn_001"
    assert result.risk_level == "suspicious"
    assert "flagged as suspicious" in result.summary
    assert "91.00%" in result.summary


def test_not_suspicious_prediction_summary() -> None:
    result = build_human_readable_explanation(
        _explanation(
            prediction=0,
            classical=_classical(),
        )
    )

    assert result.risk_level == "not_suspicious"
    assert "not flagged as suspicious" in result.summary


def test_classical_attribution_becomes_reason() -> None:
    result = build_human_readable_explanation(
        _explanation(
            classical=_classical()
        )
    )

    assert result.reason_count == 1
    assert result.reasons[0].category == "classical_attribution"
    assert (
        result.reasons[0].feature
        == "amount_deviation_from_customer_mean"
    )
    assert result.reasons[0].direction == "increases"
    assert "increases" in result.reasons[0].text


def test_negative_classical_attribution_direction() -> None:
    attribution = ClassicalAttributionResult(
        model_name="XGBoost",
        model_family="classical_ml",
        feature_attributions=(
            FeatureAttribution(
                feature="amount_deviation_from_customer_mean",
                attribution=-0.30,
                absolute_attribution=0.30,
                rank=1,
            ),
        ),
    )

    result = build_human_readable_explanation(
        _explanation(classical=attribution)
    )

    assert result.reasons[0].direction == "decreases"


def test_shap_becomes_reason() -> None:
    result = build_human_readable_explanation(
        _explanation(
            shap=_shap()
        )
    )

    assert result.reason_count == 1
    assert result.reasons[0].category == "shap"
    assert result.reasons[0].feature == "customer_txn_count_1m"
    assert result.reasons[0].attribution == 0.60


def test_classical_and_shap_are_both_preserved() -> None:
    result = build_human_readable_explanation(
        _explanation(
            classical=_classical(),
            shap=_shap(),
        )
    )

    assert result.reason_count == 2
    assert {
        reason.category
        for reason in result.reasons
    } == {
        "classical_attribution",
        "shap",
    }


def test_graph_neighborhood_becomes_graph_finding() -> None:
    result = build_human_readable_explanation(
        _explanation(
            neighborhood=_neighborhood()
        )
    )

    assert result.graph_finding_count == 1
    assert (
        result.graph_findings[0].category
        == "graph_neighborhood"
    )
    assert result.graph_findings[0].node_type == "device"
    assert result.graph_findings[0].node_index == 5
    assert "device node 5" in result.graph_findings[0].text


def test_gnn_node_importance_becomes_graph_finding() -> None:
    result = build_human_readable_explanation(
        _explanation(
            importance=_gnn_importance()
        )
    )

    assert result.graph_finding_count == 1
    assert (
        result.graph_findings[0].category
        == "gnn_node_importance"
    )
    assert result.graph_findings[0].importance == 0.70


def test_target_transaction_is_not_reported_as_neighbor() -> None:
    result = build_human_readable_explanation(
        _explanation(
            neighborhood=_neighborhood()
        )
    )

    assert all(
        finding.node_index != 10
        for finding in result.graph_findings
    )


def test_supporting_evidence_lists_available_components() -> None:
    result = build_human_readable_explanation(
        _explanation(
            classical=_classical(),
            shap=_shap(),
            neighborhood=_neighborhood(),
            importance=_gnn_importance(),
        )
    )

    assert result.supporting_evidence == (
        "classical feature attribution",
        "SHAP feature attribution",
        "historical graph neighborhood",
        "GNN feature/node importance",
    )


def test_limitations_report_missing_components() -> None:
    result = build_human_readable_explanation(
        _explanation(
            classical=_classical()
        )
    )

    assert (
        "No SHAP explanation was provided."
        in result.limitations
    )
    assert (
        "No graph neighborhood evidence was provided."
        in result.limitations
    )
    assert (
        "No GNN importance evidence was provided."
        in result.limitations
    )


def test_fraud_is_not_claimed_as_fact() -> None:
    result = build_human_readable_explanation(
        _explanation(
            classical=_classical()
        )
    )

    assert any(
        "does not establish that fraud actually occurred"
        in limitation
        for limitation in result.limitations
    )


def test_no_new_threshold_is_introduced() -> None:
    prediction = PredictionEvidence(
        transaction_id="txn_001",
        probability=0.49,
        prediction=1,
        threshold=0.45,
        model_name="XGBoost",
    )

    explanation = TransactionExplanation(
        transaction_id="txn_001",
        timestamp=TIMESTAMP,
        prediction=prediction,
        classical_attribution=_classical(),
        shap_explanation=None,
        graph_neighborhood=None,
        gnn_importance=None,
    )

    result = build_human_readable_explanation(explanation)

    assert result.risk_level == "suspicious"
    assert "49.00%" in result.summary


def test_prediction_transaction_id_mismatch_rejected() -> None:
    prediction = PredictionEvidence(
        transaction_id="txn_999",
        probability=0.90,
        prediction=1,
        threshold=0.50,
        model_name="XGBoost",
    )

    explanation = TransactionExplanation(
        transaction_id="txn_001",
        timestamp=TIMESTAMP,
        prediction=prediction,
        classical_attribution=_classical(),
        shap_explanation=None,
        graph_neighborhood=None,
        gnn_importance=None,
    )

    with pytest.raises(
        HumanReadableExplanationError,
        match="transaction_id",
    ):
        build_human_readable_explanation(explanation)


def test_invalid_probability_rejected() -> None:
    prediction = PredictionEvidence(
        transaction_id="txn_001",
        probability=1.5,
        prediction=1,
        threshold=0.50,
        model_name="XGBoost",
    )

    explanation = TransactionExplanation(
        transaction_id="txn_001",
        timestamp=TIMESTAMP,
        prediction=prediction,
        classical_attribution=_classical(),
        shap_explanation=None,
        graph_neighborhood=None,
        gnn_importance=None,
    )

    with pytest.raises(
        HumanReadableExplanationError,
        match="between 0 and 1",
    ):
        build_human_readable_explanation(explanation)


def test_empty_transaction_id_rejected() -> None:
    explanation = HumanReadableExplanation(
        transaction_id="",
        summary="test",
        risk_level="suspicious",
        reasons=(),
        graph_findings=(),
        supporting_evidence=(),
        limitations=(),
    )

    with pytest.raises(
        HumanReadableExplanationError,
        match="transaction_id",
    ):
        validate_human_readable_explanation(explanation)


def test_invalid_risk_level_rejected() -> None:
    explanation = HumanReadableExplanation(
        transaction_id="txn_001",
        summary="test",
        risk_level="high",
        reasons=(),
        graph_findings=(),
        supporting_evidence=(),
        limitations=(),
    )

    with pytest.raises(
        HumanReadableExplanationError,
        match="risk_level",
    ):
        validate_human_readable_explanation(explanation)


def test_invalid_reason_type_rejected() -> None:
    explanation = HumanReadableExplanation(
        transaction_id="txn_001",
        summary="test",
        risk_level="suspicious",
        reasons=("invalid",),
        graph_findings=(),
        supporting_evidence=(),
        limitations=(),
    )

    with pytest.raises(
        HumanReadableExplanationError,
        match="ExplanationReason",
    ):
        validate_human_readable_explanation(explanation)


def test_invalid_graph_finding_type_rejected() -> None:
    explanation = HumanReadableExplanation(
        transaction_id="txn_001",
        summary="test",
        risk_level="suspicious",
        reasons=(),
        graph_findings=("invalid",),
        supporting_evidence=(),
        limitations=(),
    )

    with pytest.raises(
        HumanReadableExplanationError,
        match="GraphFinding",
    ):
        validate_human_readable_explanation(explanation)


def test_non_finite_reason_attribution_rejected() -> None:
    explanation = HumanReadableExplanation(
        transaction_id="txn_001",
        summary="test",
        risk_level="suspicious",
        reasons=(
            ExplanationReason(
                category="shap",
                feature="feature",
                direction="increases",
                attribution=float("nan"),
                text="test",
                rank=1,
            ),
        ),
        graph_findings=(),
        supporting_evidence=(),
        limitations=(),
    )

    with pytest.raises(
        HumanReadableExplanationError,
        match="finite",
    ):
        validate_human_readable_explanation(explanation)


def test_non_finite_graph_importance_rejected() -> None:
    explanation = HumanReadableExplanation(
        transaction_id="txn_001",
        summary="test",
        risk_level="suspicious",
        reasons=(),
        graph_findings=(
            GraphFinding(
                category="gnn_node_importance",
                text="test",
                node_type="device",
                node_index=5,
                importance=float("inf"),
            ),
        ),
        supporting_evidence=(),
        limitations=(),
    )

    with pytest.raises(
        HumanReadableExplanationError,
        match="finite",
    ):
        validate_human_readable_explanation(explanation)


def test_complete_explanation() -> None:
    result = build_human_readable_explanation(
        _explanation(
            classical=_classical(),
            shap=_shap(),
            neighborhood=_neighborhood(),
            importance=_gnn_importance(),
        )
    )

    assert result.transaction_id == "txn_001"
    assert result.risk_level == "suspicious"
    assert result.reason_count == 2
    assert result.graph_finding_count == 2
    assert len(result.supporting_evidence) == 4
    assert len(result.limitations) == 1


def test_result_is_immutable() -> None:
    result = build_human_readable_explanation(
        _explanation(
            classical=_classical()
        )
    )

    with pytest.raises(AttributeError):
        result.risk_level = "not_suspicious"