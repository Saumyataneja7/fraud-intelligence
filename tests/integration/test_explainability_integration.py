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
    HumanReadableExplanation,
    build_human_readable_explanation,
)
from fraud_intelligence.explainability.shap import (
    SHAPExplanation,
    SHAPFeatureAttribution,
)
from fraud_intelligence.explainability.transaction import (
    PredictionEvidence,
    TransactionExplanation,
    assemble_transaction_explanation,
)
from fraud_intelligence.explainability.validation import (
    ExplanationValidationError,
    validate_complete_explanation,
)


TIMESTAMP = pd.Timestamp(
    "2025-06-01 12:00:00+00:00"
)


def _prediction() -> PredictionEvidence:
    return PredictionEvidence(
        transaction_id="txn_integration_001",
        probability=0.94,
        prediction=1,
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
                attribution=0.82,
                absolute_attribution=0.82,
                rank=1,
            ),
            FeatureAttribution(
                feature="customer_txn_count_1m",
                attribution=0.31,
                absolute_attribution=0.31,
                rank=2,
            ),
        ),
    )


def _shap() -> SHAPExplanation:
    return SHAPExplanation(
        model_name="XGBoost",
        feature_attributions=(
            SHAPFeatureAttribution(
                feature="amount_deviation_from_customer_mean",
                shap_value=0.73,
                absolute_shap_value=0.73,
                rank=1,
            ),
            SHAPFeatureAttribution(
                feature="customer_txn_count_1m",
                shap_value=0.22,
                absolute_shap_value=0.22,
                rank=2,
            ),
        ),
        base_value=0.10,
        model_output=0.94,
    )


def _neighborhood() -> GNNNeighborhoodExplanation:
    return GNNNeighborhoodExplanation(
        transaction_node_index=100,
        transaction_timestamp=TIMESTAMP,
        nodes=(
            NeighborhoodNode(
                node_type="transaction",
                node_index=100,
                hop=0,
            ),
            NeighborhoodNode(
                node_type="customer",
                node_index=12,
                hop=1,
            ),
            NeighborhoodNode(
                node_type="device",
                node_index=8,
                hop=1,
            ),
        ),
        edges=(),
    )


def _gnn_importance() -> GNNImportanceResult:
    return GNNImportanceResult(
        transaction_node_index=100,
        baseline_score=0.92,
        feature_importances=(
            GNNFeatureImportance(
                feature="customer_txn_count_1m",
                importance=0.41,
                rank=1,
            ),
        ),
        node_importances=(
            GNNNodeImportance(
                node_type="device",
                node_index=8,
                importance=0.77,
                rank=1,
            ),
            GNNNodeImportance(
                node_type="customer",
                node_index=12,
                importance=0.52,
                rank=2,
            ),
        ),
    )


def _complete_transaction_explanation() -> TransactionExplanation:
    return assemble_transaction_explanation(
        transaction_id="txn_integration_001",
        timestamp=TIMESTAMP,
        prediction=_prediction(),
        classical_attribution=_classical(),
        shap_explanation=_shap(),
        graph_neighborhood=_neighborhood(),
        gnn_importance=_gnn_importance(),
    )


def test_complete_explainability_pipeline() -> None:
    explanation = _complete_transaction_explanation()

    assert explanation.transaction_id == "txn_integration_001"
    assert explanation.prediction.prediction == 1

    assert explanation.has_classical_evidence is True
    assert explanation.has_graph_evidence is True

    assert explanation.evidence_types == (
        "classical_attribution",
        "shap",
        "graph_neighborhood",
        "gnn_importance",
    )


def test_human_readable_layer_consumes_assembled_explanation() -> None:
    explanation = _complete_transaction_explanation()

    human = build_human_readable_explanation(
        explanation
    )

    assert human.transaction_id == explanation.transaction_id
    assert human.risk_level == "suspicious"

    assert human.reason_count == 4
    assert human.graph_finding_count == 4

    assert (
        "classical feature attribution"
        in human.supporting_evidence
    )

    assert (
        "SHAP feature attribution"
        in human.supporting_evidence
    )

    assert (
        "historical graph neighborhood"
        in human.supporting_evidence
    )

    assert (
        "GNN feature/node importance"
        in human.supporting_evidence
    )


def test_complete_pipeline_passes_validation() -> None:
    explanation = _complete_transaction_explanation()

    human = build_human_readable_explanation(
        explanation
    )

    report = validate_complete_explanation(
        transaction_explanation=explanation,
        human_readable_explanation=human,
    )

    assert report.transaction_id == explanation.transaction_id
    assert report.has_leakage is False
    assert report.forbidden_features_found == ()
    assert report.future_transaction_indices == ()
    assert report.same_timestamp_transaction_indices == ()


def test_classical_only_pipeline() -> None:
    explanation = assemble_transaction_explanation(
        transaction_id="txn_integration_001",
        timestamp=TIMESTAMP,
        prediction=_prediction(),
        classical_attribution=_classical(),
    )

    human = build_human_readable_explanation(
        explanation
    )

    report = validate_complete_explanation(
        transaction_explanation=explanation,
        human_readable_explanation=human,
    )

    assert explanation.evidence_types == (
        "classical_attribution",
    )

    assert human.reason_count == 2
    assert human.graph_finding_count == 0
    assert report.has_leakage is False


def test_shap_only_pipeline() -> None:
    explanation = assemble_transaction_explanation(
        transaction_id="txn_integration_001",
        timestamp=TIMESTAMP,
        prediction=_prediction(),
        shap_explanation=_shap(),
    )

    human = build_human_readable_explanation(
        explanation
    )

    assert explanation.evidence_types == ("shap",)
    assert human.reason_count == 2
    assert human.graph_finding_count == 0


def test_graph_only_pipeline() -> None:
    explanation = assemble_transaction_explanation(
        transaction_id="txn_integration_001",
        timestamp=TIMESTAMP,
        prediction=_prediction(),
        graph_neighborhood=_neighborhood(),
    )

    human = build_human_readable_explanation(
        explanation
    )

    assert explanation.evidence_types == (
        "graph_neighborhood",
    )

    assert human.reason_count == 0
    assert human.graph_finding_count == 2


def test_gnn_importance_only_pipeline() -> None:
    explanation = assemble_transaction_explanation(
        transaction_id="txn_integration_001",
        timestamp=TIMESTAMP,
        prediction=_prediction(),
        gnn_importance=_gnn_importance(),
    )

    human = build_human_readable_explanation(
        explanation
    )

    assert explanation.evidence_types == (
        "gnn_importance",
    )

    assert human.reason_count == 0
    assert human.graph_finding_count == 2


def test_transaction_id_is_preserved_across_pipeline() -> None:
    explanation = _complete_transaction_explanation()

    human = build_human_readable_explanation(
        explanation
    )

    assert human.transaction_id == (
        explanation.transaction_id
    )

    assert human.transaction_id == (
        explanation.prediction.transaction_id
    )


def test_prediction_threshold_is_preserved() -> None:
    explanation = _complete_transaction_explanation()

    assert explanation.prediction.threshold == 0.50
    assert explanation.prediction.probability == 0.94


def test_graph_target_is_preserved() -> None:
    explanation = _complete_transaction_explanation()

    assert (
        explanation.graph_neighborhood.transaction_node_index
        == explanation.gnn_importance.transaction_node_index
    )

    assert (
        explanation.graph_neighborhood.transaction_node_index
        == 100
    )


def test_graph_target_timestamp_is_preserved() -> None:
    explanation = _complete_transaction_explanation()

    assert (
        explanation.graph_neighborhood.transaction_timestamp
        == explanation.timestamp
    )


def test_forbidden_feature_cannot_enter_pipeline() -> None:
    forbidden = ClassicalAttributionResult(
        model_name="XGBoost",
        model_family="classical_ml",
        feature_attributions=(
            FeatureAttribution(
                feature="is_fraud",
                attribution=1.0,
                absolute_attribution=1.0,
                rank=1,
            ),
        ),
    )

    explanation = TransactionExplanation(
        transaction_id="txn_integration_001",
        timestamp=TIMESTAMP,
        prediction=_prediction(),
        classical_attribution=forbidden,
        shap_explanation=None,
        graph_neighborhood=None,
        gnn_importance=None,
    )

    with pytest.raises(
        ExplanationValidationError,
        match="Forbidden explanation features",
    ):
        validate_complete_explanation(
            transaction_explanation=explanation
        )


def test_human_readable_output_contains_limitation() -> None:
    explanation = _complete_transaction_explanation()

    human = build_human_readable_explanation(
        explanation
    )

    assert any(
        "does not establish that fraud actually occurred"
        in limitation
        for limitation in human.limitations
    )