import pandas as pd
import pytest

from fraud_intelligence.explainability.classical import (
    ClassicalAttributionResult,
    FeatureAttribution,
)
from fraud_intelligence.explainability.contracts import (
    GNNExplanationContractError,
)
from fraud_intelligence.explainability.gnn_importance import (
    GNNFeatureImportance,
    GNNImportanceResult,
    GNNNodeImportance,
)
from fraud_intelligence.explainability.graph_neighborhood import (
    GNNNeighborhoodExplanation,
    NeighborhoodNode,
    SupportingEdge,
)
from fraud_intelligence.explainability.shap import (
    SHAPExplanation,
    SHAPFeatureAttribution,
)
from fraud_intelligence.explainability.transaction import (
    PredictionEvidence,
    TransactionExplanation,
    TransactionExplanationError,
    assemble_transaction_explanation,
    validate_transaction_explanation,
)


TIMESTAMP = pd.Timestamp(
    "2025-01-01 10:00:00+00:00"
)


def make_prediction() -> PredictionEvidence:
    return PredictionEvidence(
        transaction_id="TX0001",
        probability=0.87,
        prediction=1,
        threshold=0.50,
        model_name="XGBoost",
    )


def make_classical() -> ClassicalAttributionResult:
    return ClassicalAttributionResult(
        model_name="XGBoost",
        model_family="classical_ml",
        feature_attributions=(
            FeatureAttribution(
                feature="amount",
                attribution=0.7,
                absolute_attribution=0.7,
                rank=1,
            ),
        ),
    )


def make_shap() -> SHAPExplanation:
    return SHAPExplanation(
        model_name="XGBoost",
        feature_attributions=(
            SHAPFeatureAttribution(
                feature="amount",
                shap_value=0.5,
                absolute_shap_value=0.5,
                rank=1,
            ),
        ),
        base_value=0.1,
        model_output=0.87,
    )


def make_neighborhood() -> GNNNeighborhoodExplanation:
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
                node_index=3,
                hop=1,
            ),
        ),
        edges=(
            SupportingEdge(
                edge_type="transaction_device",
                source_type="transaction",
                source_index=10,
                target_type="device",
                target_index=3,
                hop=1,
            ),
        ),
    )


def make_gnn_importance() -> GNNImportanceResult:
    return GNNImportanceResult(
        transaction_node_index=10,
        baseline_score=0.88,
        feature_importances=(
            GNNFeatureImportance(
                feature="amount",
                importance=0.4,
                rank=1,
            ),
        ),
        node_importances=(
            GNNNodeImportance(
                node_type="device",
                node_index=3,
                importance=0.5,
                rank=1,
            ),
        ),
    )


def test_prediction_evidence():
    prediction = make_prediction()

    assert prediction.transaction_id == "TX0001"
    assert prediction.probability == 0.87
    assert prediction.prediction == 1
    assert prediction.threshold == 0.50
    assert prediction.is_suspicious is True


def test_assemble_classical_explanation():
    result = assemble_transaction_explanation(
        transaction_id="TX0001",
        timestamp=TIMESTAMP,
        prediction=make_prediction(),
        classical_attribution=make_classical(),
    )

    assert isinstance(
        result,
        TransactionExplanation,
    )

    assert result.transaction_id == "TX0001"
    assert result.has_classical_evidence is True
    assert result.has_graph_evidence is False


def test_assemble_shap_explanation():
    result = assemble_transaction_explanation(
        transaction_id="TX0001",
        timestamp=TIMESTAMP,
        prediction=make_prediction(),
        shap_explanation=make_shap(),
    )

    assert result.shap_explanation is not None
    assert "shap" in result.evidence_types


def test_assemble_graph_explanation():
    result = assemble_transaction_explanation(
        transaction_id="TX0001",
        timestamp=TIMESTAMP,
        prediction=make_prediction(),
        graph_neighborhood=make_neighborhood(),
    )

    assert result.graph_neighborhood is not None
    assert result.has_graph_evidence is True


def test_assemble_gnn_importance():
    result = assemble_transaction_explanation(
        transaction_id="TX0001",
        timestamp=TIMESTAMP,
        prediction=make_prediction(),
        gnn_importance=make_gnn_importance(),
    )

    assert result.gnn_importance is not None
    assert "gnn_importance" in result.evidence_types


def test_assemble_complete_explanation():
    result = assemble_transaction_explanation(
        transaction_id="TX0001",
        timestamp=TIMESTAMP,
        prediction=make_prediction(),
        classical_attribution=make_classical(),
        shap_explanation=make_shap(),
        graph_neighborhood=make_neighborhood(),
        gnn_importance=make_gnn_importance(),
    )

    assert result.has_classical_evidence is True
    assert result.has_graph_evidence is True

    assert set(result.evidence_types) == {
        "classical_attribution",
        "shap",
        "graph_neighborhood",
        "gnn_importance",
    }


def test_prediction_transaction_id_mismatch_rejected():
    prediction = PredictionEvidence(
        transaction_id="TX9999",
        probability=0.8,
        prediction=1,
        threshold=0.5,
        model_name="XGBoost",
    )

    with pytest.raises(
        TransactionExplanationError
    ):
        assemble_transaction_explanation(
            transaction_id="TX0001",
            timestamp=TIMESTAMP,
            prediction=prediction,
            classical_attribution=make_classical(),
        )


def test_empty_transaction_id_rejected():
    with pytest.raises(
        TransactionExplanationError
    ):
        assemble_transaction_explanation(
            transaction_id="",
            timestamp=TIMESTAMP,
            prediction=make_prediction(),
            classical_attribution=make_classical(),
        )


def test_naive_timestamp_rejected():
    with pytest.raises(
        TransactionExplanationError
    ):
        assemble_transaction_explanation(
            transaction_id="TX0001",
            timestamp=pd.Timestamp(
                "2025-01-01 10:00:00"
            ),
            prediction=make_prediction(),
            classical_attribution=make_classical(),
        )


def test_no_evidence_rejected():
    with pytest.raises(
        TransactionExplanationError
    ):
        assemble_transaction_explanation(
            transaction_id="TX0001",
            timestamp=TIMESTAMP,
            prediction=make_prediction(),
        )


def test_invalid_probability_rejected():
    prediction = PredictionEvidence(
        transaction_id="TX0001",
        probability=1.5,
        prediction=1,
        threshold=0.5,
        model_name="XGBoost",
    )

    with pytest.raises(
        TransactionExplanationError
    ):
        assemble_transaction_explanation(
            transaction_id="TX0001",
            timestamp=TIMESTAMP,
            prediction=prediction,
            classical_attribution=make_classical(),
        )


def test_invalid_prediction_value_rejected():
    prediction = PredictionEvidence(
        transaction_id="TX0001",
        probability=0.8,
        prediction=2,
        threshold=0.5,
        model_name="XGBoost",
    )

    with pytest.raises(
        TransactionExplanationError
    ):
        assemble_transaction_explanation(
            transaction_id="TX0001",
            timestamp=TIMESTAMP,
            prediction=prediction,
            classical_attribution=make_classical(),
        )


def test_invalid_threshold_rejected():
    prediction = PredictionEvidence(
        transaction_id="TX0001",
        probability=0.8,
        prediction=1,
        threshold=1.5,
        model_name="XGBoost",
    )

    with pytest.raises(
        TransactionExplanationError
    ):
        assemble_transaction_explanation(
            transaction_id="TX0001",
            timestamp=TIMESTAMP,
            prediction=prediction,
            classical_attribution=make_classical(),
        )


def test_graph_timestamp_mismatch_rejected():
    neighborhood = GNNNeighborhoodExplanation(
        transaction_node_index=10,
        transaction_timestamp=pd.Timestamp(
            "2025-01-01 11:00:00+00:00"
        ),
        nodes=(
            NeighborhoodNode(
                node_type="transaction",
                node_index=10,
                hop=0,
            ),
        ),
        edges=(),
    )

    with pytest.raises(
        TransactionExplanationError
    ):
        assemble_transaction_explanation(
            transaction_id="TX0001",
            timestamp=TIMESTAMP,
            prediction=make_prediction(),
            graph_neighborhood=neighborhood,
        )


def test_graph_and_gnn_node_mismatch_rejected():
    gnn = GNNImportanceResult(
        transaction_node_index=999,
        baseline_score=0.8,
        feature_importances=(),
        node_importances=(),
    )

    with pytest.raises(
        TransactionExplanationError
    ):
        assemble_transaction_explanation(
            transaction_id="TX0001",
            timestamp=TIMESTAMP,
            prediction=make_prediction(),
            graph_neighborhood=make_neighborhood(),
            gnn_importance=gnn,
        )


def test_zero_probability_allowed():
    prediction = PredictionEvidence(
        transaction_id="TX0001",
        probability=0.0,
        prediction=0,
        threshold=0.5,
        model_name="XGBoost",
    )

    result = assemble_transaction_explanation(
        transaction_id="TX0001",
        timestamp=TIMESTAMP,
        prediction=prediction,
        classical_attribution=make_classical(),
    )

    assert result.prediction.probability == 0.0
    assert result.prediction.is_suspicious is False


def test_one_probability_allowed():
    prediction = PredictionEvidence(
        transaction_id="TX0001",
        probability=1.0,
        prediction=1,
        threshold=0.5,
        model_name="XGBoost",
    )

    result = assemble_transaction_explanation(
        transaction_id="TX0001",
        timestamp=TIMESTAMP,
        prediction=prediction,
        classical_attribution=make_classical(),
    )

    assert result.prediction.probability == 1.0


def test_validate_complete_explanation():
    result = assemble_transaction_explanation(
        transaction_id="TX0001",
        timestamp=TIMESTAMP,
        prediction=make_prediction(),
        classical_attribution=make_classical(),
        shap_explanation=make_shap(),
        graph_neighborhood=make_neighborhood(),
        gnn_importance=make_gnn_importance(),
    )

    validate_transaction_explanation(result)


def test_invalid_explanation_type_rejected():
    with pytest.raises(
        TransactionExplanationError
    ):
        validate_transaction_explanation(
            object()
        )


def test_invalid_graph_neighborhood_rejected():
    neighborhood = GNNNeighborhoodExplanation(
        transaction_node_index=10,
        transaction_timestamp=TIMESTAMP,
        nodes=(
            NeighborhoodNode(
                node_type="device",
                node_index=3,
                hop=1,
            ),
        ),
        edges=(),
    )

    result = TransactionExplanation(
        transaction_id="TX0001",
        timestamp=TIMESTAMP,
        prediction=make_prediction(),
        classical_attribution=None,
        shap_explanation=None,
        graph_neighborhood=neighborhood,
        gnn_importance=None,
    )

    with pytest.raises(
        TransactionExplanationError
    ):
        validate_transaction_explanation(result)


def test_evidence_types_empty_when_no_evidence():
    result = TransactionExplanation(
        transaction_id="TX0001",
        timestamp=TIMESTAMP,
        prediction=make_prediction(),
        classical_attribution=None,
        shap_explanation=None,
        graph_neighborhood=None,
        gnn_importance=None,
    )

    assert result.evidence_types == ()


def test_classical_and_shap_can_coexist():
    result = assemble_transaction_explanation(
        transaction_id="TX0001",
        timestamp=TIMESTAMP,
        prediction=make_prediction(),
        classical_attribution=make_classical(),
        shap_explanation=make_shap(),
    )

    assert result.evidence_types == (
        "classical_attribution",
        "shap",
    )


def test_graph_components_can_coexist():
    result = assemble_transaction_explanation(
        transaction_id="TX0001",
        timestamp=TIMESTAMP,
        prediction=make_prediction(),
        graph_neighborhood=make_neighborhood(),
        gnn_importance=make_gnn_importance(),
    )

    assert result.evidence_types == (
        "graph_neighborhood",
        "gnn_importance",
    )