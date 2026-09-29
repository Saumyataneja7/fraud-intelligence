from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from api.contracts.common import (
    APIErrorResponse,
    EntityReferenceResponse,
    HealthResponse,
)
from api.contracts.entity import EntityInvestigationResponse
from api.contracts.explanation import (
    ExplanationResponse,
    FeatureAttributionResponse,
    GraphFindingResponse,
)
from api.contracts.fraud_ring import (
    FraudRingInvestigationResponse,
    RingEvidenceResponse,
    RingNetworkScoreResponse,
    RingTransactionResponse,
)
from api.contracts.graph import (
    GraphEdgeResponse,
    GraphNodeResponse,
    GraphResponse,
)
from api.contracts.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from api.contracts.transaction import (
    SuspicionSignalResponse,
    TransactionInvestigationResponse,
    TransactionRelationshipResponse,
)


def test_health_response():
    response = HealthResponse(
        status="ok",
        service="fraud-intelligence-api",
        version="1.0.0",
    )

    assert response.status == "ok"


def test_error_response():
    response = APIErrorResponse(
        error="NotFound",
        message="Transaction not found",
    )

    assert response.error == "NotFound"


def test_entity_reference():
    response = EntityReferenceResponse(
        entity_type="customer",
        entity_id="customer_001",
    )

    assert response.entity_type == "customer"


def test_prediction_request():
    request = PredictionRequest(
        transaction_id="txn_001",
    )

    assert request.transaction_id == "txn_001"


def test_prediction_response():
    response = PredictionResponse(
        transaction_id="txn_001",
        prediction_probability=0.82,
        prediction_label=1,
        model_name="xgboost",
        model_family="classical",
    )

    assert response.prediction_probability == 0.82


@pytest.mark.parametrize(
    "probability",
    [-0.01, 1.01],
)
def test_prediction_probability_bounds(probability):
    with pytest.raises(ValidationError):
        PredictionResponse(
            transaction_id="txn_001",
            prediction_probability=probability,
            prediction_label=1,
            model_name="xgboost",
            model_family="classical",
        )


@pytest.mark.parametrize(
    "label",
    [-1, 2],
)
def test_prediction_label_bounds(label):
    with pytest.raises(ValidationError):
        PredictionResponse(
            transaction_id="txn_001",
            prediction_probability=0.5,
            prediction_label=label,
            model_name="xgboost",
            model_family="classical",
        )


def test_suspicion_signal():
    signal = SuspicionSignalResponse(
        name="velocity",
        value=5,
        description="Five transactions in the historical window.",
        source="velocity_features",
    )

    assert signal.value == 5


def test_transaction_relationship():
    relationship = TransactionRelationshipResponse(
        relationship_type="customer_transaction",
        source=EntityReferenceResponse(
            entity_type="customer",
            entity_id="customer_001",
        ),
        target=EntityReferenceResponse(
            entity_type="transaction",
            entity_id="txn_001",
        ),
        timestamp=datetime.now(timezone.utc),
    )

    assert relationship.target.entity_type == "transaction"


def test_transaction_investigation_response():
    response = TransactionInvestigationResponse(
        transaction_id="txn_001",
        timestamp=datetime.now(timezone.utc),
        prediction_probability=0.75,
        prediction_label=1,
        suspicion_signals=[],
        related_entities=[],
        relationships=[],
        candidate_ring_ids=[],
        investigation_summary="Investigation context available.",
        temporal_rule="Historical context only.",
    )

    assert response.transaction_id == "txn_001"


def test_entity_investigation_response():
    response = EntityInvestigationResponse(
        entity=EntityReferenceResponse(
            entity_type="customer",
            entity_id="customer_001",
        ),
        related_entities=[],
        related_transactions=["txn_001"],
        candidate_ring_ids=[],
        network_scores=[],
        investigation_summary="Customer investigation view.",
        temporal_rule="Historical context only.",
    )

    assert response.entity.entity_type == "customer"


def test_ring_transaction_response():
    response = RingTransactionResponse(
        transaction_id="txn_001",
        timestamp=datetime.now(timezone.utc),
    )

    assert response.transaction_id == "txn_001"


def test_ring_evidence_response():
    response = RingEvidenceResponse(
        evidence_type="entity_connectivity",
        description="Multiple connected entities.",
        entity_types=["customer", "device"],
        relationship_types=["customer_device"],
        strength="moderate",
    )

    assert response.strength == "moderate"


def test_ring_network_score_response():
    response = RingNetworkScoreResponse(
        candidate_id="ring_001",
        entity_count=10,
        relationship_count=15,
        relationship_density=0.2,
        entity_type_count=4,
        relationship_type_count=5,
        non_transaction_entity_count=8,
        non_transaction_connectivity=0.5,
        structural_score=72.5,
    )

    assert response.structural_score == 72.5


def test_fraud_ring_investigation_response():
    response = FraudRingInvestigationResponse(
        candidate_id="ring_001",
        entities=[],
        relationships=[],
        transactions=[],
        network_score=RingNetworkScoreResponse(
            candidate_id="ring_001",
            entity_count=10,
            relationship_count=15,
            relationship_density=0.2,
            entity_type_count=4,
            relationship_type_count=5,
            non_transaction_entity_count=8,
            non_transaction_connectivity=0.5,
            structural_score=72.5,
        ),
        evidence=[],
        entity_type_counts={},
        relationship_type_counts={},
        investigation_summary="Candidate network investigation.",
        temporal_rule="Historical context only.",
    )

    assert response.candidate_id == "ring_001"


def test_graph_node_response():
    response = GraphNodeResponse(
        node_type="transaction",
        node_id=42,
    )

    assert response.node_id == 42


def test_graph_edge_response():
    response = GraphEdgeResponse(
        relationship_type="transaction_device",
        source_node_type="transaction",
        source_node_id=42,
        target_node_type="device",
        target_node_id=7,
    )

    assert response.relationship_type == "transaction_device"


def test_graph_response():
    response = GraphResponse(
        target_node_type="transaction",
        target_node_id=42,
        nodes=[
            GraphNodeResponse(
                node_type="transaction",
                node_id=42,
            ),
        ],
        edges=[],
    )

    assert len(response.nodes) == 1


def test_feature_attribution_response():
    response = FeatureAttributionResponse(
        feature="amount_vs_customer_mean",
        attribution=0.45,
        absolute_attribution=0.45,
        rank=1,
    )

    assert response.rank == 1


def test_graph_finding_response():
    response = GraphFindingResponse(
        finding_type="shared_device",
        description="Transaction shares a device with related entities.",
    )

    assert response.finding_type == "shared_device"


def test_explanation_response():
    response = ExplanationResponse(
        transaction_id="txn_001",
        prediction_probability=0.81,
        prediction_label=1,
        feature_attributions=[],
        graph_findings=[],
        summary="Model and graph explanation available.",
    )

    assert response.prediction_label == 1


def test_contracts_reject_unknown_fields():
    with pytest.raises(ValidationError):
        HealthResponse(
            status="ok",
            service="fraud-intelligence-api",
            version="1.0.0",
            unexpected="not allowed",
        )


def test_structural_score_bounds():
    with pytest.raises(ValidationError):
        RingNetworkScoreResponse(
            candidate_id="ring_001",
            entity_count=10,
            relationship_count=10,
            relationship_density=0.2,
            entity_type_count=3,
            relationship_type_count=3,
            non_transaction_entity_count=7,
            non_transaction_connectivity=0.5,
            structural_score=101.0,
        )