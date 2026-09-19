from __future__ import annotations

import pandas as pd
import pytest

from fraud_intelligence.explainability.classical import (
    ClassicalAttributionResult,
    FeatureAttribution,
)
from fraud_intelligence.explainability.human_readable import (
    HumanReadableExplanation,
)
from fraud_intelligence.explainability.transaction import (
    PredictionEvidence,
    TransactionExplanation,
)
from fraud_intelligence.explainability.validation import (
    ExplanationValidationError,
    validate_complete_explanation,
    validate_explanation_features,
    validate_graph_temporal_context,
    validate_human_readable_leakage,
    validate_transaction_explanation_leakage,
)


TIMESTAMP = pd.Timestamp(
    "2025-01-01 12:00:00+00:00"
)


def _prediction() -> PredictionEvidence:
    return PredictionEvidence(
        transaction_id="txn_001",
        probability=0.91,
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
                attribution=0.8,
                absolute_attribution=0.8,
                rank=1,
            ),
        ),
    )


def _explanation() -> TransactionExplanation:
    return TransactionExplanation(
        transaction_id="txn_001",
        timestamp=TIMESTAMP,
        prediction=_prediction(),
        classical_attribution=_classical(),
        shap_explanation=None,
        graph_neighborhood=None,
        gnn_importance=None,
    )


def test_allowed_features_pass() -> None:
    validate_explanation_features(
        (
            "amount_deviation_from_customer_mean",
            "customer_txn_count_1m",
        )
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
def test_forbidden_features_rejected(
    forbidden: str,
) -> None:
    with pytest.raises(
        ExplanationValidationError,
        match="Forbidden explanation features",
    ):
        validate_explanation_features(
            ("amount_deviation_from_customer_mean", forbidden)
        )


def test_historical_graph_context_passes() -> None:
    report = validate_graph_temporal_context(
        target_timestamp=TIMESTAMP,
        graph_transaction_timestamps={
            1: pd.Timestamp(
                "2025-01-01 11:00:00+00:00"
            ),
            2: pd.Timestamp(
                "2025-01-01 10:00:00+00:00"
            ),
        },
        target_transaction_node_index=10,
    )

    assert report.has_leakage is False


def test_future_graph_context_rejected() -> None:
    with pytest.raises(
        ExplanationValidationError,
        match="future",
    ):
        validate_graph_temporal_context(
            target_timestamp=TIMESTAMP,
            graph_transaction_timestamps={
                1: pd.Timestamp(
                    "2025-01-01 13:00:00+00:00"
                ),
            },
            target_transaction_node_index=10,
        )


def test_same_timestamp_graph_context_rejected() -> None:
    with pytest.raises(
        ExplanationValidationError,
        match="same-timestamp",
    ):
        validate_graph_temporal_context(
            target_timestamp=TIMESTAMP,
            graph_transaction_timestamps={
                1: TIMESTAMP,
            },
            target_transaction_node_index=10,
        )


def test_target_transaction_in_context_rejected() -> None:
    with pytest.raises(
        ExplanationValidationError,
        match="target-transaction",
    ):
        validate_graph_temporal_context(
            target_timestamp=TIMESTAMP,
            graph_transaction_timestamps={
                10: TIMESTAMP,
            },
            target_transaction_node_index=10,
        )


def test_naive_target_timestamp_rejected() -> None:
    with pytest.raises(
        ExplanationValidationError,
        match="timezone-aware",
    ):
        validate_graph_temporal_context(
            target_timestamp=pd.Timestamp(
                "2025-01-01 12:00:00"
            ),
            graph_transaction_timestamps={},
            target_transaction_node_index=10,
        )


def test_transaction_explanation_leakage_passes() -> None:
    report = validate_transaction_explanation_leakage(
        _explanation()
    )

    assert report.transaction_id == "txn_001"
    assert report.has_leakage is False


def test_complete_explanation_validation_passes() -> None:
    human = HumanReadableExplanation(
        transaction_id="txn_001",
        summary="Transaction was flagged as suspicious.",
        risk_level="suspicious",
        reasons=(),
        graph_findings=(),
        supporting_evidence=(
            "classical feature attribution",
        ),
        limitations=(
            "The explanation describes model evidence and "
            "does not establish that fraud actually occurred.",
        ),
    )

    report = validate_complete_explanation(
        transaction_explanation=_explanation(),
        human_readable_explanation=human,
    )

    assert report.has_leakage is False


def test_human_readable_forbidden_field_rejected() -> None:
    human = HumanReadableExplanation(
        transaction_id="txn_001",
        summary="Customer customer_id contributed to the score.",
        risk_level="suspicious",
        reasons=(),
        graph_findings=(),
        supporting_evidence=(),
        limitations=(),
    )

    with pytest.raises(
        ExplanationValidationError,
        match="forbidden field",
    ):
        validate_human_readable_leakage(human)


def test_human_readable_clean_text_passes() -> None:
    human = HumanReadableExplanation(
        transaction_id="txn_001",
        summary="Transaction was flagged as suspicious.",
        risk_level="suspicious",
        reasons=(),
        graph_findings=(),
        supporting_evidence=(
            "classical feature attribution",
        ),
        limitations=(
            "The explanation describes model evidence.",
        ),
    )

    validate_human_readable_leakage(human)