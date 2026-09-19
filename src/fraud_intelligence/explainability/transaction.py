from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_intelligence.explainability.classical import (
    ClassicalAttributionResult,
)
from fraud_intelligence.explainability.contracts import (
    ExplainabilityDataContract,
    ExplainabilityContractError,
    GNNExplanationContract,
)
from fraud_intelligence.explainability.gnn_importance import (
    GNNImportanceResult,
)
from fraud_intelligence.explainability.graph_neighborhood import (
    GNNNeighborhoodExplanation,
)
from fraud_intelligence.explainability.shap import (
    SHAPExplanation,
)


class TransactionExplanationError(ValueError):
    """Raised when transaction explanation assembly fails."""


@dataclass(frozen=True)
class PredictionEvidence:
    """Prediction information associated with a transaction."""

    transaction_id: str
    probability: float
    prediction: int
    threshold: float
    model_name: str

    @property
    def is_suspicious(self) -> bool:
        return self.prediction == 1


@dataclass(frozen=True)
class TransactionExplanation:
    """Complete structured explanation for one transaction."""

    transaction_id: str
    timestamp: pd.Timestamp
    prediction: PredictionEvidence

    classical_attribution: ClassicalAttributionResult | None
    shap_explanation: SHAPExplanation | None

    graph_neighborhood: GNNNeighborhoodExplanation | None
    gnn_importance: GNNImportanceResult | None

    @property
    def has_classical_evidence(self) -> bool:
        return (
            self.classical_attribution is not None
            or self.shap_explanation is not None
        )

    @property
    def has_graph_evidence(self) -> bool:
        return (
            self.graph_neighborhood is not None
            or self.gnn_importance is not None
        )

    @property
    def evidence_types(self) -> tuple[str, ...]:
        evidence: list[str] = []

        if self.classical_attribution is not None:
            evidence.append("classical_attribution")

        if self.shap_explanation is not None:
            evidence.append("shap")

        if self.graph_neighborhood is not None:
            evidence.append("graph_neighborhood")

        if self.gnn_importance is not None:
            evidence.append("gnn_importance")

        return tuple(evidence)


def _validate_prediction(
    prediction: PredictionEvidence,
) -> None:
    if not isinstance(
        prediction,
        PredictionEvidence,
    ):
        raise TransactionExplanationError(
            "prediction must be a PredictionEvidence."
        )

    if not prediction.transaction_id:
        raise TransactionExplanationError(
            "transaction_id cannot be empty."
        )

    if not np.isfinite(prediction.probability):
        raise TransactionExplanationError(
            "Prediction probability must be finite."
        )

    if not 0.0 <= prediction.probability <= 1.0:
        raise TransactionExplanationError(
            "Prediction probability must be between 0 and 1."
        )

    if prediction.prediction not in (0, 1):
        raise TransactionExplanationError(
            "Prediction must be binary: 0 or 1."
        )

    if not np.isfinite(prediction.threshold):
        raise TransactionExplanationError(
            "Prediction threshold must be finite."
        )

    if not 0.0 <= prediction.threshold <= 1.0:
        raise TransactionExplanationError(
            "Prediction threshold must be between 0 and 1."
        )

    if not prediction.model_name:
        raise TransactionExplanationError(
            "model_name cannot be empty."
        )


def _validate_timestamp(
    timestamp: pd.Timestamp,
) -> None:
    if not isinstance(
        timestamp,
        pd.Timestamp,
    ):
        raise TransactionExplanationError(
            "timestamp must be a pandas Timestamp."
        )

    if timestamp.tzinfo is None:
        raise TransactionExplanationError(
            "timestamp must be timezone-aware."
        )


def _validate_transaction_id_consistency(
    *,
    transaction_id: str,
    prediction: PredictionEvidence,
) -> None:
    if prediction.transaction_id != transaction_id:
        raise TransactionExplanationError(
            "Prediction transaction_id does not match "
            "explanation transaction_id."
        )


def _validate_graph_consistency(
    *,
    timestamp: pd.Timestamp,
    graph_neighborhood: GNNNeighborhoodExplanation | None,
    gnn_importance: GNNImportanceResult | None,
) -> None:
    """
    Validate consistency between graph explanation components.

    The neighborhood timestamp represents the timestamp of the
    target transaction itself, so it must equal the transaction
    explanation timestamp.

    Temporal filtering of historical graph context is owned by
    Phase 8.5 and is not duplicated here.
    """
    if graph_neighborhood is not None:
        if graph_neighborhood.transaction_timestamp != timestamp:
            raise TransactionExplanationError(
                "Graph neighborhood target timestamp does not "
                "match transaction timestamp."
            )

        # Ensure the target transaction node is actually
        # present in the returned neighborhood.
        target_present = any(
            node.node_type == "transaction"
            and node.node_index
            == graph_neighborhood.transaction_node_index
            for node in graph_neighborhood.nodes
        )

        if not target_present:
            raise TransactionExplanationError(
                "Graph neighborhood must contain the target "
                "transaction node."
            )

    if (
        graph_neighborhood is not None
        and gnn_importance is not None
    ):
        if (
            graph_neighborhood.transaction_node_index
            != gnn_importance.transaction_node_index
        ):
            raise TransactionExplanationError(
                "GNN importance transaction node does not "
                "match graph neighborhood target."
            )


def validate_transaction_explanation(
    explanation: TransactionExplanation,
    contract: ExplainabilityDataContract | None = None,
    graph_contract: GNNExplanationContract | None = None,
) -> None:
    """
    Validate the complete assembled explanation.

    This function validates structure and consistency only.
    It does not validate whether the model itself is correct.
    """
    if contract is None:
        contract = ExplainabilityDataContract()

    if graph_contract is None:
        graph_contract = GNNExplanationContract()

    if not isinstance(
        explanation,
        TransactionExplanation,
    ):
        raise TransactionExplanationError(
            "explanation must be a TransactionExplanation."
        )

    if not explanation.transaction_id:
        raise TransactionExplanationError(
            "transaction_id cannot be empty."
        )

    _validate_timestamp(
        explanation.timestamp
    )

    _validate_prediction(
        explanation.prediction
    )

    _validate_transaction_id_consistency(
        transaction_id=explanation.transaction_id,
        prediction=explanation.prediction,
    )

    if (
        explanation.classical_attribution is None
        and explanation.shap_explanation is None
        and explanation.graph_neighborhood is None
        and explanation.gnn_importance is None
    ):
        raise TransactionExplanationError(
            "At least one explanation evidence source "
            "must be provided."
        )

    _validate_graph_consistency(
        timestamp=explanation.timestamp,
        graph_neighborhood=(
            explanation.graph_neighborhood
        ),
        gnn_importance=explanation.gnn_importance,
    )

    if explanation.gnn_importance is not None:
        if (
            explanation.gnn_importance.transaction_node_index
            < 0
        ):
            raise TransactionExplanationError(
                "GNN transaction node index cannot be negative."
            )


def assemble_transaction_explanation(
    *,
    transaction_id: str,
    timestamp: pd.Timestamp,
    prediction: PredictionEvidence,
    classical_attribution: ClassicalAttributionResult | None = None,
    shap_explanation: SHAPExplanation | None = None,
    graph_neighborhood: GNNNeighborhoodExplanation | None = None,
    gnn_importance: GNNImportanceResult | None = None,
    contract: ExplainabilityDataContract | None = None,
    graph_contract: GNNExplanationContract | None = None,
) -> TransactionExplanation:
    """
    Assemble all available explanation evidence for one transaction.

    No prediction or attribution is recalculated here.
    """
    if contract is None:
        contract = ExplainabilityDataContract()

    if graph_contract is None:
        graph_contract = GNNExplanationContract()

    if not isinstance(
        transaction_id,
        str,
    ) or not transaction_id:
        raise TransactionExplanationError(
            "transaction_id must be a non-empty string."
        )

    _validate_timestamp(timestamp)

    _validate_prediction(prediction)

    _validate_transaction_id_consistency(
        transaction_id=transaction_id,
        prediction=prediction,
    )

    explanation = TransactionExplanation(
        transaction_id=transaction_id,
        timestamp=timestamp,
        prediction=prediction,
        classical_attribution=classical_attribution,
        shap_explanation=shap_explanation,
        graph_neighborhood=graph_neighborhood,
        gnn_importance=gnn_importance,
    )

    validate_transaction_explanation(
        explanation,
        contract=contract,
        graph_contract=graph_contract,
    )

    return explanation