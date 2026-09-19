from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraud_intelligence.explainability.transaction import (
    PredictionEvidence,
    TransactionExplanation,
)


class HumanReadableExplanationError(ValueError):
    """Raised when human-readable explanation generation fails."""


@dataclass(frozen=True)
class ExplanationReason:
    """One human-readable reason supporting the transaction assessment."""

    category: str
    feature: str
    direction: str
    attribution: float
    text: str
    rank: int


@dataclass(frozen=True)
class GraphFinding:
    """One human-readable finding derived from graph evidence."""

    category: str
    text: str
    node_type: str | None
    node_index: int | None
    importance: float | None


@dataclass(frozen=True)
class HumanReadableExplanation:
    """Human-readable representation of a transaction explanation."""

    transaction_id: str
    summary: str
    risk_level: str
    reasons: tuple[ExplanationReason, ...]
    graph_findings: tuple[GraphFinding, ...]
    supporting_evidence: tuple[str, ...]
    limitations: tuple[str, ...]

    @property
    def reason_count(self) -> int:
        return len(self.reasons)

    @property
    def graph_finding_count(self) -> int:
        return len(self.graph_findings)


def _validate_prediction(
    prediction: PredictionEvidence,
) -> None:
    if not isinstance(prediction, PredictionEvidence):
        raise HumanReadableExplanationError(
            "prediction must be a PredictionEvidence."
        )

    if not prediction.transaction_id:
        raise HumanReadableExplanationError(
            "Prediction transaction_id cannot be empty."
        )

    if not np.isfinite(prediction.probability):
        raise HumanReadableExplanationError(
            "Prediction probability must be finite."
        )

    if not 0.0 <= prediction.probability <= 1.0:
        raise HumanReadableExplanationError(
            "Prediction probability must be between 0 and 1."
        )

    if prediction.prediction not in (0, 1):
        raise HumanReadableExplanationError(
            "Prediction must be binary: 0 or 1."
        )


def _risk_level(
    prediction: PredictionEvidence,
) -> str:
    """
    Convert the existing binary model decision into a display label.

    No new probability threshold is introduced here.
    The prediction already reflects the model threshold from Phase 8.7.
    """
    if prediction.prediction == 1:
        return "suspicious"

    return "not_suspicious"


def _prediction_summary(
    prediction: PredictionEvidence,
) -> str:
    probability_pct = prediction.probability * 100.0

    if prediction.prediction == 1:
        return (
            f"Transaction {prediction.transaction_id} was flagged as "
            f"suspicious with a fraud probability of "
            f"{probability_pct:.2f}%."
        )

    return (
        f"Transaction {prediction.transaction_id} was not flagged as "
        f"suspicious. The model fraud probability was "
        f"{probability_pct:.2f}%."
    )


def _classical_reasons(
    explanation: TransactionExplanation,
) -> list[ExplanationReason]:
    reasons: list[ExplanationReason] = []

    attribution = explanation.classical_attribution

    if attribution is None:
        return reasons

    for item in attribution.feature_attributions:
        direction = (
            "increases"
            if item.attribution > 0
            else "decreases"
            if item.attribution < 0
            else "has no directional effect"
        )

        text = (
            f"{item.feature} {direction} the model score "
            f"(attribution {item.attribution:.4f})."
        )

        reasons.append(
            ExplanationReason(
                category="classical_attribution",
                feature=item.feature,
                direction=direction,
                attribution=float(item.attribution),
                text=text,
                rank=item.rank,
            )
        )

    return reasons


def _shap_reasons(
    explanation: TransactionExplanation,
) -> list[ExplanationReason]:
    reasons: list[ExplanationReason] = []

    shap_explanation = explanation.shap_explanation

    if shap_explanation is None:
        return reasons

    for item in shap_explanation.feature_attributions:
        direction = (
            "increases"
            if item.shap_value > 0
            else "decreases"
            if item.shap_value < 0
            else "has no directional effect"
        )

        text = (
            f"{item.feature} {direction} the model output "
            f"(SHAP value {item.shap_value:.4f})."
        )

        reasons.append(
            ExplanationReason(
                category="shap",
                feature=item.feature,
                direction=direction,
                attribution=float(item.shap_value),
                text=text,
                rank=item.rank,
            )
        )

    return reasons


def _graph_findings(
    explanation: TransactionExplanation,
) -> list[GraphFinding]:
    findings: list[GraphFinding] = []

    neighborhood = explanation.graph_neighborhood

    if neighborhood is not None:
        for node in neighborhood.neighbor_nodes:
            findings.append(
                GraphFinding(
                    category="graph_neighborhood",
                    text=(
                        f"Connected {node.node_type} node "
                        f"{node.node_index} was found at hop "
                        f"{node.hop} from the transaction."
                    ),
                    node_type=node.node_type,
                    node_index=node.node_index,
                    importance=None,
                )
            )

    importance = explanation.gnn_importance

    if importance is not None:
        for node in importance.node_importances:
            findings.append(
                GraphFinding(
                    category="gnn_node_importance",
                    text=(
                        f"Connected {node.node_type} node "
                        f"{node.node_index} has GNN importance "
                        f"{node.importance:.4f}."
                    ),
                    node_type=node.node_type,
                    node_index=node.node_index,
                    importance=float(node.importance),
                )
            )

    return findings


def _supporting_evidence(
    explanation: TransactionExplanation,
) -> tuple[str, ...]:
    evidence: list[str] = []

    if explanation.classical_attribution is not None:
        evidence.append("classical feature attribution")

    if explanation.shap_explanation is not None:
        evidence.append("SHAP feature attribution")

    if explanation.graph_neighborhood is not None:
        evidence.append("historical graph neighborhood")

    if explanation.gnn_importance is not None:
        evidence.append("GNN feature/node importance")

    return tuple(evidence)


def _limitations(
    explanation: TransactionExplanation,
) -> tuple[str, ...]:
    limitations: list[str] = []

    if explanation.classical_attribution is None:
        limitations.append(
            "No classical feature attribution was provided."
        )

    if explanation.shap_explanation is None:
        limitations.append(
            "No SHAP explanation was provided."
        )

    if explanation.graph_neighborhood is None:
        limitations.append(
            "No graph neighborhood evidence was provided."
        )

    if explanation.gnn_importance is None:
        limitations.append(
            "No GNN importance evidence was provided."
        )

    limitations.append(
        "The explanation describes model evidence and does not "
        "establish that fraud actually occurred."
    )

    return tuple(limitations)


def build_human_readable_explanation(
    explanation: TransactionExplanation,
) -> HumanReadableExplanation:
    """
    Convert a structured TransactionExplanation into analyst-readable text.

    This function only formats and summarizes existing evidence.
    It does not recalculate predictions, attributions, or graph scores.
    """
    if not isinstance(
        explanation,
        TransactionExplanation,
    ):
        raise HumanReadableExplanationError(
            "explanation must be a TransactionExplanation."
        )

    _validate_prediction(explanation.prediction)

    if (
        explanation.prediction.transaction_id
        != explanation.transaction_id
    ):
        raise HumanReadableExplanationError(
            "Prediction transaction_id does not match "
            "explanation transaction_id."
        )

    reasons = tuple(
        _classical_reasons(explanation)
        + _shap_reasons(explanation)
    )

    graph_findings = tuple(
        _graph_findings(explanation)
    )

    result = HumanReadableExplanation(
        transaction_id=explanation.transaction_id,
        summary=_prediction_summary(
            explanation.prediction
        ),
        risk_level=_risk_level(
            explanation.prediction
        ),
        reasons=reasons,
        graph_findings=graph_findings,
        supporting_evidence=_supporting_evidence(
            explanation
        ),
        limitations=_limitations(
            explanation
        ),
    )

    validate_human_readable_explanation(result)

    return result


def validate_human_readable_explanation(
    explanation: HumanReadableExplanation,
) -> None:
    """Validate a human-readable explanation."""

    if not isinstance(
        explanation,
        HumanReadableExplanation,
    ):
        raise HumanReadableExplanationError(
            "explanation must be a HumanReadableExplanation."
        )

    if not explanation.transaction_id:
        raise HumanReadableExplanationError(
            "transaction_id cannot be empty."
        )

    if explanation.risk_level not in (
        "suspicious",
        "not_suspicious",
    ):
        raise HumanReadableExplanationError(
            "risk_level must be 'suspicious' or "
            "'not_suspicious'."
        )

    if not explanation.summary:
        raise HumanReadableExplanationError(
            "summary cannot be empty."
        )

    for reason in explanation.reasons:
        if not isinstance(reason, ExplanationReason):
            raise HumanReadableExplanationError(
                "All reasons must be ExplanationReason objects."
            )

        if not reason.feature:
            raise HumanReadableExplanationError(
                "Explanation reason feature cannot be empty."
            )

        if not np.isfinite(reason.attribution):
            raise HumanReadableExplanationError(
                "Explanation attribution must be finite."
            )

        if reason.rank < 1:
            raise HumanReadableExplanationError(
                "Explanation reason rank must be positive."
            )

        if not reason.text:
            raise HumanReadableExplanationError(
                "Explanation reason text cannot be empty."
            )

    for finding in explanation.graph_findings:
        if not isinstance(finding, GraphFinding):
            raise HumanReadableExplanationError(
                "All graph findings must be GraphFinding objects."
            )

        if not finding.text:
            raise HumanReadableExplanationError(
                "Graph finding text cannot be empty."
            )

        if finding.importance is not None:
            if not np.isfinite(finding.importance):
                raise HumanReadableExplanationError(
                    "Graph finding importance must be finite."
                )

    for evidence in explanation.supporting_evidence:
        if not evidence:
            raise HumanReadableExplanationError(
                "Supporting evidence entries cannot be empty."
            )

    for limitation in explanation.limitations:
        if not limitation:
            raise HumanReadableExplanationError(
                "Limitation entries cannot be empty."
            )