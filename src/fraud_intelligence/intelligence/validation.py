from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .contracts import FraudIntelligenceContractError
from .entity_investigation import EntityInvestigationView
from .entity_relationships import EntityRelationship
from .fraud_ring_evidence import FraudRingEvidence
from .ring_investigation import FraudRingInvestigationView
from .suspicious_transaction import SuspiciousTransactionIntelligence
from .transaction_investigation import TransactionInvestigationView


FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        "is_fraud",
        "fraud_scenario",
    }
)

FORBIDDEN_IDENTIFIER_FIELDS: frozenset[str] = frozenset(
    {
        "transaction_id",
        "customer_id",
        "account_id",
        "card_id",
        "merchant_id",
        "device_id",
        "ip_id",
        "node_id",
    }
)


@dataclass(frozen=True)
class IntelligenceLeakageIssue:
    """
    One detected leakage or contract violation.
    """

    code: str
    message: str
    component: str


@dataclass(frozen=True)
class IntelligenceValidationReport:
    """
    Aggregate validation result for Phase 9 intelligence artifacts.
    """

    passed: bool
    issues: tuple[IntelligenceLeakageIssue, ...]
    checked_components: tuple[str, ...]

    @property
    def issue_count(self) -> int:
        return len(self.issues)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _contains_forbidden_field(value: object) -> str | None:
    """
    Recursively inspect common containers for forbidden predictive fields.
    """

    if isinstance(value, str):
        lowered = value.lower()

        for field in FORBIDDEN_FIELDS:
            if field in lowered:
                return field

        return None

    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()

            for field in FORBIDDEN_FIELDS:
                if field in key_text:
                    return field

            found = _contains_forbidden_field(item)

            if found is not None:
                return found

        return None

    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            found = _contains_forbidden_field(item)

            if found is not None:
                return found

    return None


def validate_no_forbidden_predictive_fields(
    value: object,
    *,
    component: str,
) -> tuple[IntelligenceLeakageIssue, ...]:
    """
    Check an intelligence artifact for forbidden target-derived fields.
    """

    found = _contains_forbidden_field(value)

    if found is None:
        return ()

    return (
        IntelligenceLeakageIssue(
            code="IL001",
            message=(
                f"Forbidden predictive field {found!r} "
                f"detected in {component}."
            ),
            component=component,
        ),
    )


# ---------------------------------------------------------------------------
# Temporal validation
# ---------------------------------------------------------------------------


def validate_relationship_temporal_context(
    *,
    relationships: Sequence[EntityRelationship],
    target_timestamp,
    component: str,
) -> tuple[IntelligenceLeakageIssue, ...]:
    """
    Ensure timestamped investigation relationships are strictly historical.

    Untimestamped relationships are allowed because the graph contains
    static structural relationships that do not represent an event time.
    """

    issues: list[IntelligenceLeakageIssue] = []

    for relationship in relationships:
        relationship_timestamp = relationship.timestamp

        if relationship_timestamp is None:
            continue

        if relationship_timestamp >= target_timestamp:
            code = (
                "IL003"
                if relationship_timestamp == target_timestamp
                else "IL002"
            )

            description = (
                "same-timestamp context is not allowed"
                if relationship_timestamp == target_timestamp
                else "future context is not allowed"
            )

            issues.append(
                IntelligenceLeakageIssue(
                    code=code,
                    message=(
                        f"Relationship timestamp "
                        f"{relationship_timestamp!r} violates the "
                        f"strict historical rule: {description}."
                    ),
                    component=component,
                )
            )

    return tuple(issues)


# ---------------------------------------------------------------------------
# Suspicious transaction validation
# ---------------------------------------------------------------------------


def validate_suspicious_transaction_intelligence_leakage(
    intelligence: SuspiciousTransactionIntelligence,
) -> tuple[IntelligenceLeakageIssue, ...]:
    issues = list(
        validate_no_forbidden_predictive_fields(
            intelligence,
            component="suspicious_transaction",
        )
    )

    if not 0.0 <= intelligence.prediction_probability <= 1.0:
        issues.append(
            IntelligenceLeakageIssue(
                code="IL004",
                message="Prediction probability is outside [0, 1].",
                component="suspicious_transaction",
            )
        )

    if intelligence.prediction_label not in (0, 1):
        issues.append(
            IntelligenceLeakageIssue(
                code="IL005",
                message="Prediction label must be 0 or 1.",
                component="suspicious_transaction",
            )
        )

    return tuple(issues)


# ---------------------------------------------------------------------------
# Transaction investigation validation
# ---------------------------------------------------------------------------


def validate_transaction_investigation_leakage(
    view: TransactionInvestigationView,
) -> tuple[IntelligenceLeakageIssue, ...]:
    issues = list(
        validate_no_forbidden_predictive_fields(
            view,
            component="transaction_investigation",
        )
    )

    issues.extend(
        validate_relationship_temporal_context(
            relationships=view.relationships,
            target_timestamp=view.timestamp,
            component="transaction_investigation",
        )
    )

    relationship_transactions = {
        relationship.source.entity_id
        for relationship in view.relationships
        if relationship.source.entity_type == "transaction"
    }

    relationship_transactions.update(
        relationship.target.entity_id
        for relationship in view.relationships
        if relationship.target.entity_type == "transaction"
    )

    if view.relationships and view.transaction_id not in (
        relationship_transactions
    ):
        issues.append(
            IntelligenceLeakageIssue(
                code="IL006",
                message=(
                    "Transaction investigation relationships do not "
                    "contain the target transaction."
                ),
                component="transaction_investigation",
            )
        )

    return tuple(issues)


# ---------------------------------------------------------------------------
# Entity investigation validation
# ---------------------------------------------------------------------------


def validate_entity_investigation_leakage(
    view: EntityInvestigationView,
    *,
    target_timestamp=None,
) -> tuple[IntelligenceLeakageIssue, ...]:
    issues = list(
        validate_no_forbidden_predictive_fields(
            view,
            component="entity_investigation",
        )
    )

    if target_timestamp is not None:
        for related in view.related_entities:
            if related.timestamp is None:
                continue

            if related.timestamp >= target_timestamp:
                code = (
                    "IL003"
                    if related.timestamp == target_timestamp
                    else "IL002"
                )

                issues.append(
                    IntelligenceLeakageIssue(
                        code=code,
                        message=(
                            f"Related entity timestamp "
                            f"{related.timestamp!r} is not strictly "
                            "historical."
                        ),
                        component="entity_investigation",
                    )
                )

    return tuple(issues)


# ---------------------------------------------------------------------------
# Ring evidence validation
# ---------------------------------------------------------------------------


def validate_ring_evidence_leakage(
    evidence: FraudRingEvidence,
    *,
    target_timestamp=None,
) -> tuple[IntelligenceLeakageIssue, ...]:
    issues = list(
        validate_no_forbidden_predictive_fields(
            evidence,
            component="fraud_ring_evidence",
        )
    )

    if target_timestamp is not None:
        issues.extend(
            validate_relationship_temporal_context(
                relationships=evidence.relationships,
                target_timestamp=target_timestamp,
                component="fraud_ring_evidence",
            )
        )

    if evidence.network_score.candidate_id != evidence.candidate_id:
        issues.append(
            IntelligenceLeakageIssue(
                code="IL007",
                message="Evidence network score candidate ID mismatch.",
                component="fraud_ring_evidence",
            )
        )

    return tuple(issues)


# ---------------------------------------------------------------------------
# Ring investigation validation
# ---------------------------------------------------------------------------


def validate_ring_investigation_leakage(
    view: FraudRingInvestigationView,
    *,
    target_timestamp=None,
) -> tuple[IntelligenceLeakageIssue, ...]:
    issues = list(
        validate_no_forbidden_predictive_fields(
            view,
            component="ring_investigation",
        )
    )

    if target_timestamp is not None:
        issues.extend(
            validate_relationship_temporal_context(
                relationships=view.relationships,
                target_timestamp=target_timestamp,
                component="ring_investigation",
            )
        )

    if view.evidence is not None:
        issues.extend(
            validate_ring_evidence_leakage(
                view.evidence,
                target_timestamp=target_timestamp,
            )
        )

    return tuple(issues)


# ---------------------------------------------------------------------------
# Cross-component consistency
# ---------------------------------------------------------------------------


def validate_cross_component_consistency(
    *,
    transaction_intelligence: SuspiciousTransactionIntelligence,
    transaction_view: TransactionInvestigationView,
) -> tuple[IntelligenceLeakageIssue, ...]:
    issues: list[IntelligenceLeakageIssue] = []

    if (
        transaction_intelligence.transaction_id
        != transaction_view.transaction_id
    ):
        issues.append(
            IntelligenceLeakageIssue(
                code="IL008",
                message=(
                    "Suspicious transaction and investigation view "
                    "transaction IDs do not match."
                ),
                component="cross_component",
            )
        )

    if (
        transaction_intelligence.timestamp
        != transaction_view.timestamp
    ):
        issues.append(
            IntelligenceLeakageIssue(
                code="IL009",
                message=(
                    "Suspicious transaction and investigation view "
                    "timestamps do not match."
                ),
                component="cross_component",
            )
        )

    if (
        transaction_intelligence.prediction_probability
        != transaction_view.prediction_probability
    ):
        issues.append(
            IntelligenceLeakageIssue(
                code="IL010",
                message=(
                    "Transaction prediction probability changed "
                    "between intelligence components."
                ),
                component="cross_component",
            )
        )

    if (
        transaction_intelligence.prediction_label
        != transaction_view.prediction_label
    ):
        issues.append(
            IntelligenceLeakageIssue(
                code="IL011",
                message=(
                    "Transaction prediction label changed "
                    "between intelligence components."
                ),
                component="cross_component",
            )
        )

    return tuple(issues)


# ---------------------------------------------------------------------------
# Complete validation
# ---------------------------------------------------------------------------


def validate_fraud_intelligence(
    *,
    transaction_intelligence: SuspiciousTransactionIntelligence | None = None,
    transaction_view: TransactionInvestigationView | None = None,
    entity_views: Iterable[EntityInvestigationView] = (),
    ring_evidence: Iterable[FraudRingEvidence] = (),
    ring_views: Iterable[FraudRingInvestigationView] = (),
    target_timestamp=None,
) -> IntelligenceValidationReport:
    """
    Run the complete Phase 9 intelligence validation suite.
    """

    issues: list[IntelligenceLeakageIssue] = []
    checked_components: list[str] = []

    if transaction_intelligence is not None:
        checked_components.append(
            "suspicious_transaction"
        )

        issues.extend(
            validate_suspicious_transaction_intelligence_leakage(
                transaction_intelligence
            )
        )

    if transaction_view is not None:
        checked_components.append(
            "transaction_investigation"
        )

        issues.extend(
            validate_transaction_investigation_leakage(
                transaction_view
            )
        )

    entity_views = tuple(entity_views)

    if entity_views:
        checked_components.append(
            "entity_investigation"
        )

        for view in entity_views:
            issues.extend(
                validate_entity_investigation_leakage(
                    view,
                    target_timestamp=target_timestamp,
                )
            )

    ring_evidence = tuple(ring_evidence)

    if ring_evidence:
        checked_components.append(
            "fraud_ring_evidence"
        )

        for evidence in ring_evidence:
            issues.extend(
                validate_ring_evidence_leakage(
                    evidence,
                    target_timestamp=target_timestamp,
                )
            )

    ring_views = tuple(ring_views)

    if ring_views:
        checked_components.append(
            "ring_investigation"
        )

        for view in ring_views:
            issues.extend(
                validate_ring_investigation_leakage(
                    view,
                    target_timestamp=target_timestamp,
                )
            )

    if (
        transaction_intelligence is not None
        and transaction_view is not None
    ):
        checked_components.append(
            "cross_component"
        )

        issues.extend(
            validate_cross_component_consistency(
                transaction_intelligence=transaction_intelligence,
                transaction_view=transaction_view,
            )
        )

    return IntelligenceValidationReport(
        passed=not issues,
        issues=tuple(issues),
        checked_components=tuple(
            sorted(set(checked_components))
        ),
    )


def assert_fraud_intelligence_valid(
    report: IntelligenceValidationReport,
) -> None:
    """
    Raise a contract error when the validation report contains issues.
    """

    if report.passed:
        return

    formatted = "\n".join(
        (
            f"[{issue.code}] "
            f"{issue.component}: "
            f"{issue.message}"
        )
        for issue in report.issues
    )

    raise FraudIntelligenceContractError(
        "Fraud intelligence validation failed:\n"
        + formatted
    )