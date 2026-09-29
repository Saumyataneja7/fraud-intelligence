from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .contracts import FraudIntelligenceContractError


# ---------------------------------------------------------------------------
# Forbidden fields
# ---------------------------------------------------------------------------

FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        "is_fraud",
        "fraud_scenario",
    }
)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuspicionSignal:
    """
    A single piece of transaction-level evidence contributing to
    suspiciousness.

    This is descriptive intelligence. It does not create or modify
    the underlying model prediction.
    """

    name: str
    value: float | int | bool | str
    description: str
    source: str


@dataclass(frozen=True)
class SuspiciousTransactionIntelligence:
    """
    Investigation-oriented intelligence for one transaction.

    prediction_probability and prediction_label come from an existing
    model/explanation pipeline. This module does not retrain or alter
    the model.
    """

    transaction_id: str
    timestamp: object
    prediction_probability: float
    prediction_label: int
    signals: tuple[SuspicionSignal, ...]
    entity_counts: tuple[tuple[str, int], ...]
    temporal_context_available: bool = True

    @property
    def signal_count(self) -> int:
        return len(self.signals)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_forbidden_fields(columns: Iterable[str]) -> None:
    """
    Reject predictive/label fields from intelligence input.
    """

    forbidden = sorted(set(columns).intersection(FORBIDDEN_FIELDS))

    if forbidden:
        raise FraudIntelligenceContractError(
            "Fraud intelligence input contains forbidden predictive "
            f"fields: {forbidden}"
        )


def _validate_probability(probability: float) -> None:
    if not 0.0 <= float(probability) <= 1.0:
        raise FraudIntelligenceContractError(
            "prediction_probability must be between 0 and 1."
        )


def _validate_prediction_label(label: int) -> None:
    if int(label) not in (0, 1):
        raise FraudIntelligenceContractError(
            "prediction_label must be either 0 or 1."
        )


def _validate_timestamp_order(
    target_timestamp,
    context_timestamps: Sequence,
) -> None:
    """
    Investigation context must be strictly earlier than the target event.

    Same-timestamp context is rejected because Phase 6/7/8 deliberately
    use strict temporal ordering when event ordering is unavailable.
    """

    for context_timestamp in context_timestamps:
        if context_timestamp >= target_timestamp:
            raise FraudIntelligenceContractError(
                "Investigation context must be strictly earlier than "
                "the target transaction timestamp."
            )


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------

def build_transaction_signals(
    transaction: Mapping[str, object],
) -> tuple[SuspicionSignal, ...]:
    """
    Build descriptive transaction-level suspicion signals.

    Expected optional transaction fields:

    - amount
    - amount_vs_customer_mean
    - amount_vs_customer_median
    - amount_vs_customer_max
    - amount_deviation_from_customer_mean
    - customer_txn_count_1m
    - customer_txn_count_5m
    - customer_txn_count_15m
    - customer_amount_sum_1m
    - customer_amount_sum_5m
    - customer_amount_sum_15m
    - customer_unique_merchants_5m
    - customer_unique_devices_5m
    - customer_unique_ips_5m
    - is_new_device
    - is_new_ip
    - is_new_merchant
    - is_new_card
    - is_new_payment_method
    - is_new_transaction_type

    The function is intentionally threshold-light: it translates
    already-engineered features into investigator-readable evidence.
    It does not introduce a second fraud classifier.
    """

    _validate_forbidden_fields(transaction.keys())

    signals: list[SuspicionSignal] = []

    def add_numeric_signal(
        field: str,
        description: str,
        *,
        predicate,
    ) -> None:
        if field not in transaction:
            return

        value = transaction[field]

        if value is None:
            return

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return

        if predicate(numeric_value):
            signals.append(
                SuspicionSignal(
                    name=field,
                    value=numeric_value,
                    description=description,
                    source="engineered_transaction_feature",
                )
            )

    def add_boolean_signal(
        field: str,
        description: str,
    ) -> None:
        if field not in transaction:
            return

        value = transaction[field]

        if value is True or value == 1:
            signals.append(
                SuspicionSignal(
                    name=field,
                    value=True,
                    description=description,
                    source="engineered_transaction_feature",
                )
            )

    # ------------------------------------------------------------------
    # Amount / behavioral signals
    # ------------------------------------------------------------------

    add_numeric_signal(
        "amount_vs_customer_mean",
        "Transaction amount is elevated relative to the customer's historical mean.",
        predicate=lambda value: value > 2.0,
    )

    add_numeric_signal(
        "amount_vs_customer_median",
        "Transaction amount is elevated relative to the customer's historical median.",
        predicate=lambda value: value > 2.0,
    )

    add_numeric_signal(
        "amount_vs_customer_max",
        "Transaction amount exceeds the customer's historical maximum.",
        predicate=lambda value: value > 1.0,
    )

    add_numeric_signal(
        "amount_deviation_from_customer_mean",
        "Transaction amount deviates materially from the customer's historical mean.",
        predicate=lambda value: abs(value) > 2.0,
    )

    # ------------------------------------------------------------------
    # Velocity signals
    # ------------------------------------------------------------------

    add_numeric_signal(
        "customer_txn_count_1m",
        "Customer has multiple transactions within the preceding one-minute window.",
        predicate=lambda value: value >= 2,
    )

    add_numeric_signal(
        "customer_txn_count_5m",
        "Customer has elevated transaction activity within the preceding five-minute window.",
        predicate=lambda value: value >= 3,
    )

    add_numeric_signal(
        "customer_txn_count_15m",
        "Customer has elevated transaction activity within the preceding fifteen-minute window.",
        predicate=lambda value: value >= 5,
    )

    # ------------------------------------------------------------------
    # Historical amount activity
    # ------------------------------------------------------------------

    add_numeric_signal(
        "customer_amount_sum_1m",
        "Customer has accumulated transaction value immediately before this transaction.",
        predicate=lambda value: value > 0,
    )

    add_numeric_signal(
        "customer_amount_sum_5m",
        "Customer has accumulated transaction value within the preceding five minutes.",
        predicate=lambda value: value > 0,
    )

    add_numeric_signal(
        "customer_amount_sum_15m",
        "Customer has accumulated transaction value within the preceding fifteen minutes.",
        predicate=lambda value: value > 0,
    )

    # ------------------------------------------------------------------
    # Entity diversity
    # ------------------------------------------------------------------

    add_numeric_signal(
        "customer_unique_merchants_5m",
        "Customer interacted with multiple merchants within the preceding five minutes.",
        predicate=lambda value: value >= 2,
    )

    add_numeric_signal(
        "customer_unique_devices_5m",
        "Customer used multiple devices within the preceding five minutes.",
        predicate=lambda value: value >= 2,
    )

    add_numeric_signal(
        "customer_unique_ips_5m",
        "Customer used multiple IP addresses within the preceding five minutes.",
        predicate=lambda value: value >= 2,
    )

    # ------------------------------------------------------------------
    # Novelty signals
    # ------------------------------------------------------------------

    add_boolean_signal(
        "is_new_device",
        "The device has not previously been observed for this customer.",
    )

    add_boolean_signal(
        "is_new_ip",
        "The IP address has not previously been observed for this customer.",
    )

    add_boolean_signal(
        "is_new_merchant",
        "The merchant has not previously been observed for this customer.",
    )

    add_boolean_signal(
        "is_new_card",
        "The card has not previously been observed for this customer.",
    )

    add_boolean_signal(
        "is_new_payment_method",
        "The payment method has not previously been observed for this customer.",
    )

    add_boolean_signal(
        "is_new_transaction_type",
        "The transaction type has not previously been observed for this customer.",
    )

    return tuple(signals)


# ---------------------------------------------------------------------------
# Intelligence assembly
# ---------------------------------------------------------------------------

def assemble_suspicious_transaction_intelligence(
    *,
    transaction_id: str,
    timestamp,
    prediction_probability: float,
    prediction_label: int,
    transaction_features: Mapping[str, object],
    entity_counts: Mapping[str, int] | None = None,
    context_timestamps: Sequence | None = None,
) -> SuspiciousTransactionIntelligence:
    """
    Assemble investigation intelligence for a single transaction.

    Parameters
    ----------
    transaction_id:
        Target transaction identifier.

    timestamp:
        Target transaction timestamp.

    prediction_probability:
        Probability produced by an already-trained model.

    prediction_label:
        Binary prediction produced by the existing model pipeline.

    transaction_features:
        Leakage-safe transaction features. Must not contain fraud labels.

    entity_counts:
        Optional descriptive counts such as the number of related
        customer/account/device/IP/merchant records.

    context_timestamps:
        Optional timestamps for context used in the investigation.
        Every timestamp must be strictly earlier than the target timestamp.
    """

    if not transaction_id:
        raise FraudIntelligenceContractError(
            "transaction_id must be non-empty."
        )

    _validate_probability(prediction_probability)
    _validate_prediction_label(prediction_label)

    _validate_forbidden_fields(transaction_features.keys())

    if context_timestamps is not None:
        _validate_timestamp_order(timestamp, context_timestamps)

    signals = build_transaction_signals(transaction_features)

    normalized_entity_counts: tuple[tuple[str, int], ...]

    if entity_counts is None:
        normalized_entity_counts = ()
    else:
        for entity_type, count in entity_counts.items():
            if int(count) < 0:
                raise FraudIntelligenceContractError(
                    f"Entity count for {entity_type!r} cannot be negative."
                )

        normalized_entity_counts = tuple(
            sorted(
                (
                    str(entity_type),
                    int(count),
                )
                for entity_type, count in entity_counts.items()
            )
        )

    return SuspiciousTransactionIntelligence(
        transaction_id=str(transaction_id),
        timestamp=timestamp,
        prediction_probability=float(prediction_probability),
        prediction_label=int(prediction_label),
        signals=signals,
        entity_counts=normalized_entity_counts,
        temporal_context_available=context_timestamps is not None,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_suspicious_transaction_intelligence(
    intelligence: SuspiciousTransactionIntelligence,
) -> None:
    """
    Validate a completed suspicious-transaction intelligence object.
    """

    if not intelligence.transaction_id:
        raise FraudIntelligenceContractError(
            "transaction_id must be non-empty."
        )

    _validate_probability(intelligence.prediction_probability)
    _validate_prediction_label(intelligence.prediction_label)

    signal_names = [signal.name for signal in intelligence.signals]

    if len(signal_names) != len(set(signal_names)):
        raise FraudIntelligenceContractError(
            "Suspicion signal names must be unique."
        )

    for signal in intelligence.signals:
        if signal.name in FORBIDDEN_FIELDS:
            raise FraudIntelligenceContractError(
                f"Forbidden field used as suspicion signal: {signal.name}"
            )

        if not signal.description:
            raise FraudIntelligenceContractError(
                "Every suspicion signal must have a description."
            )

        if not signal.source:
            raise FraudIntelligenceContractError(
                "Every suspicion signal must have a source."
            )

    entity_types = [entity_type for entity_type, _ in intelligence.entity_counts]

    if len(entity_types) != len(set(entity_types)):
        raise FraudIntelligenceContractError(
            "Entity types must be unique."
        )

    for entity_type, count in intelligence.entity_counts:
        if count < 0:
            raise FraudIntelligenceContractError(
                f"Entity count for {entity_type!r} cannot be negative."
            )