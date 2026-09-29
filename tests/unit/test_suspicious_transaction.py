from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fraud_intelligence.intelligence.contracts import (
    FraudIntelligenceContractError,
)
from fraud_intelligence.intelligence.suspicious_transaction import (
    SuspicionSignal,
    SuspiciousTransactionIntelligence,
    assemble_suspicious_transaction_intelligence,
    build_transaction_signals,
    validate_suspicious_transaction_intelligence,
)


TARGET_TIMESTAMP = datetime(
    2025,
    6,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)


def test_build_transaction_signals_detects_behavioral_signals():
    features = {
        "amount_vs_customer_mean": 3.2,
        "amount_vs_customer_median": 2.5,
        "amount_vs_customer_max": 1.4,
        "amount_deviation_from_customer_mean": 2.8,
        "customer_txn_count_1m": 4,
        "customer_txn_count_5m": 5,
        "customer_txn_count_15m": 7,
        "customer_unique_merchants_5m": 2,
        "customer_unique_devices_5m": 3,
        "customer_unique_ips_5m": 2,
    }

    signals = build_transaction_signals(features)

    names = {signal.name for signal in signals}

    assert "amount_vs_customer_mean" in names
    assert "amount_vs_customer_median" in names
    assert "amount_deviation_from_customer_mean" in names
    assert "customer_txn_count_1m" in names
    assert "customer_txn_count_5m" in names
    assert "customer_txn_count_15m" in names
    assert "customer_unique_devices_5m" in names
    assert "customer_unique_ips_5m" in names


def test_build_transaction_signals_detects_novelty():
    features = {
        "is_new_device": True,
        "is_new_ip": True,
        "is_new_merchant": True,
        "is_new_card": True,
        "is_new_payment_method": True,
        "is_new_transaction_type": True,
    }

    signals = build_transaction_signals(features)

    names = {signal.name for signal in signals}

    assert names == {
        "is_new_device",
        "is_new_ip",
        "is_new_merchant",
        "is_new_card",
        "is_new_payment_method",
        "is_new_transaction_type",
    }


def test_build_transaction_signals_ignores_non_suspicious_values():
    features = {
        "amount_vs_customer_mean": 1.1,
        "amount_vs_customer_median": 1.0,
        "amount_vs_customer_max": 0.9,
        "amount_deviation_from_customer_mean": 0.5,
        "customer_txn_count_1m": 1,
        "customer_txn_count_5m": 2,
        "customer_txn_count_15m": 3,
        "customer_unique_merchants_5m": 1,
        "customer_unique_devices_5m": 1,
        "customer_unique_ips_5m": 1,
        "is_new_device": False,
        "is_new_ip": False,
    }

    signals = build_transaction_signals(features)

    assert signals == ()


def test_forbidden_fraud_label_is_rejected():
    with pytest.raises(FraudIntelligenceContractError):
        build_transaction_signals(
            {
                "amount_vs_customer_mean": 3.0,
                "is_fraud": 1,
            }
        )


def test_forbidden_fraud_scenario_is_rejected():
    with pytest.raises(FraudIntelligenceContractError):
        build_transaction_signals(
            {
                "amount_vs_customer_mean": 3.0,
                "fraud_scenario": "FRAUD_RING",
            }
        )


def test_probability_must_be_between_zero_and_one():
    with pytest.raises(FraudIntelligenceContractError):
        assemble_suspicious_transaction_intelligence(
            transaction_id="txn-1",
            timestamp=TARGET_TIMESTAMP,
            prediction_probability=1.2,
            prediction_label=1,
            transaction_features={},
        )


@pytest.mark.parametrize("label", [-1, 2, 10])
def test_prediction_label_must_be_binary(label):
    with pytest.raises(FraudIntelligenceContractError):
        assemble_suspicious_transaction_intelligence(
            transaction_id="txn-1",
            timestamp=TARGET_TIMESTAMP,
            prediction_probability=0.8,
            prediction_label=label,
            transaction_features={},
        )


def test_transaction_id_must_be_non_empty():
    with pytest.raises(FraudIntelligenceContractError):
        assemble_suspicious_transaction_intelligence(
            transaction_id="",
            timestamp=TARGET_TIMESTAMP,
            prediction_probability=0.8,
            prediction_label=1,
            transaction_features={},
        )


def test_future_context_is_rejected():
    future_timestamp = datetime(
        2025,
        6,
        1,
        12,
        0,
        1,
        tzinfo=timezone.utc,
    )

    with pytest.raises(FraudIntelligenceContractError):
        assemble_suspicious_transaction_intelligence(
            transaction_id="txn-1",
            timestamp=TARGET_TIMESTAMP,
            prediction_probability=0.8,
            prediction_label=1,
            transaction_features={},
            context_timestamps=[future_timestamp],
        )


def test_same_timestamp_context_is_rejected():
    with pytest.raises(FraudIntelligenceContractError):
        assemble_suspicious_transaction_intelligence(
            transaction_id="txn-1",
            timestamp=TARGET_TIMESTAMP,
            prediction_probability=0.8,
            prediction_label=1,
            transaction_features={},
            context_timestamps=[TARGET_TIMESTAMP],
        )


def test_historical_context_is_allowed():
    historical_timestamp = datetime(
        2025,
        5,
        31,
        23,
        59,
        59,
        tzinfo=timezone.utc,
    )

    intelligence = assemble_suspicious_transaction_intelligence(
        transaction_id="txn-1",
        timestamp=TARGET_TIMESTAMP,
        prediction_probability=0.8,
        prediction_label=1,
        transaction_features={
            "amount_vs_customer_mean": 3.0,
            "is_new_device": True,
        },
        context_timestamps=[historical_timestamp],
    )

    assert intelligence.temporal_context_available is True
    assert intelligence.signal_count == 2


def test_entity_counts_are_normalized():
    intelligence = assemble_suspicious_transaction_intelligence(
        transaction_id="txn-1",
        timestamp=TARGET_TIMESTAMP,
        prediction_probability=0.7,
        prediction_label=1,
        transaction_features={},
        entity_counts={
            "device": 2,
            "customer": 1,
            "ip": 3,
        },
    )

    assert intelligence.entity_counts == (
        ("customer", 1),
        ("device", 2),
        ("ip", 3),
    )


def test_negative_entity_count_is_rejected():
    with pytest.raises(FraudIntelligenceContractError):
        assemble_suspicious_transaction_intelligence(
            transaction_id="txn-1",
            timestamp=TARGET_TIMESTAMP,
            prediction_probability=0.7,
            prediction_label=1,
            transaction_features={},
            entity_counts={"device": -1},
        )


def test_signal_source_is_engineered_feature():
    signals = build_transaction_signals(
        {
            "amount_vs_customer_mean": 3.0,
        }
    )

    assert len(signals) == 1
    assert signals[0].source == "engineered_transaction_feature"


def test_signal_descriptions_are_present():
    signals = build_transaction_signals(
        {
            "is_new_device": True,
            "customer_txn_count_1m": 3,
        }
    )

    assert all(signal.description for signal in signals)


def test_signal_names_are_unique():
    intelligence = SuspiciousTransactionIntelligence(
        transaction_id="txn-1",
        timestamp=TARGET_TIMESTAMP,
        prediction_probability=0.8,
        prediction_label=1,
        signals=(
            SuspicionSignal(
                name="duplicate",
                value=True,
                description="first",
                source="test",
            ),
            SuspicionSignal(
                name="duplicate",
                value=True,
                description="second",
                source="test",
            ),
        ),
        entity_counts=(),
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_suspicious_transaction_intelligence(intelligence)


def test_validation_accepts_valid_intelligence():
    intelligence = assemble_suspicious_transaction_intelligence(
        transaction_id="txn-1",
        timestamp=TARGET_TIMESTAMP,
        prediction_probability=0.82,
        prediction_label=1,
        transaction_features={
            "amount_vs_customer_mean": 3.1,
            "customer_txn_count_1m": 4,
            "is_new_device": True,
        },
        entity_counts={
            "customer": 1,
            "device": 2,
        },
    )

    validate_suspicious_transaction_intelligence(intelligence)


def test_validation_rejects_forbidden_signal_name():
    intelligence = SuspiciousTransactionIntelligence(
        transaction_id="txn-1",
        timestamp=TARGET_TIMESTAMP,
        prediction_probability=0.8,
        prediction_label=1,
        signals=(
            SuspicionSignal(
                name="is_fraud",
                value=True,
                description="forbidden",
                source="test",
            ),
        ),
        entity_counts=(),
    )

    with pytest.raises(FraudIntelligenceContractError):
        validate_suspicious_transaction_intelligence(intelligence)


def test_empty_features_produce_no_signals():
    signals = build_transaction_signals({})

    assert signals == ()


def test_none_values_are_ignored():
    signals = build_transaction_signals(
        {
            "amount_vs_customer_mean": None,
            "customer_txn_count_1m": None,
            "is_new_device": None,
        }
    )

    assert signals == ()