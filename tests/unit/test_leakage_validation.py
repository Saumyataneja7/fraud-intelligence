import pandas as pd
import pytest

from fraud_intelligence.features.leakage_validation import (
    LeakageValidationError,
    assert_feature_dataset_is_leakage_safe,
    validate_feature_dataset,
    validate_no_identifier_features,
    validate_no_target_features,
)
from fraud_intelligence.features.pipeline import build_feature_dataset


def make_transactions():
    return pd.DataFrame(
        {
            "transaction_id": ["T1", "T2", "T3", "T4", "T5"],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 10:00:00+00:00",
                    "2025-01-01 11:00:00+00:00",
                    "2025-01-01 12:00:00+00:00",
                    "2025-01-01 12:00:00+00:00",
                    "2025-01-01 13:00:00+00:00",
                ],
                utc=True,
            ),
            "customer_id": ["C1", "C1", "C1", "C1", "C2"],
            "account_id": ["A1", "A1", "A1", "A1", "A2"],
            "card_id": ["CARD1", "CARD1", "CARD1", "CARD2", "CARD3"],
            "merchant_id": ["M1", "M1", "M2", "M2", "M3"],
            "device_id": ["D1", "D1", "D2", "D2", "D3"],
            "ip_id": ["IP1", "IP1", "IP2", "IP2", "IP3"],
            "amount": [20.0, 40.0, 100.0, 200.0, 500.0],
            "currency": ["USD"] * 5,
            "payment_method": [
                "card",
                "card",
                "wallet",
                "wallet",
                "bank_transfer",
            ],
            "transaction_type": [
                "purchase",
                "purchase",
                "transfer",
                "transfer",
                "payment",
            ],
            "is_fraud": [0, 0, 0, 1, 0],
            "fraud_scenario": [
                "",
                "",
                "",
                "SHARED_DEVICE",
                "",
            ],
        }
    )


def test_target_columns_are_rejected_as_model_features():
    finding = validate_no_target_features(
        [
            "log_amount",
            "is_fraud",
        ]
    )

    assert finding.passed is False
    assert finding.rule_id == "LV001"


def test_identifier_columns_are_rejected_as_model_features():
    finding = validate_no_identifier_features(
        [
            "log_amount",
            "customer_id",
        ]
    )

    assert finding.passed is False
    assert finding.rule_id == "LV002"


def test_valid_pipeline_passes_leakage_validation():
    transactions = make_transactions()
    features = build_feature_dataset(transactions)

    findings = validate_feature_dataset(
        original=transactions,
        features=features,
    )

    assert all(finding.passed for finding in findings)


def test_pipeline_is_leakage_safe():
    transactions = make_transactions()
    features = build_feature_dataset(transactions)

    assert_feature_dataset_is_leakage_safe(
        original=transactions,
        features=features,
    )


def test_same_timestamp_transactions_do_not_leak():
    transactions = make_transactions()
    features = build_feature_dataset(transactions)

    rows = features[
        features["timestamp"]
        == pd.Timestamp(
            "2025-01-01 12:00:00+00:00"
        )
    ]

    assert rows["customer_txn_count_before"].tolist() == [2, 2]
    assert rows["customer_amount_sum_before"].tolist() == [
        60.0,
        60.0,
    ]
    assert rows["customer_amount_mean_before"].tolist() == pytest.approx(
        [30.0, 30.0]
    )


def test_future_transaction_does_not_change_previous_history():
    transactions = make_transactions()

    baseline = build_feature_dataset(
        transactions.iloc[:3].copy()
    )

    extended = build_feature_dataset(
        transactions.copy()
    )

    baseline_t3 = baseline.loc[
        baseline["transaction_id"] == "T3",
        "customer_amount_sum_before",
    ].iloc[0]

    extended_t3 = extended.loc[
        extended["transaction_id"] == "T3",
        "customer_amount_sum_before",
    ].iloc[0]

    assert baseline_t3 == extended_t3 == 60.0


def test_row_count_is_preserved():
    transactions = make_transactions()
    features = build_feature_dataset(transactions)

    findings = validate_feature_dataset(
        original=transactions,
        features=features,
    )

    row_count_finding = next(
        finding
        for finding in findings
        if finding.rule_id == "LV005"
    )

    assert row_count_finding.passed is True


def test_duplicate_transaction_ids_are_detected():
    transactions = make_transactions()
    features = build_feature_dataset(transactions)

    features.loc[
        features.index[-1],
        "transaction_id",
    ] = "T1"

    with pytest.raises(LeakageValidationError):
        assert_feature_dataset_is_leakage_safe(
            original=transactions,
            features=features,
        )


def test_target_feature_injection_is_detected():
    transactions = make_transactions()
    features = build_feature_dataset(transactions)

    model_features = [
        "log_amount",
        "is_fraud",
    ]

    with pytest.raises(LeakageValidationError):
        assert_feature_dataset_is_leakage_safe(
            original=transactions,
            features=features,
            model_features=model_features,
        )


def test_unknown_feature_is_detected():
    transactions = make_transactions()
    features = build_feature_dataset(transactions)

    features["future_fraud_score"] = 1.0

    with pytest.raises(LeakageValidationError):
        assert_feature_dataset_is_leakage_safe(
            original=transactions,
            features=features,
        )