import pandas as pd

from fraud_intelligence.analysis.leakage_audit import (
    audit_duplicate_transactions,
    audit_feature_leakage_rules,
    audit_target_leakage,
    audit_temporal_order,
    run_leakage_audit,
)


def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [
                "TX001",
                "TX002",
                "TX003",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00:00+00:00",
                    "2025-01-01 01:00:00+00:00",
                    "2025-01-01 02:00:00+00:00",
                ],
                utc=True,
            ),
            "customer_id": [
                "C001",
                "C002",
                "C003",
            ],
            "account_id": [
                "A001",
                "A002",
                "A003",
            ],
            "amount": [
                10.0,
                20.0,
                30.0,
            ],
            "currency": [
                "USD",
                "USD",
                "USD",
            ],
            "payment_method": [
                "card",
                "wallet",
                "card",
            ],
            "transaction_type": [
                "purchase",
                "transfer",
                "payment",
            ],
            "is_fraud": [
                0,
                1,
                0,
            ],
            "fraud_scenario": [
                "LEGITIMATE",
                "ACCOUNT_TAKEOVER",
                "LEGITIMATE",
            ],
        }
    )


def test_target_leakage_identification():
    transactions = sample_transactions()

    result = audit_target_leakage(
        transactions
    )

    assert set(result["column"]) == {
        "is_fraud",
        "fraud_scenario",
    }

    assert (
        result["status"] == "UNSAFE"
    ).all()


def test_temporal_audit():
    transactions = sample_transactions()

    result = audit_temporal_order(
        transactions
    )

    assert result[
        "timestamp_is_datetime"
    ]

    assert result[
        "timestamp_is_timezone_aware"
    ]

    assert result[
        "is_monotonic"
    ]

    assert result[
        "duplicate_timestamps"
    ] == 0


def test_duplicate_audit():
    transactions = sample_transactions()

    result = audit_duplicate_transactions(
        transactions
    )

    assert result[
        "duplicate_rows"
    ] == 0

    assert result[
        "duplicate_transaction_ids"
    ] == 0


def test_feature_rules():
    result = audit_feature_leakage_rules()

    assert len(result) == 8
    assert (
        result["status"] == "ENFORCE"
    ).all()


def test_complete_leakage_audit():
    transactions = sample_transactions()

    result = run_leakage_audit(
        transactions
    )

    assert "target_leakage" in result
    assert "identifier_audit" in result
    assert "transaction_columns" in result
    assert "temporal_audit" in result
    assert "duplicate_audit" in result
    assert "feature_rules" in result