from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TARGET_COLUMNS = {
    "is_fraud",
    "fraud_scenario",
}


IDENTIFIER_COLUMNS = {
    "transaction_id",
    "customer_id",
    "account_id",
    "card_id",
    "device_id",
    "ip_id",
    "merchant_id",
}


STATIC_TRANSACTION_COLUMNS = {
    "timestamp",
    "amount",
    "currency",
    "payment_method",
    "transaction_type",
}


@dataclass(frozen=True)
class LeakageFinding:
    column: str
    category: str
    status: str
    reason: str


def audit_target_leakage(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Identify columns that directly encode the target."""

    findings = []

    for column in transactions.columns:
        if column in TARGET_COLUMNS:
            findings.append(
                LeakageFinding(
                    column=column,
                    category="direct_target",
                    status="UNSAFE",
                    reason=(
                        "Column directly contains the fraud "
                        "label or fraud scenario."
                    ),
                )
            )

    return pd.DataFrame(
        [
            {
                "column": finding.column,
                "category": finding.category,
                "status": finding.status,
                "reason": finding.reason,
            }
            for finding in findings
        ]
    )


def audit_identifier_columns(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Classify identifiers that require historical aggregation."""

    findings = []

    for column in transactions.columns:
        if column in IDENTIFIER_COLUMNS:
            findings.append(
                LeakageFinding(
                    column=column,
                    category="identifier",
                    status="CONDITIONAL",
                    reason=(
                        "Raw identifier should not be treated as a "
                        "numeric predictive feature. Historical "
                        "aggregates or relational features must be "
                        "calculated point-in-time."
                    ),
                )
            )

    return pd.DataFrame(
        [
            {
                "column": finding.column,
                "category": finding.category,
                "status": finding.status,
                "reason": finding.reason,
            }
            for finding in findings
        ]
    )


def audit_transaction_columns(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Classify transaction-level fields."""

    findings = []

    for column in transactions.columns:
        if column in TARGET_COLUMNS:
            continue

        if column in IDENTIFIER_COLUMNS:
            continue

        if column in STATIC_TRANSACTION_COLUMNS:
            findings.append(
                LeakageFinding(
                    column=column,
                    category="transaction_attribute",
                    status="SAFE",
                    reason=(
                        "Transaction-level attribute is available "
                        "at transaction time."
                    ),
                )
            )

    return pd.DataFrame(
        [
            {
                "column": finding.column,
                "category": finding.category,
                "status": finding.status,
                "reason": finding.reason,
            }
            for finding in findings
        ]
    )


def audit_temporal_order(
    transactions: pd.DataFrame,
) -> dict[str, object]:
    """Audit timestamp ordering and temporal coverage."""

    timestamps = pd.to_datetime(
        transactions["timestamp"],
        utc=True,
    )

    return {
        "timestamp_is_datetime": (
            pd.api.types.is_datetime64_any_dtype(
                timestamps
            )
        ),
        "timestamp_is_timezone_aware": (
            timestamps.dt.tz is not None
        ),
        "is_monotonic": timestamps.is_monotonic_increasing,
        "min_timestamp": timestamps.min(),
        "max_timestamp": timestamps.max(),
        "unique_timestamps": timestamps.nunique(),
        "duplicate_timestamps": int(
            timestamps.duplicated().sum()
        ),
    }


def audit_duplicate_transactions(
    transactions: pd.DataFrame,
) -> dict[str, int]:
    """Check for duplicate transactions."""

    return {
        "duplicate_rows": int(
            transactions.duplicated().sum()
        ),
        "duplicate_transaction_ids": int(
            transactions["transaction_id"]
            .duplicated()
            .sum()
        ),
    }


def audit_feature_leakage_rules() -> pd.DataFrame:
    """Document the rules that future feature engineering must follow."""

    rules = [
        {
            "rule_id": "L001",
            "rule": "Never use is_fraud as a model feature.",
            "status": "ENFORCE",
        },
        {
            "rule_id": "L002",
            "rule": "Never use fraud_scenario as a model feature.",
            "status": "ENFORCE",
        },
        {
            "rule_id": "L003",
            "rule": (
                "Historical aggregates must use only transactions "
                "strictly earlier than prediction time unless an "
                "explicit event ordering is available."
            ),
            "status": "ENFORCE",
        },
        {
            "rule_id": "L004",
            "rule": (
                "Transactions at the same timestamp or in the future "
                "must not contribute to features unless an explicit "
                "event ordering is available."
            ),
            "status": "ENFORCE",
        },
        {
            "rule_id": "L005",
            "rule": (
                "Entity fraud statistics must be calculated "
                "point-in-time using only strictly historical data."
            ),
            "status": "ENFORCE",
        },
        {
            "rule_id": "L006",
            "rule": (
                "Velocity features must only use strictly previous "
                "transactions."
            ),
            "status": "ENFORCE",
        },
        {
            "rule_id": "L007",
            "rule": (
                "Fraud-ring membership discovered using labels "
                "is EDA-only and cannot be used as a predictive feature."
            ),
            "status": "ENFORCE",
        },
        {
            "rule_id": "L008",
            "rule": (
                "Train/test splits must respect transaction time."
            ),
            "status": "ENFORCE",
        },
    ]

    return pd.DataFrame(rules)


def run_leakage_audit(
    transactions: pd.DataFrame,
) -> dict[str, object]:
    """Run the complete Phase 3.6 leakage audit."""

    return {
        "target_leakage": audit_target_leakage(
            transactions
        ),
        "identifier_audit": audit_identifier_columns(
            transactions
        ),
        "transaction_columns": audit_transaction_columns(
            transactions
        ),
        "temporal_audit": audit_temporal_order(
            transactions
        ),
        "duplicate_audit": audit_duplicate_transactions(
            transactions
        ),
        "feature_rules": audit_feature_leakage_rules(),
    }