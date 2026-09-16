from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
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
    "merchant_id",
    "device_id",
    "ip_id",
}

RAW_TRANSACTION_COLUMNS = {
    "timestamp",
    "amount",
    "currency",
    "payment_method",
    "transaction_type",
}

ENGINEERED_FEATURE_COLUMNS = {
    # Phase 4.1 — Transaction + Temporal
    "log_amount",
    "is_card_payment",
    "is_bank_transfer",
    "is_wallet",
    "is_online",
    "is_purchase",
    "is_transfer",
    "is_withdrawal",
    "is_payment",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "is_weekend",
    "is_night",

    # Phase 4.2 — Customer Historical
    "customer_txn_count_before",
    "customer_amount_sum_before",
    "customer_amount_mean_before",
    "customer_amount_median_before",
    "customer_amount_max_before",
    "customer_unique_merchants_before",
    "customer_unique_devices_before",
    "customer_unique_ips_before",

    # Phase 4.3 — Velocity
    "customer_txn_count_1m",
    "customer_txn_count_5m",
    "customer_txn_count_15m",
    "customer_amount_sum_1m",
    "customer_amount_sum_5m",
    "customer_amount_sum_15m",
    "customer_unique_merchants_5m",
    "customer_unique_devices_5m",
    "customer_unique_ips_5m",

    # Phase 4.4 — Entity Historical
    "account_txn_count_before",
    "card_txn_count_before",
    "merchant_txn_count_before",
    "device_txn_count_before",
    "ip_txn_count_before",
    "account_amount_sum_before",
    "card_amount_sum_before",
    "merchant_amount_sum_before",
    "device_amount_sum_before",
    "ip_amount_sum_before",

    # Phase 4.5 — Novelty
    "is_new_device",
    "is_new_ip",
    "is_new_merchant",
    "is_new_card",

    # Phase 4.6 — Behavioral Deviation
    "amount_vs_customer_mean",
    "amount_vs_customer_median",
    "amount_vs_customer_max",
    "amount_deviation_from_customer_mean",
    "is_new_payment_method",
    "is_new_transaction_type",
}


@dataclass(frozen=True)
class LeakageFinding:
    rule_id: str
    severity: str
    passed: bool
    message: str


class LeakageValidationError(ValueError):
    """Raised when the engineered feature dataset violates leakage rules."""


def _as_list(columns: Iterable[str]) -> list[str]:
    return list(columns)


def validate_no_target_features(
    columns: Iterable[str],
) -> LeakageFinding:
    columns = set(columns)
    leaked_targets = columns.intersection(TARGET_COLUMNS)

    if leaked_targets:
        return LeakageFinding(
            rule_id="LV001",
            severity="CRITICAL",
            passed=False,
            message=(
                "Target columns must not be used as model features: "
                f"{sorted(leaked_targets)}"
            ),
        )

    return LeakageFinding(
        rule_id="LV001",
        severity="CRITICAL",
        passed=True,
        message="No target columns are present in the model feature set.",
    )


def validate_no_identifier_features(
    columns: Iterable[str],
) -> LeakageFinding:
    columns = set(columns)
    leaked_identifiers = columns.intersection(IDENTIFIER_COLUMNS)

    if leaked_identifiers:
        return LeakageFinding(
            rule_id="LV002",
            severity="HIGH",
            passed=False,
            message=(
                "Identifier columns must not be used as model features: "
                f"{sorted(leaked_identifiers)}"
            ),
        )

    return LeakageFinding(
        rule_id="LV002",
        severity="HIGH",
        passed=True,
        message="No raw identifiers are present in the model feature set.",
    )


def validate_feature_names(
    columns: Iterable[str],
) -> LeakageFinding:
    columns = set(columns)

    allowed = (
        ENGINEERED_FEATURE_COLUMNS
        | RAW_TRANSACTION_COLUMNS
    )

    unexpected = columns - allowed - TARGET_COLUMNS - IDENTIFIER_COLUMNS

    if unexpected:
        return LeakageFinding(
            rule_id="LV003",
            severity="HIGH",
            passed=False,
            message=(
                "Unexpected feature columns detected: "
                f"{sorted(unexpected)}"
            ),
        )

    return LeakageFinding(
        rule_id="LV003",
        severity="HIGH",
        passed=True,
        message="All feature columns belong to the approved feature contract.",
    )


def validate_no_duplicate_transactions(
    df: pd.DataFrame,
) -> LeakageFinding:
    if "transaction_id" not in df.columns:
        return LeakageFinding(
            rule_id="LV004",
            severity="HIGH",
            passed=False,
            message="transaction_id is required for duplicate validation.",
        )

    duplicates = int(df["transaction_id"].duplicated().sum())

    if duplicates:
        return LeakageFinding(
            rule_id="LV004",
            severity="HIGH",
            passed=False,
            message=f"Found {duplicates} duplicate transaction IDs.",
        )

    return LeakageFinding(
        rule_id="LV004",
        severity="HIGH",
        passed=True,
        message="Transaction IDs are unique.",
    )


def validate_row_count(
    original: pd.DataFrame,
    features: pd.DataFrame,
) -> LeakageFinding:
    if len(original) != len(features):
        return LeakageFinding(
            rule_id="LV005",
            severity="HIGH",
            passed=False,
            message=(
                f"Row count changed from {len(original)} "
                f"to {len(features)}."
            ),
        )

    return LeakageFinding(
        rule_id="LV005",
        severity="HIGH",
        passed=True,
        message="Feature engineering preserved row count.",
    )


def validate_historical_feature_names(
    columns: Iterable[str],
) -> LeakageFinding:
    columns = set(columns)

    historical_features = {
        column
        for column in columns
        if (
            "_before" in column
            or "_1m" in column
            or "_5m" in column
            or "_15m" in column
        )
    }

    unexpected = historical_features - ENGINEERED_FEATURE_COLUMNS

    if unexpected:
        return LeakageFinding(
            rule_id="LV006",
            severity="CRITICAL",
            passed=False,
            message=(
                "Unrecognized historical/velocity feature names detected: "
                f"{sorted(unexpected)}"
            ),
        )

    return LeakageFinding(
        rule_id="LV006",
        severity="CRITICAL",
        passed=True,
        message="Historical and velocity features use approved names.",
    )


def validate_historical_features_non_negative(
    df: pd.DataFrame,
) -> LeakageFinding:
    historical_columns = [
        column
        for column in ENGINEERED_FEATURE_COLUMNS
        if (
            column in df.columns
            and (
                "_before" in column
                or "_1m" in column
                or "_5m" in column
                or "_15m" in column
            )
        )
    ]

    if not historical_columns:
        return LeakageFinding(
            rule_id="LV007",
            severity="MEDIUM",
            passed=True,
            message="No historical/velocity columns require numeric validation.",
        )

    numeric = df[historical_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    invalid = (numeric < 0).any().any()

    if invalid:
        return LeakageFinding(
            rule_id="LV007",
            severity="MEDIUM",
            passed=False,
            message="Historical/velocity features contain negative values.",
        )

    return LeakageFinding(
        rule_id="LV007",
        severity="MEDIUM",
        passed=True,
        message="Historical/velocity features are non-negative.",
    )


def validate_same_timestamp_history(
    df: pd.DataFrame,
) -> LeakageFinding:
    required = {
        "timestamp",
        "customer_id",
        "amount",
        "customer_txn_count_before",
        "customer_amount_sum_before",
    }

    missing = required - set(df.columns)

    if missing:
        return LeakageFinding(
            rule_id="LV008",
            severity="CRITICAL",
            passed=False,
            message=(
                "Cannot validate same-timestamp history. "
                f"Missing columns: {sorted(missing)}"
            ),
        )

    work = df[
        [
            "timestamp",
            "customer_id",
            "amount",
            "customer_txn_count_before",
            "customer_amount_sum_before",
        ]
    ].copy()

    # Build one row per customer/timestamp block.
    #
    # This is critical: transactions sharing the same customer and
    # timestamp must be treated as one temporal event. We cannot use
    # a row-level cumulative sum because that would make the second
    # transaction in the same timestamp block see the first one.

    blocks = (
        work.groupby(
            ["customer_id", "timestamp"],
            sort=False,
            dropna=False,
        )
        .agg(
            timestamp_txn_count=("amount", "size"),
            timestamp_amount_sum=("amount", "sum"),
        )
        .reset_index()
    )

    # Calculate cumulative history at the BLOCK level.
    blocks = blocks.sort_values(
        ["customer_id", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)

    blocks["expected_prior_count"] = (
        blocks.groupby(
            "customer_id",
            sort=False,
        )["timestamp_txn_count"]
        .cumsum()
        - blocks["timestamp_txn_count"]
    )

    blocks["expected_prior_sum"] = (
        blocks.groupby(
            "customer_id",
            sort=False,
        )["timestamp_amount_sum"]
        .cumsum()
        - blocks["timestamp_amount_sum"]
    )

    # Join the block-level expectations back to every transaction
    # belonging to that customer/timestamp.
    work = work.merge(
        blocks[
            [
                "customer_id",
                "timestamp",
                "expected_prior_count",
                "expected_prior_sum",
            ]
        ],
        on=["customer_id", "timestamp"],
        how="left",
        validate="many_to_one",
    )

    count_ok = (
        work["customer_txn_count_before"].astype(float)
        == work["expected_prior_count"].astype(float)
    )

    sum_ok = np.isclose(
        work["customer_amount_sum_before"].astype(float),
        work["expected_prior_sum"].astype(float),
    )

    if not bool(count_ok.all() and sum_ok.all()):
        return LeakageFinding(
            rule_id="LV008",
            severity="CRITICAL",
            passed=False,
            message=(
                "Historical customer features include information "
                "from the current timestamp."
            ),
        )

    return LeakageFinding(
        rule_id="LV008",
        severity="CRITICAL",
        passed=True,
        message=(
            "Customer historical count/sum features exclude "
            "same-timestamp transactions."
        ),
    )


def validate_feature_dataset(
    original: pd.DataFrame,
    features: pd.DataFrame,
    model_features: Iterable[str] | None = None,
) -> list[LeakageFinding]:
    """
    Run the Phase 4.8 leakage validation suite.

    Parameters
    ----------
    original:
        Raw transaction dataframe before feature engineering.

    features:
        Output from build_feature_dataset().

    model_features:
        Optional explicit list of columns intended for model training.
        If omitted, all non-target/non-identifier columns are inspected.
    """

    if model_features is None:
        model_features = [
            column
            for column in features.columns
            if column not in TARGET_COLUMNS
            and column not in IDENTIFIER_COLUMNS
        ]

    model_features = _as_list(model_features)

    findings = [
        validate_no_target_features(model_features),
        validate_no_identifier_features(model_features),
        validate_feature_names(features.columns),
        validate_no_duplicate_transactions(features),
        validate_row_count(original, features),
        validate_historical_feature_names(features.columns),
        validate_historical_features_non_negative(features),
        validate_same_timestamp_history(features),
    ]

    return findings


def assert_feature_dataset_is_leakage_safe(
    original: pd.DataFrame,
    features: pd.DataFrame,
    model_features: Iterable[str] | None = None,
) -> None:
    findings = validate_feature_dataset(
        original=original,
        features=features,
        model_features=model_features,
    )

    failures = [
        finding
        for finding in findings
        if not finding.passed
    ]

    if failures:
        details = "\n".join(
            (
                f"[{finding.rule_id}] "
                f"{finding.severity}: "
                f"{finding.message}"
            )
            for finding in failures
        )

        raise LeakageValidationError(
            "Feature dataset failed leakage validation:\n"
            + details
        )


__all__ = [
    "TARGET_COLUMNS",
    "IDENTIFIER_COLUMNS",
    "RAW_TRANSACTION_COLUMNS",
    "ENGINEERED_FEATURE_COLUMNS",
    "LeakageFinding",
    "LeakageValidationError",
    "validate_feature_dataset",
    "assert_feature_dataset_is_leakage_safe",
]