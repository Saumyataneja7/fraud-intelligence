from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "feature_dataset.parquet"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "feature_dataset_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "feature_engineering"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "FEATURE_ENGINEERING_REPORT.md"
)


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

FEATURE_FAMILIES = {
    "Transaction + Temporal": [
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
    ],
    "Customer Historical": [
        "customer_txn_count_before",
        "customer_amount_sum_before",
        "customer_amount_mean_before",
        "customer_amount_median_before",
        "customer_amount_max_before",
        "customer_unique_merchants_before",
        "customer_unique_devices_before",
        "customer_unique_ips_before",
    ],
    "Velocity": [
        "customer_txn_count_1m",
        "customer_txn_count_5m",
        "customer_txn_count_15m",
        "customer_amount_sum_1m",
        "customer_amount_sum_5m",
        "customer_amount_sum_15m",
        "customer_unique_merchants_5m",
        "customer_unique_devices_5m",
        "customer_unique_ips_5m",
    ],
    "Entity Historical": [
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
    ],
    "Novelty": [
        "is_new_device",
        "is_new_ip",
        "is_new_merchant",
        "is_new_card",
    ],
    "Behavioral Deviation": [
        "amount_vs_customer_mean",
        "amount_vs_customer_median",
        "amount_vs_customer_max",
        "amount_deviation_from_customer_mean",
        "is_new_payment_method",
        "is_new_transaction_type",
    ],
}


IDENTIFIER_COLUMNS = [
    "transaction_id",
    "customer_id",
    "account_id",
    "card_id",
    "merchant_id",
    "device_id",
    "ip_id",
]


TARGET_COLUMNS = [
    "is_fraud",
    "fraud_scenario",
]


# ---------------------------------------------------------------------------
# Feature descriptions
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS = {
    # Transaction + Temporal
    "log_amount": "Log-transformed transaction amount.",
    "is_card_payment": "Whether the payment method is card.",
    "is_bank_transfer": "Whether the payment method is bank transfer.",
    "is_wallet": "Whether the payment method is wallet.",
    "is_online": "Whether the payment method is online.",
    "is_purchase": "Whether the transaction type is purchase.",
    "is_transfer": "Whether the transaction type is transfer.",
    "is_withdrawal": "Whether the transaction type is withdrawal.",
    "is_payment": "Whether the transaction type is payment.",
    "hour": "Hour extracted from the transaction timestamp.",
    "day_of_week": "Day of week extracted from the transaction timestamp.",
    "day_of_month": "Day of month extracted from the transaction timestamp.",
    "month": "Calendar month extracted from the transaction timestamp.",
    "quarter": "Calendar quarter extracted from the transaction timestamp.",
    "is_weekend": "Whether the transaction occurred on Saturday or Sunday.",
    "is_night": "Whether the transaction occurred during the configured night period.",

    # Customer Historical
    "customer_txn_count_before": (
        "Number of earlier transactions by the customer."
    ),
    "customer_amount_sum_before": (
        "Total amount of earlier customer transactions."
    ),
    "customer_amount_mean_before": (
        "Mean amount of earlier customer transactions."
    ),
    "customer_amount_median_before": (
        "Median amount of earlier customer transactions."
    ),
    "customer_amount_max_before": (
        "Maximum amount among earlier customer transactions."
    ),
    "customer_unique_merchants_before": (
        "Unique merchants previously used by the customer."
    ),
    "customer_unique_devices_before": (
        "Unique devices previously used by the customer."
    ),
    "customer_unique_ips_before": (
        "Unique IP addresses previously used by the customer."
    ),

    # Velocity
    "customer_txn_count_1m": (
        "Customer transaction count in the previous 1 minute."
    ),
    "customer_txn_count_5m": (
        "Customer transaction count in the previous 5 minutes."
    ),
    "customer_txn_count_15m": (
        "Customer transaction count in the previous 15 minutes."
    ),
    "customer_amount_sum_1m": (
        "Total customer transaction amount in the previous 1 minute."
    ),
    "customer_amount_sum_5m": (
        "Total customer transaction amount in the previous 5 minutes."
    ),
    "customer_amount_sum_15m": (
        "Total customer transaction amount in the previous 15 minutes."
    ),
    "customer_unique_merchants_5m": (
        "Unique merchants used by the customer in the previous 5 minutes."
    ),
    "customer_unique_devices_5m": (
        "Unique devices used by the customer in the previous 5 minutes."
    ),
    "customer_unique_ips_5m": (
        "Unique IP addresses used by the customer in the previous 5 minutes."
    ),

    # Entity Historical
    "account_txn_count_before": (
        "Number of earlier transactions associated with the account."
    ),
    "card_txn_count_before": (
        "Number of earlier transactions associated with the card."
    ),
    "merchant_txn_count_before": (
        "Number of earlier transactions associated with the merchant."
    ),
    "device_txn_count_before": (
        "Number of earlier transactions associated with the device."
    ),
    "ip_txn_count_before": (
        "Number of earlier transactions associated with the IP address."
    ),
    "account_amount_sum_before": (
        "Total amount of earlier transactions associated with the account."
    ),
    "card_amount_sum_before": (
        "Total amount of earlier transactions associated with the card."
    ),
    "merchant_amount_sum_before": (
        "Total amount of earlier transactions associated with the merchant."
    ),
    "device_amount_sum_before": (
        "Total amount of earlier transactions associated with the device."
    ),
    "ip_amount_sum_before": (
        "Total amount of earlier transactions associated with the IP address."
    ),

    # Novelty
    "is_new_device": (
        "Whether the device is new for the customer."
    ),
    "is_new_ip": (
        "Whether the IP address is new for the customer."
    ),
    "is_new_merchant": (
        "Whether the merchant is new for the customer."
    ),
    "is_new_card": (
        "Whether the card is new for the customer."
    ),

    # Behavioral Deviation
    "amount_vs_customer_mean": (
        "Current amount relative to the historical customer mean."
    ),
    "amount_vs_customer_median": (
        "Current amount relative to the historical customer median."
    ),
    "amount_vs_customer_max": (
        "Current amount relative to the historical customer maximum."
    ),
    "amount_deviation_from_customer_mean": (
        "Deviation of the current amount from the historical customer mean."
    ),
    "is_new_payment_method": (
        "Whether the payment method is new for the customer."
    ),
    "is_new_transaction_type": (
        "Whether the transaction type is new for the customer."
    ),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_metadata() -> dict:
    """Load feature dataset metadata."""
    return json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )


def _expected_feature_columns() -> list[str]:
    """Return the complete ordered list of engineered features."""
    return [
        feature
        for features in FEATURE_FAMILIES.values()
        for feature in features
    ]


def _validate_dataset(
    df: pd.DataFrame,
    metadata: dict,
) -> None:
    """Validate that the materialized dataset matches the report contract."""

    expected_rows = metadata["row_count"]

    if len(df) != expected_rows:
        raise ValueError(
            "Dataset row count mismatch: "
            f"{len(df)} != {expected_rows}"
        )

    missing_identifiers = (
        set(IDENTIFIER_COLUMNS) - set(df.columns)
    )

    if missing_identifiers:
        raise ValueError(
            "Missing identifier columns: "
            f"{sorted(missing_identifiers)}"
        )

    missing_targets = (
        set(TARGET_COLUMNS) - set(df.columns)
    )

    if missing_targets:
        raise ValueError(
            "Missing target columns: "
            f"{sorted(missing_targets)}"
        )

    expected_features = set(
        _expected_feature_columns()
    )

    missing_features = (
        expected_features - set(df.columns)
    )

    if missing_features:
        raise ValueError(
            "Missing engineered features: "
            f"{sorted(missing_features)}"
        )

    metadata_features = metadata.get(
        "model_feature_columns",
        [],
    )

    if len(metadata_features) != len(
        expected_features
    ):
        raise ValueError(
            "Model feature count mismatch: "
            f"{len(metadata_features)} != "
            f"{len(expected_features)}"
        )

    if set(metadata_features) != expected_features:
        raise ValueError(
            "Metadata model features do not match "
            "the approved feature family definitions."
        )


def _feature_table(df: pd.DataFrame) -> str:
    """Build the complete feature dictionary markdown table."""

    del df  # Kept in signature for compatibility with the original helper.

    lines = [
        "| Family | Feature | Description |",
        "|---|---|---|",
    ]

    for family, features in FEATURE_FAMILIES.items():
        for feature in features:
            description = FEATURE_DESCRIPTIONS.get(
                feature,
                "",
            )

            lines.append(
                f"| {family} | `{feature}` | "
                f"{description} |"
            )

    return "\n".join(lines)


def _family_summary() -> str:
    """Build the feature-family summary markdown table."""

    lines = [
        "| Feature Family | Feature Count |",
        "|---|---:|",
    ]

    for family, features in FEATURE_FAMILIES.items():
        lines.append(
            f"| {family} | {len(features)} |"
        )

    total = sum(
        len(features)
        for features in FEATURE_FAMILIES.values()
    )

    lines.append(
        f"| **Total** | **{total}** |"
    )

    return "\n".join(lines)


def _bullet_list(columns: list[str]) -> str:
    """Convert a list of column names into markdown bullets."""
    return "\n".join(
        f"- `{column}`"
        for column in columns
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report() -> Path:
    """Generate the Phase 4 feature engineering report."""

    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: "
            f"{FEATURE_DATASET_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature metadata not found: "
            f"{METADATA_PATH}"
        )

    df = pd.read_parquet(
        FEATURE_DATASET_PATH
    )

    metadata = _load_metadata()

    _validate_dataset(
        df,
        metadata,
    )

    # -----------------------------------------------------------------------
    # Dataset statistics
    # -----------------------------------------------------------------------

    fraud_count = int(
        df["is_fraud"].sum()
    )

    fraud_rate = (
        fraud_count / len(df)
        if len(df)
        else 0.0
    )

    timestamp_min = df["timestamp"].min()
    timestamp_max = df["timestamp"].max()

    model_features = metadata[
        "model_feature_columns"
    ]

    feature_count = len(
        _expected_feature_columns()
    )

    if len(model_features) != feature_count:
        raise ValueError(
            "Unexpected model feature count: "
            f"{len(model_features)}"
        )

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------

    report = f"""# Phase 4 — Feature Engineering Report

**Project:** Fraud Intelligence — Graph ML + Explainable AI  
**Phase:** 4 — Feature Engineering  
**Status:** Complete and frozen  
**Generated:** {datetime.now(timezone.utc).isoformat()}

---

## 1. Executive Summary

Phase 4 transforms the validated synthetic transaction dataset into a
production-style feature dataset suitable for downstream fraud modeling.

The feature engineering pipeline contains six feature families:

1. Transaction + Temporal
2. Customer Historical
3. Velocity
4. Entity Historical
5. Novelty
6. Behavioral Deviation

The final pipeline produces **{feature_count} engineered model features**.

The resulting materialized dataset contains:

- **{len(df):,} transactions**
- **{len(df.columns)} total columns**
- **{len(model_features)} model features**
- **{fraud_count:,} fraud transactions**
- **{fraud_rate:.2%} fraud rate**

All temporal features are designed to prevent future and same-timestamp
information from entering historical baselines.

---

## 2. Phase 4 Architecture

```text
Raw Transactions
       |
       v
4.1 Transaction + Temporal
       |
       v
4.2 Customer Historical
       |
       v
4.3 Velocity
       |
       v
4.4 Entity Historical
       |
       v
4.5 Novelty
       |
       v
4.6 Behavioral Deviation
       |
       v
4.7 Integrated Feature Pipeline
       |
       v
4.8 Leakage Validation
       |
       v
4.9 Materialized Feature Dataset
       |
       v
4.10 Feature Engineering Report & Freeze
'''

## 3. Dataset Profile

The final materialized feature dataset was generated from the project's
versioned synthetic transaction dataset.

Property	                         Value
Transactions	                  {len(df):,}
Total columns	                  {len(df.columns)}
Engineered model features	      {len(model_features)}
Fraud transactions	              {fraud_count:,}
Fraud rate	                      {fraud_rate:.2%}
Timestamp minimum	              {timestamp_min}
Timestamp maximum	              {timestamp_max}
Dataset format	                  Parquet

The dataset retains identifiers and target columns for investigation,
validation, and downstream analysis, while these columns are excluded from
the model feature boundary.

## 4. Feature Family Summary

{_family_summary()}

The six families contain complementary signals:

-Transaction + Temporal: transaction amount transformations,
categorical indicators, and calendar/time-of-day information.
-Customer Historical: point-in-time customer behavior accumulated from
strictly earlier transactions.
-Velocity: short-term transaction and amount activity over 1-minute,
5-minute, and 15-minute windows.
-Entity Historical: historical activity associated with accounts, cards,
merchants, devices, and IP addresses.
-Novelty: whether an entity is being observed for the first time by a
customer.
-Behavioral Deviation: deviation from the customer's historical
behavioral baseline.

## 5. Complete Feature Dictionary

{_feature_table(df)}

All {feature_count} engineered features are part of the approved model
feature contract.

The original transaction identifiers and target columns are intentionally
not included in this feature dictionary because they are not model features.

## 6. Temporal Semantics

Temporal semantics are a central design constraint of the feature pipeline.

Historical features use information from transactions that occurred
strictly before the current transaction timestamp.

For a current transaction at timestamp T:

-Transactions with timestamp < T may contribute to historical features.
-Transactions with timestamp == T cannot contribute to one another's
historical features.
-Transactions with timestamp > T cannot contribute.
-The current transaction itself cannot contribute to its own historical
features.

### Same-Timestamp Treatment

Transactions sharing the same customer and timestamp are treated as one
temporal block.

This prevents arbitrary dataframe row ordering from creating artificial
historical information.

For example, if two transactions belonging to the same customer occur at
exactly 12:00, neither transaction is allowed to use the other transaction
when calculating the customer's historical baseline.

Velocity windows follow the same principle: the current transaction and
same-timestamp transactions are excluded.

## 7. Cold-Start Behavior

The first transaction for a customer or entity has no valid historical
observations.

Therefore:

-Historical counts start at 0.
-Historical sums start at 0.
-Historical means start at 0.
-Historical medians start at 0.
-Historical maxima start at 0.
-Historical uniqueness counts start at 0.
-Velocity counts and sums start at 0 when no prior activity exists.
-Novelty features identify first observations.
-Behavioral deviation features use the defined zero-baseline behavior when
no historical customer baseline exists.

This provides deterministic behavior for previously unseen customers and
entities.

## 8. Model Feature Boundary

The final dataset contains three conceptual column groups.

### Model Features

The approved model feature set contains {len(model_features)} engineered
features.

These are the columns supplied to downstream machine-learning models.

### Identifiers

The following columns are retained for investigation and entity tracing:

{_bullet_list(IDENTIFIER_COLUMNS)}

Identifiers are explicitly excluded from the model feature matrix.

### Targets

The following columns are retained as prediction targets and investigation
metadata:

{_bullet_list(TARGET_COLUMNS)}

Target columns are never supplied to the model as input features.

## 9. Leakage Controls

The Phase 4 implementation contains multiple controls designed to prevent
temporal and target leakage.

### Same-timestamp leakage validation

Transactions occurring at the same timestamp are treated as a temporal
block for historical calculations.

A transaction cannot use another transaction from the same timestamp to
construct its historical customer or entity baseline.

### Future-transaction invariance

Feature values for a transaction are invariant to the presence of
transactions that occur after its timestamp.

Future transactions cannot influence historical customer features,
entity-level historical features, novelty features, behavioral baselines,
or velocity windows.

This ensures that adding or removing future observations does not alter
features for transactions that have already occurred.

### Historical Feature Controls

Historical features use only strictly earlier observations.

The following are protected:

-Customer historical counts and amounts.
-Entity historical counts and amounts.
-Historical customer means, medians, and maxima.
-Historical uniqueness features.
-Customer-specific novelty features.
-Customer behavioral baselines.

### Velocity Controls

Velocity windows exclude:

-The current transaction.
-Same-timestamp transactions.
-Future transactions.

### Schema Controls

The feature dataset is validated to ensure:

-Target columns excluded from model features.
-Identifier columns excluded from model features.
-Approved engineered feature names are present.
-Duplicate transaction IDs are not introduced.
-Row count is preserved.
-Historical and velocity features satisfy non-negativity constraints.
-Same-timestamp historical calculations satisfy the expected temporal
semantics.

These controls are implemented in the Phase 4 leakage validation layer.

## 10. Reproducibility

The feature dataset is generated deterministically from the versioned
synthetic transaction dataset.

The project uses:

-Dataset version: 1.0.0
-Random seed: 42
-Fixed feature definitions.
-Deterministic temporal ordering.
-Version-controlled feature engineering code.
-Parquet materialization.
-Feature metadata stored alongside the dataset.

The materialization process can therefore be rerun from the same source
dataset and configuration.

The generated report timestamp is intentionally dynamic; reproducibility
refers to the underlying dataset and feature artifacts rather than the
timestamp printed in this document.

## 11. Validation

Phase 4 validation covers both implementation behavior and the final
materialized artifact.

Validation includes:

-Unit tests for each feature family.
-Integration tests for the complete feature pipeline.
-Leakage validation tests.
-Model feature contract validation.
-Materialization tests.
-Reproducibility tests.
-Feature report tests.

The final feature dataset contains:

-{len(df):,} rows
-{len(df.columns)} total columns
-{len(model_features)} model features
-{fraud_count:,} fraud transactions

The Phase 4 test suite validates that the engineered features preserve
transaction ordering, row counts, input immutability, feature boundaries,
and temporal leakage constraints.

## 12. Known Design Limitations

The following limitations are intentionally documented:

-The dataset is synthetic and does not represent real customer behavior.
-Fraud scenarios are generated according to the project's synthetic
generation assumptions.
-Historical features are currently calculated from transaction-level
data and are not backed by a streaming feature store.
-Feature computation is designed for correctness and reproducibility,
not yet for distributed production-scale execution.
-Entity relationships are currently represented in the relational
transaction dataset; graph construction is a subsequent project phase.
-No production model performance claims are made from the engineered
features alone.
-Label-derived fraud-ring information remains an investigation/EDA
concept and is not used as a predictive feature.

## 13. Phase 4 Deliverables

Phase 4 produces the following artifacts:

-Transaction and temporal feature implementation.
-Customer historical feature implementation.
-Velocity feature implementation.
-Entity historical feature implementation.
-Novelty feature implementation.
-Behavioral deviation feature implementation.
-Integrated feature pipeline.
-Leakage validation layer.
-Materialized feature dataset.
-Feature dataset metadata.
-Feature engineering report.
-Unit and integration tests covering the feature pipeline.

The primary materialized artifacts are:

data/processed/features/feature_dataset.parquet
data/processed/features/feature_dataset_metadata.json
reports/feature_engineering/FEATURE_ENGINEERING_REPORT.md

## 14. Final Phase 4 Conclusion

Phase 4 is complete and frozen.

The project now has a validated, leakage-controlled, materialized feature
dataset containing {len(model_features)} engineered model features.

The feature pipeline establishes the point-in-time behavioral foundation
required for downstream fraud modeling and graph-based fraud intelligence.

No predictive model is trained as part of Phase 4. Model development begins
only in the subsequent phase.
"""
    
# -----------------------------------------------------------------------
# Write report
# -----------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    return REPORT_PATH


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    path = generate_report()

    print(
        f"Feature engineering report created: {path}"
    )