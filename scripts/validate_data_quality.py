
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
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

EXPECTED_ROWS = 100_000
EXPECTED_COLUMNS = 67
EXPECTED_FEATURE_COUNT = 53
EXPECTED_FRAUD_RATE = 0.01
FRAUD_RATE_TOLERANCE = 0.001

REQUIRED_COLUMNS = {
    "transaction_id",
    "timestamp",
    "is_fraud",
    "fraud_scenario",
}

CANONICAL_FEATURES = {
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
    "customer_txn_count_before",
    "customer_amount_sum_before",
    "customer_amount_mean_before",
    "customer_amount_median_before",
    "customer_amount_max_before",
    "customer_unique_merchants_before",
    "customer_unique_devices_before",
    "customer_unique_ips_before",
    "customer_txn_count_1m",
    "customer_amount_sum_1m",
    "customer_txn_count_5m",
    "customer_amount_sum_5m",
    "customer_txn_count_15m",
    "customer_amount_sum_15m",
    "customer_unique_merchants_5m",
    "customer_unique_devices_5m",
    "customer_unique_ips_5m",
    "account_txn_count_before",
    "account_amount_sum_before",
    "card_txn_count_before",
    "card_amount_sum_before",
    "merchant_txn_count_before",
    "merchant_amount_sum_before",
    "device_txn_count_before",
    "device_amount_sum_before",
    "ip_txn_count_before",
    "ip_amount_sum_before",
    "is_new_device",
    "is_new_ip",
    "is_new_merchant",
    "is_new_card",
    "amount_vs_customer_mean",
    "amount_vs_customer_median",
    "amount_vs_customer_max",
    "amount_deviation_from_customer_mean",
    "is_new_payment_method",
    "is_new_transaction_type",
}


def validate_dataset(
    dataset_path: Path = DATASET_PATH,
    metadata_path: Path = METADATA_PATH,
) -> list[str]:
    """Return quality violations; an empty list means the gate passed."""
    errors: list[str] = []

    if not dataset_path.is_file():
        return [f"Feature dataset not found: {dataset_path}"]

    if not metadata_path.is_file():
        return [f"Feature metadata not found: {metadata_path}"]

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Unable to read feature metadata: {exc}"]

    try:
        dataset = pd.read_parquet(dataset_path)
    except Exception as exc:
        return [f"Unable to read feature dataset: {exc}"]

    if len(dataset) != EXPECTED_ROWS:
        errors.append(
            f"Expected {EXPECTED_ROWS} rows; found {len(dataset)}."
        )

    if len(dataset.columns) != EXPECTED_COLUMNS:
        errors.append(
            f"Expected {EXPECTED_COLUMNS} columns; "
            f"found {len(dataset.columns)}."
        )

    missing_columns = REQUIRED_COLUMNS - set(dataset.columns)
    if missing_columns:
        errors.append(
            f"Missing required columns: {sorted(missing_columns)}."
        )

    if dataset.columns.duplicated().any():
        errors.append("Dataset contains duplicate column names.")

    if "transaction_id" in dataset.columns:
        if dataset["transaction_id"].isna().any():
            errors.append("transaction_id contains missing values.")
        if dataset["transaction_id"].duplicated().any():
            errors.append("transaction_id contains duplicates.")

    if "timestamp" in dataset.columns:
        if dataset["timestamp"].isna().any():
            errors.append("timestamp contains missing values.")
        if not isinstance(dataset["timestamp"].dtype, pd.DatetimeTZDtype):
            errors.append("timestamp must be timezone-aware.")

    if "is_fraud" in dataset.columns:
        target = dataset["is_fraud"]
        if target.isna().any():
            errors.append("is_fraud contains missing values.")
        elif not target.isin([0, 1]).all():
            errors.append("is_fraud must contain only 0 and 1.")
        else:
            fraud_rate = float(target.mean())
            if abs(fraud_rate - EXPECTED_FRAUD_RATE) > FRAUD_RATE_TOLERANCE:
                errors.append(
                    f"Expected fraud rate near {EXPECTED_FRAUD_RATE:.3%}; "
                    f"found {fraud_rate:.3%}."
                )

    feature_columns = metadata.get("model_feature_columns")
    if not isinstance(feature_columns, list):
        errors.append("Metadata model_feature_columns must be a list.")
    else:
        if len(feature_columns) != EXPECTED_FEATURE_COUNT:
            errors.append(
                f"Expected {EXPECTED_FEATURE_COUNT} model features; "
                f"metadata declares {len(feature_columns)}."
            )
        if set(feature_columns) != CANONICAL_FEATURES:
            errors.append(
                "Metadata model features do not match the canonical "
                "53-feature contract."
            )
        missing_features = set(feature_columns) - set(dataset.columns)
        if missing_features:
            errors.append(
                f"Model features missing from dataset: "
                f"{sorted(missing_features)}."
            )
        present_features = [
            feature for feature in feature_columns
            if feature in dataset.columns
        ]
        if present_features and dataset[present_features].isna().any().any():
            errors.append("Model features contain missing values.")
        if present_features and not all(
            pd.api.types.is_numeric_dtype(dataset[column].dtype)
            for column in present_features
        ):
            errors.append("Model features must all be numeric.")

    metadata_checks = {
        "row_count": len(dataset),
        "column_count": len(dataset.columns),
        "model_feature_count": EXPECTED_FEATURE_COUNT,
    }
    for key, actual in metadata_checks.items():
        if metadata.get(key) != actual:
            errors.append(
                f"Metadata {key} mismatch: expected {actual}, "
                f"found {metadata.get(key)!r}."
            )

    if "is_fraud" in dataset.columns:
        if metadata.get("fraud_count") != int(dataset["is_fraud"].sum()):
            errors.append("Metadata fraud_count does not match dataset.")
        if metadata.get("fraud_rate") != float(dataset["is_fraud"].mean()):
            errors.append("Metadata fraud_rate does not match dataset.")

    return errors


def main() -> int:
    errors = validate_dataset()
    if errors:
        print("DATA QUALITY GATE: FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("DATA QUALITY GATE: PASSED")
    print(f"Dataset: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {EXPECTED_ROWS:,}")
    print(f"Model features: {EXPECTED_FEATURE_COUNT}")
    print(f"Expected fraud rate: {EXPECTED_FRAUD_RATE:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
