
import json

import pandas as pd

from scripts.validate_data_quality import (
    CANONICAL_FEATURES,
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    EXPECTED_FRAUD_RATE,
    validate_dataset,
)


def make_valid_dataset(tmp_path):
    """Create a small fixture with a matching, test-specific contract."""
    dataset_path = tmp_path / "feature_dataset.parquet"
    metadata_path = tmp_path / "feature_dataset_metadata.json"

    features = sorted(CANONICAL_FEATURES)
    row_count = EXPECTED_ROWS

    data = {
        feature: [0.0] * row_count
        for feature in features
    }
    data.update({
        "transaction_id": [f"TXN_{i}" for i in range(row_count)],
        "timestamp": pd.date_range(
            "2025-01-01",
            periods=row_count,
            freq="min",
            tz="UTC",
        ),
        "is_fraud": [1] * int(row_count * EXPECTED_FRAUD_RATE)
        + [0] * (row_count - int(row_count * EXPECTED_FRAUD_RATE)),
        "fraud_scenario": [""] * row_count,
    })

    dataset = pd.DataFrame(data)

    # Add the remaining investigation columns expected by the artifact contract.
    while len(dataset.columns) < EXPECTED_COLUMNS:
        dataset[f"extra_{len(dataset.columns)}"] = 0

    dataset.to_parquet(dataset_path, index=False)

    metadata = {
        "row_count": len(dataset),
        "column_count": len(dataset.columns),
        "model_feature_count": len(features),
        "model_feature_columns": features,
        "fraud_count": int(dataset["is_fraud"].sum()),
        "fraud_rate": float(dataset["is_fraud"].mean()),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    return dataset_path, metadata_path, dataset


def test_valid_dataset_passes(tmp_path):
    dataset_path, metadata_path, _ = make_valid_dataset(tmp_path)

    assert validate_dataset(dataset_path, metadata_path) == []


def test_missing_transaction_id_fails(tmp_path):
    dataset_path, metadata_path, dataset = make_valid_dataset(tmp_path)
    dataset["transaction_id"] = None
    dataset.to_parquet(dataset_path, index=False)

    errors = validate_dataset(dataset_path, metadata_path)

    assert any("transaction_id contains missing values" in error for error in errors)


def test_duplicate_transaction_id_fails(tmp_path):
    dataset_path, metadata_path, dataset = make_valid_dataset(tmp_path)
    dataset.loc[1, "transaction_id"] = dataset.loc[0, "transaction_id"]
    dataset.to_parquet(dataset_path, index=False)

    errors = validate_dataset(dataset_path, metadata_path)

    assert any("transaction_id contains duplicates" in error for error in errors)


def test_invalid_target_fails(tmp_path):
    dataset_path, metadata_path, dataset = make_valid_dataset(tmp_path)
    dataset.loc[0, "is_fraud"] = 2
    dataset.to_parquet(dataset_path, index=False)

    errors = validate_dataset(dataset_path, metadata_path)

    assert any("is_fraud must contain only 0 and 1" in error for error in errors)


def test_missing_model_feature_fails(tmp_path):
    dataset_path, metadata_path, dataset = make_valid_dataset(tmp_path)
    dataset.drop(columns=["log_amount"], inplace=True)
    dataset.to_parquet(dataset_path, index=False)

    errors = validate_dataset(dataset_path, metadata_path)

    assert any("Expected 67 columns" in error for error in errors)
    assert any("Model features missing from dataset" in error for error in errors)
