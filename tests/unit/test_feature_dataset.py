import json

import pandas as pd
import pytest

from fraud_intelligence.features.dataset import (
    build_feature_metadata,
    build_validated_feature_dataset,
    get_model_feature_columns,
    materialize_feature_dataset,
    validate_model_feature_columns,
)
from fraud_intelligence.features.leakage_validation import (
    IDENTIFIER_COLUMNS,
    TARGET_COLUMNS,
)


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


def test_model_feature_columns_exclude_identifiers_and_targets():
    transactions = make_transactions()

    features, model_features = (
        build_validated_feature_dataset(transactions)
    )

    assert model_features

    assert not (
        set(model_features)
        & IDENTIFIER_COLUMNS
    )

    assert not (
        set(model_features)
        & TARGET_COLUMNS
    )


def test_model_feature_contract_accepts_valid_features():
    transactions = make_transactions()

    features, model_features = (
        build_validated_feature_dataset(transactions)
    )

    validate_model_feature_columns(
        model_features
    )


def test_model_feature_contract_rejects_target():
    with pytest.raises(ValueError):
        validate_model_feature_columns(
            [
                "log_amount",
                "is_fraud",
            ]
        )


def test_model_feature_contract_rejects_identifier():
    with pytest.raises(ValueError):
        validate_model_feature_columns(
            [
                "log_amount",
                "customer_id",
            ]
        )


def test_validated_dataset_preserves_targets():
    transactions = make_transactions()

    features, _ = build_validated_feature_dataset(
        transactions
    )

    assert "is_fraud" in features.columns
    assert "fraud_scenario" in features.columns

    assert features["is_fraud"].tolist() == [
        0,
        0,
        0,
        1,
        0,
    ]


def test_metadata_contains_feature_schema():
    transactions = make_transactions()

    features, model_features = (
        build_validated_feature_dataset(transactions)
    )

    metadata = build_feature_metadata(
        features=features,
        model_features=model_features,
    )

    assert metadata["dataset_name"] == (
        "fraud-intelligence-feature-dataset"
    )

    assert metadata["dataset_version"] == "1.0.0"
    assert metadata["phase"] == "4.9"
    assert metadata["row_count"] == 5

    assert metadata["model_feature_count"] == (
        len(model_features)
    )

    assert metadata["fraud_count"] == 1
    assert metadata["fraud_rate"] == pytest.approx(
        0.2
    )


def test_materialization_writes_parquet_and_metadata(
    tmp_path,
):
    transactions = make_transactions()

    artifacts = materialize_feature_dataset(
        transactions=transactions,
        output_dir=tmp_path,
    )

    assert artifacts.dataset_path.exists()
    assert artifacts.metadata_path.exists()

    loaded = pd.read_parquet(
        artifacts.dataset_path
    )

    assert loaded.shape == (
        5,
        67,
    )

    metadata = json.loads(
        artifacts.metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert metadata["row_count"] == 5
    assert metadata["column_count"] == 67
    assert metadata["model_feature_count"] > 0
    assert metadata["sha256"]


def test_materialized_dataset_is_reproducible(
    tmp_path,
):
    transactions = make_transactions()

    first = materialize_feature_dataset(
        transactions=transactions,
        output_dir=tmp_path / "first",
    )

    second = materialize_feature_dataset(
        transactions=transactions,
        output_dir=tmp_path / "second",
    )

    first_df = pd.read_parquet(
        first.dataset_path
    )

    second_df = pd.read_parquet(
        second.dataset_path
    )

    pd.testing.assert_frame_equal(
        first_df,
        second_df,
        check_dtype=True,
    )

    first_metadata = json.loads(
        first.metadata_path.read_text(
            encoding="utf-8"
        )
    )

    second_metadata = json.loads(
        second.metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        first_metadata["sha256"]
        == second_metadata["sha256"]
    )


def test_get_model_feature_columns_excludes_raw_transaction_columns():
    columns = [
        "transaction_id",
        "timestamp",
        "amount",
        "currency",
        "payment_method",
        "transaction_type",
        "customer_id",
        "log_amount",
        "hour",
        "is_fraud",
        "fraud_scenario",
    ]

    result = get_model_feature_columns(columns)

    assert result == [
        "log_amount",
        "hour",
    ]