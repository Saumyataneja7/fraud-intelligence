from __future__ import annotations

import pandas as pd
import pytest

from fraud_intelligence.features.leakage_validation import (
    ENGINEERED_FEATURE_COLUMNS,
)
from fraud_intelligence.ml.dataset import (
    ModelDatasetPreparationError,
    build_model_dataset_summary,
    prepare_model_dataset,
    validate_model_dataset,
)
from fraud_intelligence.ml.splits import (
    temporal_train_validation_test_split,
)



def make_feature_fixture() -> pd.DataFrame:
    """Create a small dataset containing the complete Phase 4 feature set."""

    timestamps = pd.date_range(
        "2025-01-01",
        periods=20,
        freq="h",
        tz="UTC",
    )

    data = {
        "transaction_id": [f"T{i}" for i in range(20)],
        "customer_id": [f"C{i % 5}" for i in range(20)],
        "account_id": [f"A{i % 5}" for i in range(20)],
        "card_id": [f"CA{i % 5}" for i in range(20)],
        "merchant_id": [f"M{i % 3}" for i in range(20)],
        "device_id": [f"D{i % 4}" for i in range(20)],
        "ip_id": [f"IP{i % 4}" for i in range(20)],
        "timestamp": timestamps,
        "amount": [100.0 + i for i in range(20)],
        "currency": ["USD"] * 20,
        "payment_method": ["card"] * 20,
        "transaction_type": ["purchase"] * 20,
        "is_fraud": [0, 1] * 10,
        "fraud_scenario": [""] * 20,
    }

    # Populate every approved Phase 4 engineered feature.
    #
    # Values are deterministic because this is a unit-test fixture.
    # The purpose here is schema validation, not feature-value testing.
    for index, feature in enumerate(ENGINEERED_FEATURE_COLUMNS):
        if feature not in data:
            data[feature] = [
                float(index + row)
                for row in range(20)
            ]

    # Keep a few representative values meaningful.
    data["log_amount"] = [
        1.0 + i * 0.01
        for i in range(20)
    ]

    data["hour"] = list(range(20))
    data["day_of_week"] = [i % 7 for i in range(20)]
    data["day_of_month"] = [i + 1 for i in range(20)]
    data["month"] = [1] * 20
    data["quarter"] = [1] * 20
    data["is_weekend"] = [0] * 20
    data["is_night"] = [0] * 20

    return pd.DataFrame(data)


def make_split():
    df = make_feature_fixture()

    return temporal_train_validation_test_split(df)


def test_model_dataset_contains_only_model_features():
    split = make_split()

    dataset = prepare_model_dataset(split)

    assert dataset.n_features == 53

    forbidden = {
        "transaction_id",
        "customer_id",
        "account_id",
        "card_id",
        "merchant_id",
        "device_id",
        "ip_id",
        "timestamp",
        "amount",
        "currency",
        "payment_method",
        "transaction_type",
        "is_fraud",
        "fraud_scenario",
    }

    assert not forbidden.intersection(
        dataset.X_train.columns
    )


def test_targets_are_separate_from_features():
    split = make_split()

    dataset = prepare_model_dataset(split)

    assert "is_fraud" not in dataset.X_train.columns
    assert "is_fraud" not in dataset.X_validation.columns
    assert "is_fraud" not in dataset.X_test.columns

    assert dataset.y_train.name == "is_fraud"
    assert dataset.y_validation.name == "is_fraud"
    assert dataset.y_test.name == "is_fraud"


def test_transaction_ids_are_retained_separately():
    split = make_split()

    dataset = prepare_model_dataset(split)

    assert "transaction_id" not in dataset.X_train.columns
    assert "transaction_id" not in dataset.X_validation.columns
    assert "transaction_id" not in dataset.X_test.columns

    assert len(dataset.train_ids) == len(dataset.X_train)
    assert len(dataset.validation_ids) == len(
        dataset.X_validation
    )
    assert len(dataset.test_ids) == len(dataset.X_test)


def test_model_dataset_preserves_split_row_counts():
    split = make_split()

    dataset = prepare_model_dataset(split)

    assert len(dataset.X_train) == 14
    assert len(dataset.X_validation) == 3
    assert len(dataset.X_test) == 3

    assert len(dataset.y_train) == 14
    assert len(dataset.y_validation) == 3
    assert len(dataset.y_test) == 3


def test_model_dataset_has_aligned_indexes():
    split = make_split()

    dataset = prepare_model_dataset(split)

    assert list(dataset.X_train.index) == list(
        range(len(dataset.X_train))
    )

    assert list(dataset.y_train.index) == list(
        range(len(dataset.y_train))
    )

    assert list(dataset.train_ids.index) == list(
        range(len(dataset.train_ids))
    )


def test_model_dataset_feature_columns_are_consistent():
    split = make_split()

    dataset = prepare_model_dataset(split)

    assert list(dataset.X_train.columns) == list(
        dataset.feature_columns
    )

    assert list(dataset.X_validation.columns) == list(
        dataset.feature_columns
    )

    assert list(dataset.X_test.columns) == list(
        dataset.feature_columns
    )


def test_model_dataset_target_is_binary():
    split = make_split()

    dataset = prepare_model_dataset(split)

    for target in (
        dataset.y_train,
        dataset.y_validation,
        dataset.y_test,
    ):
        assert set(target.unique()).issubset({0, 1})


def test_model_dataset_has_no_transaction_overlap():
    split = make_split()

    dataset = prepare_model_dataset(split)

    train_ids = set(dataset.train_ids)
    validation_ids = set(dataset.validation_ids)
    test_ids = set(dataset.test_ids)

    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids


def test_model_dataset_validation_accepts_valid_dataset():
    split = make_split()

    dataset = prepare_model_dataset(split)

    validate_model_dataset(dataset)


def test_model_dataset_summary():
    split = make_split()

    dataset = prepare_model_dataset(split)
    summary = build_model_dataset_summary(dataset)

    assert summary["n_features"] == 53
    assert summary["train_rows"] == 14
    assert summary["validation_rows"] == 3
    assert summary["test_rows"] == 3

    assert (
        summary["train_fraud_count"]
        + summary["validation_fraud_count"]
        + summary["test_fraud_count"]
        == 10
    )


def test_missing_model_feature_raises():
    split = make_split()

    train = split.train.drop(columns=["log_amount"])

    modified_split = type(split)(
        train=train,
        validation=split.validation,
        test=split.test,
    )

    with pytest.raises(ModelDatasetPreparationError):
        prepare_model_dataset(modified_split)


def test_target_leakage_is_rejected():
    split = make_split()

    # The preparation function obtains the approved feature list
    # from Phase 4, so an unexpected target-like feature must not
    # silently become a model input.
    train = split.train.copy()
    train["fraud_probability"] = train["is_fraud"]

    modified_split = type(split)(
        train=train,
        validation=split.validation,
        test=split.test,
    )

    dataset = prepare_model_dataset(modified_split)

    assert "fraud_probability" not in dataset.X_train.columns


def test_summary_contains_fraud_rates():
    split = make_split()

    dataset = prepare_model_dataset(split)
    summary = build_model_dataset_summary(dataset)

    assert 0.0 <= summary["train_fraud_rate"] <= 1.0
    assert 0.0 <= summary["validation_fraud_rate"] <= 1.0
    assert 0.0 <= summary["test_fraud_rate"] <= 1.0