from __future__ import annotations

import pandas as pd
import pytest

from fraud_intelligence.ml.splits import (
    TemporalSplitError,
    build_split_metadata,
    temporal_train_validation_test_split,
    validate_temporal_split,
)


def make_fixture() -> pd.DataFrame:
    """Create a small deterministic feature dataset."""

    return pd.DataFrame(
        {
            "transaction_id": [f"T{i}" for i in range(1, 21)],
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=20,
                freq="h",
                tz="UTC",
            ),
            "amount": range(20),
            "log_amount": range(20),
            "is_fraud": [0, 1] * 10,
        }
    )


def test_temporal_split_uses_expected_row_counts():
    df = make_fixture()

    split = temporal_train_validation_test_split(df)

    assert len(split.train) == 14
    assert len(split.validation) == 3
    assert len(split.test) == 3


def test_temporal_split_is_chronological():
    df = make_fixture()

    split = temporal_train_validation_test_split(df)

    assert (
        split.train["timestamp"].max()
        <= split.validation["timestamp"].min()
    )

    assert (
        split.validation["timestamp"].max()
        <= split.test["timestamp"].min()
    )


def test_each_split_is_internally_sorted():
    df = make_fixture().sample(frac=1.0, random_state=42)

    split = temporal_train_validation_test_split(df)

    assert split.train["timestamp"].is_monotonic_increasing
    assert split.validation["timestamp"].is_monotonic_increasing
    assert split.test["timestamp"].is_monotonic_increasing


def test_split_preserves_all_rows():
    df = make_fixture()

    split = temporal_train_validation_test_split(df)

    assert split.total_rows == len(df)

    transaction_ids = (
        set(split.train["transaction_id"])
        | set(split.validation["transaction_id"])
        | set(split.test["transaction_id"])
    )

    assert transaction_ids == set(df["transaction_id"])


def test_split_does_not_duplicate_rows():
    df = make_fixture()

    split = temporal_train_validation_test_split(df)

    combined_ids = (
        list(split.train["transaction_id"])
        + list(split.validation["transaction_id"])
        + list(split.test["transaction_id"])
    )

    assert len(combined_ids) == len(set(combined_ids))


def test_original_dataframe_is_not_modified():
    df = make_fixture()
    original = df.copy(deep=True)

    temporal_train_validation_test_split(df)

    pd.testing.assert_frame_equal(df, original)


def test_custom_ratios_are_supported():
    df = make_fixture()

    split = temporal_train_validation_test_split(
        df,
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
    )

    assert len(split.train) == 12
    assert len(split.validation) == 4
    assert len(split.test) == 4


def test_invalid_ratios_raise():
    df = make_fixture()

    with pytest.raises(TemporalSplitError):
        temporal_train_validation_test_split(
            df,
            train_ratio=0.70,
            validation_ratio=0.20,
            test_ratio=0.20,
        )


def test_missing_timestamp_raises():
    df = make_fixture().drop(columns=["timestamp"])

    with pytest.raises(TemporalSplitError):
        temporal_train_validation_test_split(df)


def test_missing_target_raises():
    df = make_fixture().drop(columns=["is_fraud"])

    with pytest.raises(TemporalSplitError):
        temporal_train_validation_test_split(df)


def test_empty_dataset_raises():
    df = make_fixture().iloc[0:0]

    with pytest.raises(TemporalSplitError):
        temporal_train_validation_test_split(df)


def test_split_metadata_contains_temporal_boundaries():
    df = make_fixture()

    split = temporal_train_validation_test_split(df)
    metadata = build_split_metadata(split)

    assert metadata["split_strategy"] == "chronological"
    assert metadata["train_rows"] == 14
    assert metadata["validation_rows"] == 3
    assert metadata["test_rows"] == 3
    assert metadata["total_rows"] == 20

    assert metadata["train_start"] < metadata["train_end"]
    assert metadata["validation_start"] < metadata["validation_end"]
    assert metadata["test_start"] < metadata["test_end"]


def test_validate_temporal_split_accepts_valid_split():
    df = make_fixture()

    split = temporal_train_validation_test_split(df)

    validate_temporal_split(split)