import pandas as pd

from fraud_intelligence.analysis.eda import (
    cardinality_summary,
    dataset_overview,
    fraud_summary,
    missing_value_summary,
)


def make_test_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [
                "T1",
                "T2",
                "T3",
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
                "C1",
                "C1",
                "C2",
            ],
            "amount": [
                100.0,
                200.0,
                50.0,
            ],
            "is_fraud": [
                0,
                1,
                0,
            ],
        }
    )


def test_dataset_overview():
    df = make_test_transactions()

    result = dataset_overview(df)

    assert result["rows"] == 3
    assert result["columns"] == 5
    assert result["duplicate_rows"] == 0
    assert result["duplicate_transaction_ids"] == 0


def test_missing_value_summary():
    df = make_test_transactions()

    result = missing_value_summary(df)

    assert result["missing_count"].sum() == 0


def test_cardinality_summary():
    df = make_test_transactions()

    result = cardinality_summary(df)

    assert result.loc[
        "transaction_id",
        "unique_count",
    ] == 3

    assert result.loc[
        "customer_id",
        "unique_count",
    ] == 2


def test_fraud_summary():
    df = make_test_transactions()

    result = fraud_summary(df)

    assert result["total_transactions"] == 3
    assert result["fraud_transactions"] == 1
    assert result["non_fraud_transactions"] == 2

    assert result["fraud_rate_pct"] == 33.3333