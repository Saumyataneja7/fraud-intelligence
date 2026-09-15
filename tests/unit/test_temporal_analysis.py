import pandas as pd

from fraud_intelligence.analysis.temporal_analysis import (
    daily_fraud_activity,
    fraud_rate_by_day_of_week,
    fraud_rate_by_hour,
    fraud_rate_by_month,
    hourly_fraud_concentration,
    prepare_temporal_columns,
    velocity_windows,
)


def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [
                "TX001",
                "TX002",
                "TX003",
                "TX004",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00:00+00:00",
                    "2025-01-01 00:01:00+00:00",
                    "2025-01-02 12:00:00+00:00",
                    "2025-02-01 18:00:00+00:00",
                ],
                utc=True,
            ),
            "customer_id": [
                "C001",
                "C001",
                "C002",
                "C003",
            ],
            "amount": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
            "is_fraud": [
                0,
                1,
                0,
                1,
            ],
        }
    )


def test_prepare_temporal_columns():
    transactions = sample_transactions()

    result = prepare_temporal_columns(
        transactions
    )

    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert "month" in result.columns
    assert "date" in result.columns


def test_fraud_rate_by_hour():
    transactions = sample_transactions()

    result = fraud_rate_by_hour(
        transactions
    )

    assert result["transaction_count"].sum() == 4
    assert result["fraud_count"].sum() == 2


def test_fraud_rate_by_day_of_week():
    transactions = sample_transactions()

    result = fraud_rate_by_day_of_week(
        transactions
    )

    assert result["transaction_count"].sum() == 4
    assert result["fraud_count"].sum() == 2


def test_fraud_rate_by_month():
    transactions = sample_transactions()

    result = fraud_rate_by_month(
        transactions
    )

    assert result["transaction_count"].sum() == 4
    assert result["fraud_count"].sum() == 2


def test_daily_fraud_activity():
    transactions = sample_transactions()

    result = daily_fraud_activity(
        transactions
    )

    assert result["transaction_count"].sum() == 4
    assert result["fraud_count"].sum() == 2


def test_hourly_fraud_concentration():
    transactions = sample_transactions()

    result = hourly_fraud_concentration(
        transactions
    )

    assert result["fraud_count"].sum() == 2
    assert result["fraud_percentage"].sum() == 100


def test_velocity_windows():
    transactions = sample_transactions()

    result = velocity_windows(
        transactions
    )

    assert len(result) == 1
    assert result.iloc[0][
        "seconds_since_previous"
    ] == 60