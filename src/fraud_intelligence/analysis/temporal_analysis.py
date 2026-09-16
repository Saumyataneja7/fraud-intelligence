from __future__ import annotations

import pandas as pd


def prepare_temporal_columns(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Add derived temporal columns for EDA."""

    result = transactions.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
    )

    result["hour"] = result["timestamp"].dt.hour

    result["day_of_week"] = (
        result["timestamp"]
        .dt.dayofweek
    )

    result["day_name"] = (
        result["timestamp"]
        .dt.day_name()
    )

    result["month"] = (
        result["timestamp"]
        .dt.month
    )

    result["month_name"] = (
        result["timestamp"]
        .dt.month_name()
    )

    result["date"] = (
        result["timestamp"]
        .dt.date
    )

    return result


def fraud_rate_by_hour(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate transaction volume and fraud rate by UTC hour."""

    data = prepare_temporal_columns(
        transactions
    )

    result = (
        data.groupby("hour")
        .agg(
            transaction_count=(
                "transaction_id",
                "count",
            ),
            fraud_count=(
                "is_fraud",
                "sum",
            ),
        )
        .reset_index()
    )

    result["fraud_rate_pct"] = (
        result["fraud_count"]
        / result["transaction_count"]
        * 100
    ).round(4)

    return result.sort_values("hour")


def fraud_rate_by_day_of_week(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate transaction volume and fraud rate by day of week."""

    data = prepare_temporal_columns(
        transactions
    )

    result = (
        data.groupby(
            [
                "day_of_week",
                "day_name",
            ]
        )
        .agg(
            transaction_count=(
                "transaction_id",
                "count",
            ),
            fraud_count=(
                "is_fraud",
                "sum",
            ),
        )
        .reset_index()
    )

    result["fraud_rate_pct"] = (
        result["fraud_count"]
        / result["transaction_count"]
        * 100
    ).round(4)

    return result.sort_values(
        "day_of_week"
    )


def fraud_rate_by_month(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate transaction volume and fraud rate by month."""

    data = prepare_temporal_columns(
        transactions
    )

    result = (
        data.groupby(
            [
                "month",
                "month_name",
            ]
        )
        .agg(
            transaction_count=(
                "transaction_id",
                "count",
            ),
            fraud_count=(
                "is_fraud",
                "sum",
            ),
        )
        .reset_index()
    )

    result["fraud_rate_pct"] = (
        result["fraud_count"]
        / result["transaction_count"]
        * 100
    ).round(4)

    return result.sort_values("month")


def daily_fraud_activity(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate daily transaction and fraud activity."""

    data = prepare_temporal_columns(
        transactions
    )

    result = (
        data.groupby("date")
        .agg(
            transaction_count=(
                "transaction_id",
                "count",
            ),
            fraud_count=(
                "is_fraud",
                "sum",
            ),
            total_amount=(
                "amount",
                "sum",
            ),
            fraud_amount=(
                "amount",
                lambda values: values[
                    data.loc[
                        values.index,
                        "is_fraud",
                    ].eq(1)
                ].sum(),
            ),
        )
        .reset_index()
    )

    result["fraud_rate_pct"] = (
        result["fraud_count"]
        / result["transaction_count"]
        * 100
    ).round(4)

    return result


def hourly_fraud_concentration(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Measure how fraud transactions are distributed across hours."""

    data = prepare_temporal_columns(
        transactions
    )

    fraud = data[
        data["is_fraud"] == 1
    ]

    result = (
        fraud["hour"]
        .value_counts()
        .rename_axis("hour")
        .reset_index(
            name="fraud_count"
        )
    )

    result["fraud_percentage"] = (
        result["fraud_count"]
        / len(fraud)
        * 100
    ).round(4)

    return result.sort_values("hour")


def velocity_windows(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Measure short-term transaction bursts per customer."""

    data = transactions.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
    )

    data = data.sort_values(
        [
            "customer_id",
            "timestamp",
        ]
    )

    data["seconds_since_previous"] = (
        data.groupby("customer_id")[
            "timestamp"
        ]
        .diff()
        .dt.total_seconds()
    )

    result = data[
        data["seconds_since_previous"]
        .notna()
    ].copy()

    result["within_1_minute"] = (
        result["seconds_since_previous"]
        <= 60
    )

    result["within_5_minutes"] = (
        result["seconds_since_previous"]
        <= 300
    )

    result["within_15_minutes"] = (
        result["seconds_since_previous"]
        <= 900
    )

    return result[
        [
            "transaction_id",
            "customer_id",
            "timestamp",
            "is_fraud",
            "seconds_since_previous",
            "within_1_minute",
            "within_5_minutes",
            "within_15_minutes",
        ]
    ]