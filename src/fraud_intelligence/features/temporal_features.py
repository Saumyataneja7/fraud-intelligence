from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
}


def validate_temporal_columns(
    transactions: pd.DataFrame,
) -> None:
    """
    Validate the timestamp required for temporal features.
    """
    missing = REQUIRED_COLUMNS.difference(
        transactions.columns
    )

    if missing:
        raise ValueError(
            "Missing required temporal columns: "
            f"{sorted(missing)}"
        )

    if not pd.api.types.is_datetime64_any_dtype(
        transactions["timestamp"]
    ):
        raise TypeError(
            "timestamp must be a pandas datetime column."
        )

    if transactions["timestamp"].dt.tz is None:
        raise TypeError(
            "timestamp must be timezone-aware."
        )


def add_temporal_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add point-in-time temporal features.

    These features use only the transaction's own timestamp.
    """
    validate_temporal_columns(transactions)

    result = transactions.copy()

    timestamp = result["timestamp"]

    result["hour"] = timestamp.dt.hour.astype("int8")

    result["day_of_week"] = (
        timestamp.dt.dayofweek.astype("int8")
    )

    result["day_of_month"] = (
        timestamp.dt.day.astype("int8")
    )

    result["month"] = (
        timestamp.dt.month.astype("int8")
    )

    result["quarter"] = (
        timestamp.dt.quarter.astype("int8")
    )

    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype("int8")

    result["is_night"] = (
        (result["hour"] < 6)
        | (result["hour"] >= 22)
    ).astype("int8")

    return result