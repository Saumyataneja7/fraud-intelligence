from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "customer_id",
    "amount",
    "merchant_id",
    "device_id",
    "ip_id",
}


HISTORICAL_FEATURE_COLUMNS = [
    "customer_txn_count_before",
    "customer_amount_sum_before",
    "customer_amount_mean_before",
    "customer_amount_median_before",
    "customer_amount_max_before",
    "customer_unique_merchants_before",
    "customer_unique_devices_before",
    "customer_unique_ips_before",
]


def validate_historical_columns(
    transactions: pd.DataFrame,
) -> None:
    """
    Validate columns required for customer historical features.
    """
    missing = REQUIRED_COLUMNS.difference(
        transactions.columns
    )

    if missing:
        raise ValueError(
            "Missing required historical feature columns: "
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

    if transactions["customer_id"].isna().any():
        raise ValueError(
            "customer_id cannot contain null values."
        )

def _historical_unique_count(
    dataframe: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.Series:
    """
    Calculate the number of distinct values observed strictly before
    each row within each group.
    """
    seen: dict[object, set] = {}
    output: list[int] = []

    for group_value, value in zip(
        dataframe[group_column],
        dataframe[value_column],
    ):
        group_seen = seen.setdefault(
            group_value,
            set(),
        )

        output.append(
            len(group_seen)
        )

        group_seen.add(value)

    return pd.Series(
        output,
        index=dataframe.index,
        dtype="int64",
    )


def _historical_max(
    dataframe: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.Series:
    """
    Calculate historical maximum using only strictly previous rows.
    """
    historical_max: dict[object, float] = {}
    output: list[float] = []

    for group_value, value in zip(
        dataframe[group_column],
        dataframe[value_column],
    ):
        previous_max = historical_max.get(
            group_value
        )

        if previous_max is None:
            output.append(0.0)
        else:
            output.append(previous_max)

        historical_max[group_value] = max(
            previous_max if previous_max is not None else value,
            value,
        )

    return pd.Series(
        output,
        index=dataframe.index,
        dtype="float64",
    )


def _historical_median(
    dataframe: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.Series:
    """
    Calculate historical median using only strictly previous rows.

    This implementation is intentionally simple and deterministic for the
    100K development dataset.
    """
    history: dict[object, list[float]] = {}
    output: list[float] = []

    for group_value, value in zip(
        dataframe[group_column],
        dataframe[value_column],
    ):
        previous_values = history.setdefault(
            group_value,
            [],
        )

        if not previous_values:
            output.append(0.0)
        else:
            output.append(
                float(
                    np.median(
                        previous_values
                    )
                )
            )

        previous_values.append(
            float(value)
        )

    return pd.Series(
        output,
        index=dataframe.index,
        dtype="float64",
    )

def add_customer_historical_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add leakage-safe customer historical features.

    For each transaction, only transactions with a timestamp strictly
    earlier than the current transaction timestamp are used.

    Same-timestamp transactions are intentionally excluded because the
    dataset does not provide an explicit event ordering.
    """
    validate_historical_columns(
        transactions
    )

    result = transactions.copy()

    original_index = result.index

    result["_original_order"] = np.arange(
        len(result)
    )

    result = result.sort_values(
        [
            "customer_id",
            "timestamp",
            "_original_order",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    customer_group = result.groupby(
        "customer_id",
        sort=False,
        dropna=False,
    )

    result["customer_txn_count_before"] = (
        customer_group.cumcount()
        .astype("int64")
    )

    result["customer_amount_sum_before"] = (
        customer_group["amount"]
        .transform(
            lambda values: values.cumsum()
            .shift(fill_value=0)
        )
        .astype("float64")
    )

    result["customer_amount_mean_before"] = (
        result["customer_amount_sum_before"]
        / result["customer_txn_count_before"]
        .replace(0, np.nan)
    ).fillna(0.0)

    result["customer_amount_median_before"] = (
        _historical_median(
            result,
            "customer_id",
            "amount",
        )
    )

    result["customer_amount_max_before"] = (
        _historical_max(
            result,
            "customer_id",
            "amount",
        )
    )

    result["customer_unique_merchants_before"] = (
        _historical_unique_count(
            result,
            "customer_id",
            "merchant_id",
        )
    )

    result["customer_unique_devices_before"] = (
        _historical_unique_count(
            result,
            "customer_id",
            "device_id",
        )
    )

    result["customer_unique_ips_before"] = (
        _historical_unique_count(
            result,
            "customer_id",
            "ip_id",
        )
    )

    result = result.sort_values(
        "_original_order"
    )

    result.index = original_index

    result = result.drop(
        columns=["_original_order"]
    )

    return result