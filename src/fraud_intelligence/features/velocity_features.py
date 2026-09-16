from __future__ import annotations

from collections import deque

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


VELOCITY_WINDOWS = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
}


VELOCITY_FEATURE_COLUMNS = [
    "customer_txn_count_1m",
    "customer_txn_count_5m",
    "customer_txn_count_15m",
    "customer_amount_sum_1m",
    "customer_amount_sum_5m",
    "customer_amount_sum_15m",
    "customer_unique_merchants_5m",
    "customer_unique_devices_5m",
    "customer_unique_ips_5m",
]

def _clean_amount_sum(value: float) -> float:
    """
    Remove floating-point noise from amount-window sums.

    Values extremely close to zero are mathematically zero.
    """
    if abs(value) < 1e-10:
        return 0.0

    return float(value)


def validate_velocity_columns(
    transactions: pd.DataFrame,
) -> None:
    """
    Validate columns required for velocity features.
    """
    missing = REQUIRED_COLUMNS.difference(
        transactions.columns
    )

    if missing:
        raise ValueError(
            "Missing required velocity feature columns: "
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

    if (transactions["amount"] <= 0).any():
        raise ValueError(
            "Transaction amounts must be positive."
        )
    
def _calculate_customer_velocity(
    dataframe: pd.DataFrame,
    window: pd.Timedelta,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate transaction count and amount sum for each customer
    within the historical window.

    Window semantics:

        current_timestamp - window
        <= historical_timestamp
        < current_timestamp

    Same-timestamp transactions are excluded.
    """
    counts = np.zeros(
        len(dataframe),
        dtype=np.int64,
    )

    amount_sums = np.zeros(
        len(dataframe),
        dtype=np.float64,
    )

    for _, group in dataframe.groupby(
        "customer_id",
        sort=False,
    ):
        timestamps = (
            group["timestamp"]
            .astype("int64")
            .to_numpy()
        )

        amounts = (
            group["amount"]
            .astype(float)
            .to_numpy()
        )

        left = 0
        right = 0

        running_sum = 0.0

        indices = group.index.to_numpy()

        for position in range(len(group)):
            current_timestamp = timestamps[
                position
            ]

            window_start = (
                current_timestamp
                - window.value
            )

            # Add strictly previous timestamps.
            while (
                right < position
                and timestamps[right]
                < current_timestamp
            ):
                running_sum += amounts[right]
                right += 1

            # Remove timestamps outside the window.
            while (
                left < right
                and timestamps[left]
                < window_start
            ):
                running_sum -= amounts[left]
                left += 1

            counts[
                indices[position]
            ] = right - left

            amount_sums[
                indices[position]
            ] = running_sum

    return (
        pd.Series(
            counts,
            index=dataframe.index,
            dtype="int64",
        ),
        pd.Series(
            amount_sums,
            index=dataframe.index,
            dtype="float64",
        ),
    )

def _calculate_unique_velocity(
    dataframe: pd.DataFrame,
    value_column: str,
    window: pd.Timedelta,
) -> pd.Series:
    """
    Calculate unique entity count in a customer's historical
    time window.

    Only timestamps strictly earlier than the current transaction
    are included.
    """
    result = np.zeros(
        len(dataframe),
        dtype=np.int64,
    )

    for _, group in dataframe.groupby(
        "customer_id",
        sort=False,
    ):
        timestamps = (
            group["timestamp"]
            .astype("int64")
            .to_numpy()
        )

        values = group[
            value_column
        ].to_numpy()

        left = 0
        right = 0

        active_values: dict[object, int] = {}

        indices = group.index.to_numpy()

        for position in range(len(group)):
            current_timestamp = timestamps[
                position
            ]

            window_start = (
                current_timestamp
                - window.value
            )

            while (
                right < position
                and timestamps[right]
                < current_timestamp
            ):
                value = values[right]

                active_values[value] = (
                    active_values.get(
                        value,
                        0,
                    )
                    + 1
                )

                right += 1

            while (
                left < right
                and timestamps[left]
                < window_start
            ):
                value = values[left]

                active_values[value] -= 1

                if active_values[value] == 0:
                    del active_values[value]

                left += 1

            result[
                indices[position]
            ] = len(active_values)

    return pd.Series(
        result,
        index=dataframe.index,
        dtype="int64",
    )

def add_velocity_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add leakage-safe customer velocity features.

    All velocity windows use only transactions that occurred
    strictly before the current transaction timestamp.
    """
    validate_velocity_columns(
        transactions
    )

    result = transactions.copy()

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

    for suffix, window in VELOCITY_WINDOWS.items():
        count, amount_sum = (
            _calculate_customer_velocity(
                result,
                window,
            )
        )

        result[
            f"customer_txn_count_{suffix}"
        ] = count

        result[
            f"customer_amount_sum_{suffix}"
        ] = np.where(
            np.isclose(
                amount_sum,
                0.0,
                atol=1e-10,
            ),
            0.0,
            amount_sum,
        )

    result[
        "customer_unique_merchants_5m"
    ] = _calculate_unique_velocity(
        result,
        "merchant_id",
        VELOCITY_WINDOWS["5m"],
    )

    result[
        "customer_unique_devices_5m"
    ] = _calculate_unique_velocity(
        result,
        "device_id",
        VELOCITY_WINDOWS["5m"],
    )

    result[
        "customer_unique_ips_5m"
    ] = _calculate_unique_velocity(
        result,
        "ip_id",
        VELOCITY_WINDOWS["5m"],
    )

    result = result.sort_values(
        "_original_order"
    )

    result = result.drop(
        columns="_original_order"
    )

    result = result.reset_index(
        drop=True
    )

    return result