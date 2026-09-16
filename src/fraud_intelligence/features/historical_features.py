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
    """Validate columns required for customer historical features."""

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

    if transactions["amount"].isna().any():
        raise ValueError(
            "amount cannot contain null values."
        )

    if (transactions["amount"] <= 0).any():
        raise ValueError(
            "Transaction amounts must be positive."
        )

    for column in [
        "merchant_id",
        "device_id",
        "ip_id",
    ]:
        if transactions[column].isna().any():
            raise ValueError(
                f"{column} cannot contain null values."
            )


def _customer_historical_statistics(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate strictly historical customer statistics.

    For every customer/timestamp pair, statistics include only
    transactions where:

        historical_timestamp < current_timestamp

    All transactions occurring at the same timestamp receive
    exactly the same historical values.
    """

    working = transactions[
        [
            "customer_id",
            "timestamp",
            "amount",
            "merchant_id",
            "device_id",
            "ip_id",
        ]
    ].copy()

    # ---------------------------------------------------------
    # Aggregate all transactions at the same customer/timestamp
    # into one block.
    # ---------------------------------------------------------

    blocks = (
        working
        .groupby(
            [
                "customer_id",
                "timestamp",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            block_txn_count=(
                "amount",
                "size",
            ),
            block_amount_sum=(
                "amount",
                "sum",
            ),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Calculate count and amount history at block level.
    # ---------------------------------------------------------

    blocks["customer_txn_count_before"] = (
        blocks
        .groupby(
            "customer_id",
            sort=False,
        )["block_txn_count"]
        .cumsum()
        - blocks["block_txn_count"]
    )

    blocks["customer_amount_sum_before"] = (
        blocks
        .groupby(
            "customer_id",
            sort=False,
        )["block_amount_sum"]
        .cumsum()
        - blocks["block_amount_sum"]
    )

    blocks["customer_amount_mean_before"] = np.where(
        blocks["customer_txn_count_before"] > 0,
        (
            blocks["customer_amount_sum_before"]
            / blocks["customer_txn_count_before"]
        ),
        0.0,
    )

    # ---------------------------------------------------------
    # Historical median and max.
    #
    # These are calculated from actual prior transactions,
    # never from the current timestamp block.
    # ---------------------------------------------------------

    median_values = []
    max_values = []

    for customer_id, customer_group in working.groupby(
        "customer_id",
        sort=False,
    ):
        customer_group = customer_group.sort_values(
            "timestamp",
            kind="mergesort",
        )

        previous_amounts: list[float] = []

        for timestamp, timestamp_group in customer_group.groupby(
            "timestamp",
            sort=True,
        ):
            if previous_amounts:
                median_value = float(
                    np.median(previous_amounts)
                )
                max_value = float(
                    np.max(previous_amounts)
                )
            else:
                median_value = 0.0
                max_value = 0.0

            median_values.append(
                (
                    customer_id,
                    timestamp,
                    median_value,
                )
            )

            max_values.append(
                (
                    customer_id,
                    timestamp,
                    max_value,
                )
            )

            previous_amounts.extend(
                timestamp_group[
                    "amount"
                ].astype(float).tolist()
            )

    median_lookup = pd.DataFrame(
        median_values,
        columns=[
            "customer_id",
            "timestamp",
            "customer_amount_median_before",
        ],
    )

    max_lookup = pd.DataFrame(
        max_values,
        columns=[
            "customer_id",
            "timestamp",
            "customer_amount_max_before",
        ],
    )

    blocks = blocks.merge(
        median_lookup,
        on=[
            "customer_id",
            "timestamp",
        ],
        how="left",
        validate="one_to_one",
    )

    blocks = blocks.merge(
        max_lookup,
        on=[
            "customer_id",
            "timestamp",
        ],
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # Historical unique entity counts.
    #
    # Again, only timestamps strictly before the current
    # timestamp are considered.
    # ---------------------------------------------------------

    unique_values = []

    for customer_id, customer_group in working.groupby(
        "customer_id",
        sort=False,
    ):
        customer_group = customer_group.sort_values(
            "timestamp",
            kind="mergesort",
        )

        previous_merchants: set = set()
        previous_devices: set = set()
        previous_ips: set = set()

        for timestamp, timestamp_group in customer_group.groupby(
            "timestamp",
            sort=True,
        ):
            unique_values.append(
                (
                    customer_id,
                    timestamp,
                    len(previous_merchants),
                    len(previous_devices),
                    len(previous_ips),
                )
            )

            previous_merchants.update(
                timestamp_group[
                    "merchant_id"
                ].tolist()
            )

            previous_devices.update(
                timestamp_group[
                    "device_id"
                ].tolist()
            )

            previous_ips.update(
                timestamp_group[
                    "ip_id"
                ].tolist()
            )

    unique_lookup = pd.DataFrame(
        unique_values,
        columns=[
            "customer_id",
            "timestamp",
            "customer_unique_merchants_before",
            "customer_unique_devices_before",
            "customer_unique_ips_before",
        ],
    )

    blocks = blocks.merge(
        unique_lookup,
        on=[
            "customer_id",
            "timestamp",
        ],
        how="left",
        validate="one_to_one",
    )

    return blocks


def add_customer_historical_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add leakage-safe customer historical features.

    Historical information is strictly:

        timestamp < current transaction timestamp

    Same-timestamp transactions cannot influence each other.

    Original row order is preserved.
    """

    validate_historical_columns(
        transactions
    )

    result = transactions.copy()

    # Keep explicit original position.
    result["_original_order"] = np.arange(
        len(result)
    )

    blocks = _customer_historical_statistics(
        result
    )

    # ---------------------------------------------------------
    # Map customer/timestamp history back to transactions.
    # ---------------------------------------------------------

    result = result.merge(
        blocks[
            [
                "customer_id",
                "timestamp",
                *HISTORICAL_FEATURE_COLUMNS,
            ]
        ],
        on=[
            "customer_id",
            "timestamp",
        ],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    # Restore original transaction order.
    result = result.sort_values(
        "_original_order",
        kind="mergesort",
    )

    result = result.drop(
        columns="_original_order"
    )

    result = result.reset_index(
        drop=True
    )

    # Enforce expected dtypes.
    result[
        "customer_txn_count_before"
    ] = result[
        "customer_txn_count_before"
    ].astype("int64")

    for column in [
        "customer_amount_sum_before",
        "customer_amount_mean_before",
        "customer_amount_median_before",
        "customer_amount_max_before",
    ]:
        result[column] = result[column].astype(
            "float64"
        )

    for column in [
        "customer_unique_merchants_before",
        "customer_unique_devices_before",
        "customer_unique_ips_before",
    ]:
        result[column] = result[column].astype(
            "int64"
        )

    return result


__all__ = [
    "HISTORICAL_FEATURE_COLUMNS",
    "add_customer_historical_features",
    "validate_historical_columns",
]