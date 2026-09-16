from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "customer_id",
    "timestamp",
    "amount",
    "payment_method",
    "transaction_type",
}


BEHAVIORAL_FEATURE_COLUMNS = [
    "amount_vs_customer_mean",
    "amount_vs_customer_median",
    "amount_vs_customer_max",
    "amount_deviation_from_customer_mean",
    "is_new_payment_method",
    "is_new_transaction_type",
]


def validate_behavioral_columns(
    transactions: pd.DataFrame,
) -> None:
    """Validate columns required for behavioral deviation features."""

    missing = REQUIRED_COLUMNS.difference(
        transactions.columns
    )

    if missing:
        raise ValueError(
            "Missing required behavioral feature columns: "
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

    if transactions["payment_method"].isna().any():
        raise ValueError(
            "payment_method cannot contain null values."
        )

    if transactions["transaction_type"].isna().any():
        raise ValueError(
            "transaction_type cannot contain null values."
        )


def _historical_amount_statistics(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Calculate strictly historical customer amount statistics.

    For a transaction at timestamp T, only transactions belonging
    to the same customer with timestamp < T are considered.

    Same-timestamp transactions are processed as a single block.
    """

    working = dataframe[
        [
            "customer_id",
            "timestamp",
            "amount",
        ]
    ].copy()

    working["_row_position"] = np.arange(
        len(working)
    )

    # Aggregate customer/timestamp blocks.
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
            block_count=(
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

    # Sort chronologically within customer.
    blocks = blocks.sort_values(
        [
            "customer_id",
            "timestamp",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # Historical count before the current timestamp.
    blocks["historical_count"] = (
        blocks
        .groupby(
            "customer_id",
            sort=False,
        )["block_count"]
        .cumsum()
        - blocks["block_count"]
    )

    # Historical amount sum before current timestamp.
    blocks["historical_amount_sum"] = (
        blocks
        .groupby(
            "customer_id",
            sort=False,
        )["block_amount_sum"]
        .cumsum()
        - blocks["block_amount_sum"]
    )

    # Historical mean.
    blocks["historical_mean"] = np.where(
        blocks["historical_count"] > 0,
        (
            blocks["historical_amount_sum"]
            / blocks["historical_count"]
        ),
        0.0,
    )

    # Build customer/timestamp lookups for median and max.
    median_lookup = {}
    max_lookup = {}

    for customer_id, customer_group in working.groupby(
        "customer_id",
        sort=False,
    ):
        customer_group = customer_group.sort_values(
            "timestamp",
            kind="mergesort",
        )

        previous_amounts = []

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

            median_lookup[
                (customer_id, timestamp)
            ] = median_value

            max_lookup[
                (customer_id, timestamp)
            ] = max_value

            previous_amounts.extend(
                timestamp_group[
                    "amount"
                ].astype(float).tolist()
            )

    blocks["historical_median"] = [
        median_lookup[
            (
                row["customer_id"],
                row["timestamp"],
            )
        ]
        for _, row in blocks.iterrows()
    ]

    blocks["historical_max"] = [
        max_lookup[
            (
                row["customer_id"],
                row["timestamp"],
            )
        ]
        for _, row in blocks.iterrows()
    ]

    lookup = blocks.set_index(
        [
            "customer_id",
            "timestamp",
        ]
    )[
        [
            "historical_mean",
            "historical_median",
            "historical_max",
        ]
    ]

    keys = pd.MultiIndex.from_arrays(
        [
            dataframe["customer_id"].to_numpy(),
            dataframe["timestamp"].to_numpy(),
        ],
        names=[
            "customer_id",
            "timestamp",
        ],
    )

    matched = lookup.reindex(keys)

    return (
        pd.Series(
            matched["historical_mean"].to_numpy(),
            index=dataframe.index,
            dtype="float64",
        ),
        pd.Series(
            matched["historical_median"].to_numpy(),
            index=dataframe.index,
            dtype="float64",
        ),
        pd.Series(
            matched["historical_max"].to_numpy(),
            index=dataframe.index,
            dtype="float64",
        ),
    )


def _historical_novelty(
    dataframe: pd.DataFrame,
    value_column: str,
) -> pd.Series:
    """
    Determine whether a customer is using a categorical value
    for the first time.

    Same-timestamp transactions cannot influence one another.
    """

    working = dataframe[
        [
            "customer_id",
            "timestamp",
            value_column,
        ]
    ].copy()

    working["_row_position"] = np.arange(
        len(working)
    )

    blocks = (
        working
        .groupby(
            [
                "customer_id",
                value_column,
                "timestamp",
            ],
            sort=True,
            dropna=False,
        )
        .size()
        .reset_index(
            name="block_count"
        )
    )

    blocks["previous_count"] = (
        blocks
        .groupby(
            [
                "customer_id",
                value_column,
            ],
            sort=False,
        )["block_count"]
        .cumsum()
        - blocks["block_count"]
    )

    blocks["is_new"] = (
        blocks["previous_count"] == 0
    ).astype("int8")

    lookup = blocks.set_index(
        [
            "customer_id",
            value_column,
            "timestamp",
        ]
    )["is_new"]

    keys = pd.MultiIndex.from_arrays(
        [
            dataframe["customer_id"].to_numpy(),
            dataframe[value_column].to_numpy(),
            dataframe["timestamp"].to_numpy(),
        ],
        names=[
            "customer_id",
            value_column,
            "timestamp",
        ],
    )

    matched = lookup.reindex(keys)

    return pd.Series(
        matched.to_numpy(),
        index=dataframe.index,
        dtype="int8",
    )


def add_behavioral_deviation_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add leakage-safe customer behavioral deviation features.

    Numeric features compare the current transaction amount with
    the customer's strictly historical transaction behavior.

    Categorical novelty features identify first-time payment methods
    and transaction types for the customer.

    Same-timestamp and future transactions are excluded.
    Original row order is preserved.
    """

    validate_behavioral_columns(
        transactions
    )

    result = transactions.copy()

    (
        historical_mean,
        historical_median,
        historical_max,
    ) = _historical_amount_statistics(
        result
    )

    result[
        "amount_vs_customer_mean"
    ] = np.where(
        historical_mean > 0,
        result["amount"]
        / historical_mean,
        0.0,
    )

    result[
        "amount_vs_customer_median"
    ] = np.where(
        historical_median > 0,
        result["amount"]
        / historical_median,
        0.0,
    )

    result[
        "amount_vs_customer_max"
    ] = np.where(
        historical_max > 0,
        result["amount"]
        / historical_max,
        0.0,
    )

    result[
        "amount_deviation_from_customer_mean"
    ] = np.where(
        historical_mean > 0,
        (
            result["amount"]
            - historical_mean
        )
        / historical_mean,
        0.0,
    )

    result[
        "is_new_payment_method"
    ] = _historical_novelty(
        result,
        "payment_method",
    ).to_numpy()

    result[
        "is_new_transaction_type"
    ] = _historical_novelty(
        result,
        "transaction_type",
    ).to_numpy()

    return result