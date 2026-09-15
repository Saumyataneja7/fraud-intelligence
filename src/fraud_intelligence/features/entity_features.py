from __future__ import annotations

import numpy as np
import pandas as pd


ENTITY_COLUMNS = [
    "account_id",
    "card_id",
    "merchant_id",
    "device_id",
    "ip_id",
]


REQUIRED_COLUMNS = {
    "timestamp",
    "amount",
    *ENTITY_COLUMNS,
}


ENTITY_HISTORICAL_FEATURE_COLUMNS = [
    "account_txn_count_before",
    "card_txn_count_before",
    "merchant_txn_count_before",
    "device_txn_count_before",
    "ip_txn_count_before",
    "account_amount_sum_before",
    "card_amount_sum_before",
    "merchant_amount_sum_before",
    "device_amount_sum_before",
    "ip_amount_sum_before",
]


def validate_entity_columns(
    transactions: pd.DataFrame,
) -> None:
    """Validate the input required for entity historical features."""

    missing = REQUIRED_COLUMNS.difference(
        transactions.columns
    )

    if missing:
        raise ValueError(
            "Missing required entity feature columns: "
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

    if (transactions["amount"] <= 0).any():
        raise ValueError(
            "Transaction amounts must be positive."
        )

    for column in ENTITY_COLUMNS:
        if transactions[column].isna().any():
            raise ValueError(
                f"{column} cannot contain null values."
            )


def _historical_entity_statistics(
    dataframe: pd.DataFrame,
    entity_column: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate point-in-time historical statistics for one entity.

    For an entity E and transaction at timestamp T:

        history(E, T) =
            all transactions for E where timestamp < T

    Transactions sharing timestamp T are treated as one block.
    """

    working = dataframe[
        [
            entity_column,
            "timestamp",
            "amount",
        ]
    ].copy()

    # ---------------------------------------------------------
    # Build one row per entity + timestamp.
    # ---------------------------------------------------------

    blocks = (
        working
        .groupby(
            [
                entity_column,
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
    # Historical count:
    #
    # cumulative total - current timestamp block
    # ---------------------------------------------------------

    blocks["historical_count"] = (
        blocks
        .groupby(
            entity_column,
            sort=False,
        )["block_txn_count"]
        .cumsum()
        - blocks["block_txn_count"]
    )

    # ---------------------------------------------------------
    # Historical amount:
    #
    # cumulative amount - current timestamp block
    # ---------------------------------------------------------

    blocks["historical_amount_sum"] = (
        blocks
        .groupby(
            entity_column,
            sort=False,
        )["block_amount_sum"]
        .cumsum()
        - blocks["block_amount_sum"]
    )

    # ---------------------------------------------------------
    # Create an explicit lookup:
    #
    # (entity, timestamp) -> historical statistics
    # ---------------------------------------------------------

    lookup = blocks.set_index(
        [
            entity_column,
            "timestamp",
        ]
    )[
        [
            "historical_count",
            "historical_amount_sum",
        ]
    ]

    # ---------------------------------------------------------
    # Look up every original transaction directly.
    # ---------------------------------------------------------

    keys = pd.MultiIndex.from_arrays(
        [
            dataframe[entity_column].to_numpy(),
            dataframe["timestamp"].to_numpy(),
        ],
        names=[
            entity_column,
            "timestamp",
        ],
    )

    matched = lookup.reindex(keys)

    counts = pd.Series(
        matched["historical_count"].to_numpy(),
        index=dataframe.index,
        dtype="int64",
    )

    amount_sums = pd.Series(
        matched["historical_amount_sum"].to_numpy(),
        index=dataframe.index,
        dtype="float64",
    )

    return counts, amount_sums


def add_entity_historical_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add leakage-safe historical features for:

    - Account
    - Card
    - Merchant
    - Device
    - IP

    Historical information is strictly:

        entity_timestamp < transaction_timestamp

    Same-timestamp and future transactions are excluded.
    Original row order is preserved.
    """

    validate_entity_columns(
        transactions
    )

    result = transactions.copy()

    for entity_column in ENTITY_COLUMNS:

        counts, amount_sums = (
            _historical_entity_statistics(
                result,
                entity_column,
            )
        )

        prefix = entity_column.removesuffix(
            "_id"
        )

        result[
            f"{prefix}_txn_count_before"
        ] = counts.to_numpy()

        result[
            f"{prefix}_amount_sum_before"
        ] = amount_sums.to_numpy()

    return result