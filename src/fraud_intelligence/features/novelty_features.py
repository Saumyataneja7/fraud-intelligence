from __future__ import annotations

import numpy as np
import pandas as pd


NOVELTY_ENTITY_COLUMNS = [
    "device_id",
    "ip_id",
    "merchant_id",
    "card_id",
]


NOVELTY_FEATURE_COLUMNS = [
    "is_new_device",
    "is_new_ip",
    "is_new_merchant",
    "is_new_card",
]


REQUIRED_COLUMNS = {
    "customer_id",
    "timestamp",
    *NOVELTY_ENTITY_COLUMNS,
}


def validate_novelty_columns(
    transactions: pd.DataFrame,
) -> None:
    """Validate columns required for novelty features."""

    missing = REQUIRED_COLUMNS.difference(
        transactions.columns
    )

    if missing:
        raise ValueError(
            "Missing required novelty feature columns: "
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

    for column in NOVELTY_ENTITY_COLUMNS:
        if transactions[column].isna().any():
            raise ValueError(
                f"{column} cannot contain null values."
            )


def _customer_entity_novelty(
    dataframe: pd.DataFrame,
    entity_column: str,
) -> pd.Series:
    """
    Determine whether a customer is using an entity for the first time.

    A transaction is considered new when the pair:

        (customer_id, entity_id)

    has not appeared in any transaction with:

        timestamp < current timestamp

    Same-timestamp transactions are treated as a single block and
    therefore cannot make one another non-new.
    """

    working = dataframe[
        [
            "customer_id",
            entity_column,
            "timestamp",
        ]
    ].copy()

    working["_row_position"] = np.arange(
        len(working)
    )

    # ---------------------------------------------------------
    # Create customer + entity + timestamp blocks.
    # ---------------------------------------------------------

    blocks = (
        working
        .groupby(
            [
                "customer_id",
                entity_column,
                "timestamp",
            ],
            sort=True,
            dropna=False,
        )
        .size()
        .reset_index(
            name="transaction_count"
        )
    )

    # ---------------------------------------------------------
    # For each customer/entity pair, determine whether there
    # was any earlier timestamp.
    # ---------------------------------------------------------

    blocks["previous_transaction_count"] = (
        blocks
        .groupby(
            [
                "customer_id",
                entity_column,
            ],
            sort=False,
        )["transaction_count"]
        .cumsum()
        - blocks["transaction_count"]
    )

    blocks["is_new"] = (
        blocks["previous_transaction_count"] == 0
    ).astype("int8")

    # ---------------------------------------------------------
    # Map the customer/entity/timestamp result back to every
    # original transaction.
    # ---------------------------------------------------------

    lookup = blocks.set_index(
        [
            "customer_id",
            entity_column,
            "timestamp",
        ]
    )["is_new"]

    keys = pd.MultiIndex.from_arrays(
        [
            dataframe["customer_id"].to_numpy(),
            dataframe[entity_column].to_numpy(),
            dataframe["timestamp"].to_numpy(),
        ],
        names=[
            "customer_id",
            entity_column,
            "timestamp",
        ],
    )

    matched = lookup.reindex(keys)

    return pd.Series(
        matched.to_numpy(),
        index=dataframe.index,
        dtype="int8",
    )


def add_novelty_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add leakage-safe customer/entity novelty features.

    Features:

    - is_new_device
    - is_new_ip
    - is_new_merchant
    - is_new_card

    A value of 1 means the customer is using that entity for the
    first time based only on strictly earlier transactions.

    Same-timestamp and future transactions are excluded.
    Original row order is preserved.
    """

    validate_novelty_columns(
        transactions
    )

    result = transactions.copy()

    for entity_column in NOVELTY_ENTITY_COLUMNS:
        novelty = _customer_entity_novelty(
            result,
            entity_column,
        )

        feature_name = (
            f"is_new_{entity_column.removesuffix('_id')}"
        )

        result[feature_name] = novelty.to_numpy()

    return result