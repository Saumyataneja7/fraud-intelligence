from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "amount",
    "payment_method",
    "transaction_type",
}


def validate_transaction_columns(
    transactions: pd.DataFrame,
) -> None:
    """
    Validate the columns required to construct transaction features.
    """
    missing = REQUIRED_COLUMNS.difference(
        transactions.columns
    )

    if missing:
        raise ValueError(
            "Missing required transaction columns: "
            f"{sorted(missing)}"
        )


def add_transaction_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add transaction-level features.

    These features depend only on the transaction itself and therefore
    do not require historical information.
    """
    validate_transaction_columns(transactions)

    result = transactions.copy()

    if (result["amount"] <= 0).any():
        raise ValueError(
            "Transaction amounts must be positive."
        )

    result["log_amount"] = np.log1p(
        result["amount"].astype(float)
    )

    result["is_card_payment"] = (
        result["payment_method"] == "card"
    ).astype("int8")

    result["is_bank_transfer"] = (
        result["payment_method"] == "bank_transfer"
    ).astype("int8")

    result["is_wallet"] = (
        result["payment_method"] == "wallet"
    ).astype("int8")

    result["is_online"] = (
        result["payment_method"] == "online"
    ).astype("int8")

    result["is_purchase"] = (
        result["transaction_type"] == "purchase"
    ).astype("int8")

    result["is_transfer"] = (
        result["transaction_type"] == "transfer"
    ).astype("int8")

    result["is_withdrawal"] = (
        result["transaction_type"] == "withdrawal"
    ).astype("int8")

    result["is_payment"] = (
        result["transaction_type"] == "payment"
    ).astype("int8")

    return result