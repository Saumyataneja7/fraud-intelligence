from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_transactions(path: str | Path) -> pd.DataFrame:
    """
    Load the frozen transaction dataset.

    Parameters
    ----------
    path:
        Path to transactions.parquet.

    Returns
    -------
    pd.DataFrame
        Transaction dataset.
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: {dataset_path}"
        )

    transactions = pd.read_parquet(dataset_path)

    if transactions.empty:
        raise ValueError("Transaction dataset is empty.")

    return transactions


def dataset_overview(
    transactions: pd.DataFrame,
) -> dict[str, object]:
    """
    Generate high-level dataset statistics.
    """
    return {
        "rows": len(transactions),
        "columns": len(transactions.columns),
        "memory_mb": round(
            transactions.memory_usage(deep=True).sum()
            / (1024**2),
            2,
        ),
        "duplicate_rows": int(
            transactions.duplicated().sum()
        ),
        "duplicate_transaction_ids": int(
            transactions["transaction_id"]
            .duplicated()
            .sum()
        ),
    }


def missing_value_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return missing-value counts and percentages.
    """
    summary = pd.DataFrame(
        {
            "missing_count": transactions.isna().sum(),
            "missing_pct": (
                transactions.isna().mean() * 100
            ).round(4),
        }
    )

    return summary.sort_values(
        "missing_count",
        ascending=False,
    )


def cardinality_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return unique-value counts for every column.
    """
    summary = pd.DataFrame(
        {
            "dtype": transactions.dtypes.astype(str),
            "unique_count": transactions.nunique(
                dropna=True
            ),
            "missing_count": transactions.isna().sum(),
        }
    )

    return summary.sort_values(
        "unique_count",
        ascending=False,
    )


def numeric_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return descriptive statistics for numeric columns.
    """
    numeric_columns = transactions.select_dtypes(
        include="number"
    ).columns

    return transactions[numeric_columns].describe().T


def categorical_summary(
    transactions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Return frequency tables for categorical columns.
    """
    categorical_columns = transactions.select_dtypes(
        include=["object", "category"]
    ).columns

    summaries: dict[str, pd.DataFrame] = {}

    for column in categorical_columns:
        counts = (
            transactions[column]
            .value_counts(dropna=False)
            .rename_axis(column)
            .reset_index(name="count")
        )

        counts["percentage"] = (
            counts["count"]
            / len(transactions)
            * 100
        ).round(4)

        summaries[column] = counts

    return summaries


def fraud_summary(
    transactions: pd.DataFrame,
) -> dict[str, object]:
    """
    Return baseline fraud statistics.
    """
    fraud_count = int(
        transactions["is_fraud"].sum()
    )

    total_count = len(transactions)

    return {
        "total_transactions": total_count,
        "fraud_transactions": fraud_count,
        "non_fraud_transactions": (
            total_count - fraud_count
        ),
        "fraud_rate_pct": round(
            fraud_count / total_count * 100,
            4,
        ),
    }