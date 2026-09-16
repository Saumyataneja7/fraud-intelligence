from __future__ import annotations

import pandas as pd


def fraud_class_distribution(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return legitimate/fraud transaction counts
    and percentages.
    """
    result = (
        transactions["is_fraud"]
        .value_counts()
        .rename_axis("is_fraud")
        .reset_index(name="transaction_count")
    )

    result["percentage"] = (
        result["transaction_count"]
        / len(transactions)
        * 100
    ).round(4)

    result["label"] = result["is_fraud"].map(
        {
            0: "LEGITIMATE",
            1: "FRAUD",
        }
    )

    return result[
        [
            "label",
            "is_fraud",
            "transaction_count",
            "percentage",
        ]
    ]


def fraud_rate_by_category(
    transactions: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Calculate transaction count and fraud rate
    for each category.
    """
    grouped = (
        transactions.groupby(column)
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

    grouped["fraud_rate_pct"] = (
        grouped["fraud_count"]
        / grouped["transaction_count"]
        * 100
    ).round(4)

    return grouped.sort_values(
        "fraud_rate_pct",
        ascending=False,
    )


def amount_statistics_by_fraud(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare transaction amount statistics
    between legitimate and fraudulent transactions.
    """
    result = (
        transactions.groupby("is_fraud")["amount"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
    )

    result["label"] = result["is_fraud"].map(
        {
            0: "LEGITIMATE",
            1: "FRAUD",
        }
    )

    return result[
        [
            "label",
            "is_fraud",
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    ]


def fraud_scenario_distribution(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return fraud scenario counts and percentages,
    excluding legitimate transactions.
    """
    fraud = transactions[
        transactions["is_fraud"] == 1
    ]

    result = (
        fraud["fraud_scenario"]
        .value_counts()
        .rename_axis("fraud_scenario")
        .reset_index(name="fraud_count")
    )

    result["percentage"] = (
        result["fraud_count"]
        / len(fraud)
        * 100
    ).round(4)

    return result