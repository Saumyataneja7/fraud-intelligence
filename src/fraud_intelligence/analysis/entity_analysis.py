from __future__ import annotations

import pandas as pd


ENTITY_COLUMNS = [
    "customer_id",
    "account_id",
    "card_id",
    "device_id",
    "ip_id",
    "merchant_id",
]


def entity_fraud_summary(
    transactions: pd.DataFrame,
    entity_column: str,
) -> pd.DataFrame:
    """Calculate transaction and fraud statistics for an entity."""

    if entity_column not in ENTITY_COLUMNS:
        raise ValueError(
            f"Unsupported entity column: {entity_column}"
        )

    result = (
        transactions.groupby(entity_column)
        .agg(
            transaction_count=(
                "transaction_id",
                "count",
            ),
            fraud_count=(
                "is_fraud",
                "sum",
            ),
            total_amount=(
                "amount",
                "sum",
            ),
            fraud_amount=(
                "amount",
                lambda values: values[
                    transactions.loc[
                        values.index,
                        "is_fraud",
                    ].eq(1)
                ].sum(),
            ),
        )
        .reset_index()
    )

    result["fraud_rate_pct"] = (
        result["fraud_count"]
        / result["transaction_count"]
        * 100
    ).round(4)

    return result.sort_values(
        [
            "fraud_count",
            "fraud_rate_pct",
        ],
        ascending=False,
    )


def top_fraud_entities(
    transactions: pd.DataFrame,
    entity_column: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return entities ranked by fraud count."""

    summary = entity_fraud_summary(
        transactions,
        entity_column,
    )

    return summary.head(top_n)


def high_fraud_rate_entities(
    transactions: pd.DataFrame,
    entity_column: str,
    min_transactions: int = 5,
    top_n: int = 20,
) -> pd.DataFrame:
    """Find entities with unusually high fraud rates.

    A minimum transaction threshold is used to avoid ranking
    one-off entities with a 100% fraud rate.
    """

    summary = entity_fraud_summary(
        transactions,
        entity_column,
    )

    result = summary[
        summary["transaction_count"]
        >= min_transactions
    ].copy()

    return result.sort_values(
        [
            "fraud_rate_pct",
            "fraud_count",
        ],
        ascending=False,
    ).head(top_n)


def entity_fraud_concentration(
    transactions: pd.DataFrame,
    entity_column: str,
) -> pd.DataFrame:
    """Measure how fraud is distributed across entities."""

    summary = entity_fraud_summary(
        transactions,
        entity_column,
    )

    total_fraud = summary["fraud_count"].sum()

    if total_fraud == 0:
        summary["fraud_contribution_pct"] = 0.0
    else:
        summary["fraud_contribution_pct"] = (
            summary["fraud_count"]
            / total_fraud
            * 100
        ).round(4)

    return summary


def shared_entity_customer_counts(
    transactions: pd.DataFrame,
    entity_column: str,
) -> pd.DataFrame:
    """Count how many unique customers use each entity."""

    if entity_column not in [
        "device_id",
        "ip_id",
        "merchant_id",
    ]:
        raise ValueError(
            "Shared-entity analysis supports "
            "device_id, ip_id, and merchant_id."
        )

    result = (
        transactions.groupby(entity_column)
        .agg(
            unique_customer_count=(
                "customer_id",
                "nunique",
            ),
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

    result["fraud_rate_pct"] = (
        result["fraud_count"]
        / result["transaction_count"]
        * 100
    ).round(4)

    return result.sort_values(
        [
            "unique_customer_count",
            "fraud_count",
        ],
        ascending=False,
    )


def entity_type_overview(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Create a high-level summary for every entity type."""

    rows = []

    for entity_column in ENTITY_COLUMNS:
        summary = entity_fraud_summary(
            transactions,
            entity_column,
        )

        rows.append(
            {
                "entity_type": entity_column,
                "unique_entities": summary[
                    entity_column
                ].nunique(),
                "entities_with_fraud": (
                    summary["fraud_count"] > 0
                ).sum(),
                "total_transactions": summary[
                    "transaction_count"
                ].sum(),
                "total_fraud": summary[
                    "fraud_count"
                ].sum(),
            }
        )

    return pd.DataFrame(rows)