from __future__ import annotations

from pathlib import Path

from fraud_intelligence.analysis.fraud_analysis import (
    amount_statistics_by_fraud,
    fraud_class_distribution,
    fraud_rate_by_category,
    fraud_scenario_distribution,
)

from fraud_intelligence.analysis.temporal_analysis import (
    daily_fraud_activity,
    fraud_rate_by_day_of_week,
    fraud_rate_by_hour,
    fraud_rate_by_month,
    hourly_fraud_concentration,
    velocity_windows,
)

from fraud_intelligence.analysis.entity_analysis import (
    entity_fraud_concentration,
    entity_fraud_summary,
    entity_type_overview,
    high_fraud_rate_entities,
    shared_entity_customer_counts,
    top_fraud_entities,
)

from fraud_intelligence.analysis.ring_analysis import (
    component_fraud_ring_rate,
    fraud_connected_components,
    fraud_ring_transaction_summary,
    shared_device_fraud_connections,
    shared_ip_fraud_connections,
)

from fraud_intelligence.analysis.leakage_audit import (
    run_leakage_audit,
)

from fraud_intelligence.analysis.eda import (
    cardinality_summary,
    categorical_summary,
    dataset_overview,
    fraud_summary,
    load_transactions,
    missing_value_summary,
    numeric_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRANSACTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "synthetic"
    / "transactions.parquet"
)


def main() -> None:
    transactions = load_transactions(
        TRANSACTIONS_PATH
    )

    print("\n" + "=" * 70)
    print("FRAUD INTELLIGENCE — DATASET PROFILE")
    print("=" * 70)

    print("\n--- Dataset Overview ---")
    for key, value in dataset_overview(
        transactions
    ).items():
        print(f"{key}: {value}")

    print("\n--- Schema ---")
    print(transactions.dtypes)

    print("\n--- Missing Values ---")
    print(missing_value_summary(transactions))

    print("\n--- Cardinality ---")
    print(cardinality_summary(transactions))

    print("\n--- Numeric Summary ---")
    print(numeric_summary(transactions))

    print("\n--- Fraud Summary ---")
    for key, value in fraud_summary(
        transactions
    ).items():
        print(f"{key}: {value}")

    print("\n--- Fraud Class Distribution ---")
    print(
        fraud_class_distribution(
            transactions
        ).to_string(index=False)
    )

    print("\n--- Fraud Rate by Payment Method ---")
    print(
        fraud_rate_by_category(
            transactions,
            "payment_method",
        ).to_string(index=False)
    )

    print("\n--- Fraud Rate by Transaction Type ---")
    print(
        fraud_rate_by_category(
            transactions,
            "transaction_type",
        ).to_string(index=False)
    )

    print("\n--- Amount Statistics by Fraud Class ---")
    print(
        amount_statistics_by_fraud(
            transactions
        ).to_string(index=False)
    )

    print("\n--- Fraud Scenario Distribution ---")
    print(
        fraud_scenario_distribution(
            transactions
        ).to_string(index=False)
    )

    print("\n--- Fraud Rate by Hour (UTC) ---")
    print(
        fraud_rate_by_hour(
            transactions
        ).to_string(index=False)
    )

    print("\n--- Fraud Rate by Day of Week ---")
    print(
        fraud_rate_by_day_of_week(
            transactions
        ).to_string(index=False)
    )

    print("\n--- Fraud Rate by Month ---")
    print(
        fraud_rate_by_month(
            transactions
        ).to_string(index=False)
    )

    print("\n--- Daily Fraud Activity ---")
    daily_activity = daily_fraud_activity(
        transactions
    )

    print(
        daily_activity.head(20)
        .to_string(index=False)
    )

    print(
        f"\nTotal days analyzed: "
        f"{len(daily_activity)}"
    )

    print("\n--- Hourly Fraud Concentration ---")
    print(
        hourly_fraud_concentration(
            transactions
        ).to_string(index=False)
    )

    print("\n--- Customer Velocity Windows ---")
    velocity = velocity_windows(
        transactions
    )

    print(
        velocity[
            [
                "within_1_minute",
                "within_5_minutes",
                "within_15_minutes",
            ]
        ].sum()
    )

    print("\n--- Velocity by Fraud Class ---")

    velocity_summary = (
        velocity.groupby("is_fraud")[
            [
                "within_1_minute",
                "within_5_minutes",
                "within_15_minutes",
            ]
        ]
        .mean()
        .mul(100)
        .round(4)
    )

    print(
        velocity_summary
    )

    print("\n--- Entity Type Overview ---")
    print(
        entity_type_overview(
            transactions
        ).to_string(index=False)
    )

    for entity_column in [
        "customer_id",
        "account_id",
        "card_id",
        "device_id",
        "ip_id",
        "merchant_id",
    ]:
        print(
            f"\n--- Top Fraud Entities: "
            f"{entity_column} ---"
        )

        print(
            top_fraud_entities(
                transactions,
                entity_column,
                top_n=10,
            ).to_string(index=False)
        )

        print(
            f"\n--- High Fraud Rate Entities: "
            f"{entity_column} ---"
        )

        print(
            high_fraud_rate_entities(
                transactions,
                entity_column,
                min_transactions=5,
                top_n=10,
            ).to_string(index=False)
        )

    for entity_column in [
        "device_id",
        "ip_id",
        "merchant_id",
    ]:
        print(
            f"\n--- Shared "
            f"{entity_column} ---"
        )

        print(
            shared_entity_customer_counts(
                transactions,
                entity_column,
            )
            .head(10)
            .to_string(index=False)
        )

    print("\n--- Entity Fraud Concentration ---")

    for entity_column in [
        "customer_id",
        "account_id",
        "card_id",
        "device_id",
        "ip_id",
        "merchant_id",
    ]:
        concentration = (
            entity_fraud_concentration(
                transactions,
                entity_column,
            )
        )

        top_10_contribution = (
            concentration
            .head(10)["fraud_contribution_pct"]
            .sum()
        )

        print(
            f"{entity_column}: "
            f"top_10_fraud_contribution="
            f"{top_10_contribution:.2f}%"
        )

    print("\n--- Explicit Fraud Ring Summary ---")
    print(
        fraud_ring_transaction_summary(
            transactions
        ).to_string(index=False)
    )

    print("\n--- Fraud Customer Connected Components ---")

    components = fraud_connected_components(
        transactions,
        min_customers=2,
    )

    print(
        f"Connected components with "
        f"2+ customers: {len(components)}"
    )

    print(
        components.head(20)
        .to_string(index=False)
    )

    print(
        "\n--- Fraud Ring Concentration "
        "Within Components ---"
    )

    component_rates = (
        component_fraud_ring_rate(
            transactions
        )
    )

    print(
        component_rates.head(20)
        .to_string(index=False)
    )

    print("\n--- Shared Device Fraud Connections ---")

    print(
        shared_device_fraud_connections(
            transactions
        )
        .head(20)
        .to_string(index=False)
    )

    print("\n--- Shared IP Fraud Connections ---")

    print(
        shared_ip_fraud_connections(
            transactions
        )
        .head(20)
        .to_string(index=False)
    )

    print("\n--- Leakage Audit ---")

    leakage = run_leakage_audit(
        transactions
    )

    print("\n[Direct Target Leakage]")
    print(
        leakage[
            "target_leakage"
        ].to_string(index=False)
    )

    print("\n[Identifier Audit]")
    print(
        leakage[
            "identifier_audit"
        ].to_string(index=False)
    )

    print("\n[Transaction Attribute Audit]")
    print(
        leakage[
            "transaction_columns"
        ].to_string(index=False)
    )

    print("\n[Temporal Audit]")

    for key, value in leakage[
        "temporal_audit"
    ].items():
        print(f"{key}: {value}")

    print("\n[Duplicate Audit]")

    for key, value in leakage[
        "duplicate_audit"
    ].items():
        print(f"{key}: {value}")

    print("\n[Feature Engineering Rules]")
    print(
        leakage[
            "feature_rules"
        ].to_string(index=False)
    )

    print("\n--- Categorical Distributions ---")

    categorical = categorical_summary(
        transactions
    )

    for column, summary in categorical.items():
        print(f"\n[{column}]")

        if len(summary) > 20:
            print(
                f"Showing top 20 of {len(summary)} unique values:"
            )

        print(
            summary.head(20).to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("EDA PROFILE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()