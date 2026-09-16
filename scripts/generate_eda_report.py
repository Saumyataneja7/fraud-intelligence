from __future__ import annotations
from textwrap import dedent

from pathlib import Path

from fraud_intelligence.analysis.eda import (
    cardinality_summary,
    dataset_overview,
    load_transactions,
    missing_value_summary,
    numeric_summary,
)
from fraud_intelligence.analysis.entity_analysis import (
    entity_fraud_concentration,
    entity_type_overview,
    high_fraud_rate_entities,
    shared_entity_customer_counts,
    top_fraud_entities,
)
from fraud_intelligence.analysis.fraud_analysis import (
    amount_statistics_by_fraud,
    fraud_class_distribution,
    fraud_rate_by_category,
    fraud_scenario_distribution,
)
from fraud_intelligence.analysis.leakage_audit import run_leakage_audit
from fraud_intelligence.analysis.ring_analysis import (
    build_fraud_customer_graph,
    component_fraud_ring_rate,
    fraud_connected_components,
    fraud_ring_transaction_summary,
    shared_device_fraud_connections,
    shared_ip_fraud_connections,
)
from fraud_intelligence.analysis.temporal_analysis import (
    daily_fraud_activity,
    fraud_rate_by_day_of_week,
    fraud_rate_by_hour,
    fraud_rate_by_month,
    hourly_fraud_concentration,
    velocity_windows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRANSACTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "synthetic"
    / "transactions.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "eda"
    / "EDA_REPORT.md"
)


def dataframe_to_markdown(data) -> str:
    """Convert a DataFrame or dictionary to Markdown."""
    if data is None:
        return "_No results._"

    if isinstance(data, dict):
        if not data:
            return "_No results._"

        lines = []

        for key, value in data.items():
            lines.append(
                f"- **{key}:** {format_value(value)}"
            )

        return "\n".join(lines)

    if data.empty:
        return "_No results._"

    return data.to_markdown(index=False)


def format_value(value) -> str:
    """Format report values safely."""
    if hasattr(value, "item"):
        return value.item()

    return value


def format_dict(data: dict) -> str:
    """Convert a dictionary to Markdown bullets."""
    lines = []

    for key, value in data.items():
        lines.append(
            f"- **{key}:** {format_value(value)}"
        )

    return "\n".join(lines)


def generate_report() -> None:
    """Generate the Phase 3.7 EDA report."""

    transactions = load_transactions(
        TRANSACTION_PATH
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # 1. Dataset profiling
    # ------------------------------------------------------------------

    overview = dataset_overview(
        transactions
    )

    missing = missing_value_summary(
        transactions
    )

    cardinality = cardinality_summary(
        transactions
    )

    numeric = numeric_summary(
        transactions
    )

    # ------------------------------------------------------------------
    # 2. Fraud analysis
    # ------------------------------------------------------------------

    fraud_classes = fraud_class_distribution(
        transactions
    )

    payment_fraud = fraud_rate_by_category(
        transactions,
        "payment_method",
    )

    transaction_type_fraud = fraud_rate_by_category(
        transactions,
        "transaction_type",
    )

    amount_fraud = amount_statistics_by_fraud(
        transactions
    )

    scenarios = fraud_scenario_distribution(
        transactions
    )

    # ------------------------------------------------------------------
    # 3. Temporal analysis
    # ------------------------------------------------------------------

    hourly = fraud_rate_by_hour(
        transactions
    )

    daily = fraud_rate_by_day_of_week(
        transactions
    )

    monthly = fraud_rate_by_month(
        transactions
    )

    daily_activity = daily_fraud_activity(
        transactions
    )

    hourly_concentration = hourly_fraud_concentration(
        transactions
    )

    velocity = velocity_windows(
        transactions
    )

    # ------------------------------------------------------------------
    # 4. Entity analysis
    # ------------------------------------------------------------------

    entities = entity_type_overview(
        transactions
    )

    customer_top = top_fraud_entities(
        transactions,
        "customer_id",
        top_n=10,
    )

    device_top = top_fraud_entities(
        transactions,
        "device_id",
        top_n=10,
    )

    ip_top = top_fraud_entities(
        transactions,
        "ip_id",
        top_n=10,
    )

    merchant_top = top_fraud_entities(
        transactions,
        "merchant_id",
        top_n=10,
    )

    customer_high_rate = high_fraud_rate_entities(
        transactions,
        "customer_id",
        min_transactions=10,
    )

    device_high_rate = high_fraud_rate_entities(
        transactions,
        "device_id",
        min_transactions=2,
    )

    ip_high_rate = high_fraud_rate_entities(
        transactions,
        "ip_id",
        min_transactions=2,
    )

    merchant_high_rate = high_fraud_rate_entities(
        transactions,
        "merchant_id",
        min_transactions=10,
    )

    concentration = {}

    for entity_column in [
        "customer_id",
        "account_id",
        "card_id",
        "device_id",
        "ip_id",
        "merchant_id",
    ]:
        concentration[entity_column] = entity_fraud_concentration(
            transactions,
            entity_column,
        )

    shared_devices = shared_entity_customer_counts(
        transactions,
        "device_id",
    )

    shared_ips = shared_entity_customer_counts(
        transactions,
        "ip_id",
    )

    # ------------------------------------------------------------------
    # 5. Fraud-ring analysis
    # ------------------------------------------------------------------

    ring_summary = fraud_ring_transaction_summary(
        transactions
    )

    fraud_graph = build_fraud_customer_graph(
        transactions
    )

    fraud_connected_components(
        transactions
    )

    component_rates = component_fraud_ring_rate(
        transactions,
    )

    shared_device_connections = (
        shared_device_fraud_connections(
            transactions
        )
    )

    shared_ip_connections = (
        shared_ip_fraud_connections(
            transactions
        )
    )

    # ------------------------------------------------------------------
    # 6. Leakage audit
    # ------------------------------------------------------------------

    leakage = run_leakage_audit(
        transactions
    )

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------

    sections = []

    sections.append(
        "# Fraud Intelligence — Exploratory Data Analysis"
    )

    sections.append(
            """
        **Dataset:** `fraud-intelligence-synthetic`

        **Dataset version:** `1.0.0`

        **Analysis phase:** Phase 3 — Exploratory Data Analysis

        **Random seed:** `42`

        **Dataset type:** Synthetic

        ---

        ## 1. Executive Summary

        This report documents the exploratory analysis performed on the frozen
        synthetic transaction dataset used by the Fraud Intelligence project.

        The dataset contains 100,000 transactions covering the 2025 calendar year,
        with a deliberately imbalanced fraud rate of 1%.

        The analysis identifies several important signals:

        - Fraudulent transactions have substantially higher transaction amounts.
        - Transfer transactions have the highest observed fraud rate.
        - Wallet transactions have the highest observed fraud rate among payment
          methods.
        - Fraud occurs throughout the day rather than being restricted to one
          specific period.
        - Fraudulent transactions are strongly concentrated around certain devices
          and IP addresses.
        - Explicit fraud-ring transactions demonstrate multi-entity relationships.
        - Shared infrastructure does not automatically imply fraud.
        - Historical and velocity features require strict point-in-time handling.
        - Label-derived fraud-ring information must remain EDA-only.
        """
    )

    sections.append(
        "## 2. Dataset Profile\n\n"
        + format_dict(overview)
    )

    sections.append(
        "### Transaction Cardinality\n\n"
        + dataframe_to_markdown(cardinality)
    )

    sections.append(
        """
        ### Transaction Schema

        | Column | Role |
        |---|---|
        | `transaction_id` | Transaction identifier |
        | `timestamp` | Transaction event time |
        | `customer_id` | Customer entity |
        | `account_id` | Account entity |
        | `card_id` | Card entity |
        | `merchant_id` | Merchant entity |
        | `device_id` | Device entity |
        | `ip_id` | IP entity |
        | `amount` | Transaction amount |
        | `currency` | Transaction currency |
        | `payment_method` | Payment mechanism |
        | `transaction_type` | Transaction category |
        | `is_fraud` | Binary target |
        | `fraud_scenario` | Synthetic fraud scenario |
        """
    )

    sections.append(
        "## 3. Data Quality\n\n"
        "### Missing Values\n\n"
        + dataframe_to_markdown(missing)
        + """

        The transaction dataset contains no missing values.

        ### Numeric Summary

        """
        + dataframe_to_markdown(numeric)
        + """

        ### Duplicate Checks

        | Check | Result |
        |---|---:|
        | Duplicate rows | 0 |
        | Duplicate transaction IDs | 0 |

        ### Timestamp Quality

        The timestamps are timezone-aware UTC timestamps.

        The transaction table is not physically sorted by timestamp. This is not
        itself a data-quality failure because chronological ordering will be
        explicitly enforced when constructing temporal features.

        The dataset contains 277 records sharing timestamps with other records.
        Because no explicit event ordering exists for those records, Phase 4 will
        use strictly earlier timestamps for historical features.
        """
    )

    sections.append(
        "## 4. Fraud Distribution\n\n"
        "### Class Distribution\n\n"
        + dataframe_to_markdown(fraud_classes)
        + """

        The dataset contains 99,000 legitimate transactions and 1,000 fraudulent
        transactions, corresponding to a 1% fraud rate.

        This creates an approximately 99:1 class imbalance.

        Accuracy should therefore not be treated as the primary evaluation metric.
        """
    )

    sections.append(
        "## 5. Fraud Scenario Distribution\n\n"
        + dataframe_to_markdown(scenarios)
        + """

        The fraud scenarios represent multiple fraud mechanisms, including account
        takeover, shared infrastructure, merchant abuse, velocity attacks,
        geo-anomaly behavior, and coordinated fraud rings.

        The scenario labels are useful for analysis but must not be used as
        predictive features.
        """
    )

    sections.append(
        "## 6. Transaction Amount Analysis\n\n"
        + dataframe_to_markdown(amount_fraud)
        + """

        Fraudulent transactions have substantially higher transaction amounts than
        legitimate transactions.

        The amount distribution is also strongly right-skewed, making transformed
        and historical amount features promising candidates for Phase 4.
        """
    )

    sections.append(
        "## 7. Fraud by Payment Method\n\n"
        + dataframe_to_markdown(payment_fraud)
    )

    sections.append(
        "## 8. Fraud by Transaction Type\n\n"
        + dataframe_to_markdown(transaction_type_fraud)
        + """

        Transfer transactions have the highest observed fraud rate.

        This relationship is a property of the synthetic generation process and
        must not be interpreted as a real-world financial fraud statistic.
        """
    )

    sections.append(
        "## 9. Temporal Analysis\n\n"
        "### Fraud by Hour\n\n"
        + dataframe_to_markdown(hourly)
        + "\n\n### Fraud by Day of Week\n\n"
        + dataframe_to_markdown(daily)
        + "\n\n### Fraud by Month\n\n"
        + dataframe_to_markdown(monthly)
        + """

        Fraud activity is distributed across the day and across the year.

        Temporal features may therefore be useful, but seasonal observations from
        a single synthetic year should not be treated as stable real-world trends.
        """
    )

    sections.append(
        "## 10. Velocity Analysis\n\n"
        + dataframe_to_markdown(velocity)
        + """

        Short-window transaction bursts are substantially more common among
        fraudulent transactions.

        Velocity features are therefore expected to be important in Phase 4.

        These features must only use transactions with timestamps strictly earlier
        than the transaction being scored.
        """
    )

    sections.append(
        "## 11. Entity-Level Analysis\n\n"
        + dataframe_to_markdown(entities)
        + """

        Fraud is not uniformly distributed across customers, accounts, cards,
        devices, IP addresses, and merchants.

        This supports the use of historical entity-level features and graph-based
        representations.
        """
    )

    concentration_sections = [
    "## 12. Fraud Concentration",
    ]

    for entity_column, result in concentration.items():
        concentration_sections.append(
            f"### {entity_column}\n\n"
            + dataframe_to_markdown(result)
        )

    concentration_sections.append(
        """
        Devices and IP addresses show particularly strong fraud concentration.

        However, concentration alone is not sufficient evidence of fraud because
        legitimate users can share infrastructure.
        """
    )

    sections.append(
        "\n\n".join(concentration_sections)
    )

    sections.append(
        "## 13. Shared Infrastructure\n\n"
        "### Shared Devices\n\n"
        + dataframe_to_markdown(shared_devices.head(20))
        + "\n\n### Shared IP Addresses\n\n"
        + dataframe_to_markdown(shared_ips.head(20))
    )

    sections.append(
        "## 14. Fraud-Ring Exploration\n\n"
        + dataframe_to_markdown(ring_summary)
        + """

        The explicit fraud-ring scenario connects multiple customers through
        shared devices, IP addresses, and merchants.

        The fraud-only customer graph contains multiple connected components with
        strong concentrations of fraud-ring transactions.

        However, connected infrastructure is not automatically equivalent to a
        fraud ring. Some connected components contain fraudulent transactions
        without explicit FRAUD_RING labels.

        This distinction is important for the future graph model.
        """
    )

    sections.append(
        "### Fraud-Ring Graph Components\n\n"
        + dataframe_to_markdown(component_rates)
    )

    sections.append(
        "### Shared Device Fraud Connections\n\n"
        + dataframe_to_markdown(
            shared_device_connections.head(20)
        )
    )

    sections.append(
        "### Shared IP Fraud Connections\n\n"
        + dataframe_to_markdown(
            shared_ip_connections.head(20)
        )
    )

    sections.append(
        "## 15. High Fraud-Rate Entities\n\n"
        "### Customers\n\n"
        + dataframe_to_markdown(
            customer_high_rate.head(20)
        )
        + "\n\n### Devices\n\n"
        + dataframe_to_markdown(
            device_high_rate.head(20)
        )
        + "\n\n### IP Addresses\n\n"
        + dataframe_to_markdown(
            ip_high_rate.head(20)
        )
        + "\n\n### Merchants\n\n"
        + dataframe_to_markdown(
            merchant_high_rate.head(20)
        )
    )

    sections.append(
        "## 16. Top Fraud-Associated Entities\n\n"
        "### Customers\n\n"
        + dataframe_to_markdown(customer_top)
        + "\n\n### Devices\n\n"
        + dataframe_to_markdown(device_top)
        + "\n\n### IP Addresses\n\n"
        + dataframe_to_markdown(ip_top)
        + "\n\n### Merchants\n\n"
        + dataframe_to_markdown(merchant_top)
    )

    sections.append(
        """
        ## 17. Leakage Audit

        The leakage audit identifies the following categories.

        ### Direct Target Leakage

        `is_fraud` and `fraud_scenario` are unsafe as model features.

        ### Identifiers

        Raw transaction, customer, account, card, merchant, device, and IP
        identifiers are conditional.

        They may participate in relational or historical feature construction, but
        must not be treated as ordinary numeric predictors.

        ### Temporal Leakage

        Historical features must use transactions strictly earlier than the
        transaction being scored.

        Transactions occurring at the same timestamp or in the future must not
        contribute unless an explicit event-ordering mechanism is introduced.

        ### Fraud-Ring Leakage

        Fraud-ring membership discovered using fraud labels is an EDA-only
        construct and cannot be used as a predictive feature.
        """
    )

    sections.append(
        "## 18. Feature Engineering Rules\n\n"
        + dataframe_to_markdown(
            leakage
        )
    )

    sections.append(
        """
        ## 19. Key Findings

        ### Finding 1 — Severe class imbalance

        Fraud represents only 1% of transactions.

        **Implication:** Phase 5 must emphasize precision, recall, F1, PR-AUC,
        ROC-AUC, Precision@K, Recall@K, false positives, and false negatives.

        ### Finding 2 — Fraudulent amounts are substantially larger

        Fraudulent transactions have significantly higher mean and median amounts.

        **Implication:** amount transformations and historical amount deviation
        features should be investigated.

        ### Finding 3 — Velocity is a strong signal

        Fraudulent transactions occur disproportionately in short temporal
        windows.

        **Implication:** point-in-time velocity features should be a major part of
        Phase 4.

        ### Finding 4 — Fraud concentrates around relational entities

        Certain devices and IP addresses are strongly associated with fraud.

        **Implication:** entity-history and graph features are justified.

        ### Finding 5 — Fraud rings have multi-entity structure

        The synthetic FRAUD_RING scenario connects customers through shared
        devices, IPs, and merchants.

        **Implication:** graph-based modeling can capture relationships that a
        transaction-only model may miss.

        ### Finding 6 — Leakage prevention is fundamental

        The dataset contains target labels, identifiers, temporal relationships,
        and same-timestamp transactions.

        **Implication:** feature engineering must be strictly point-in-time.
        """
    )

    sections.append(
        """
        ## 20. Phase 4 Requirements

        Phase 4 should construct a leakage-safe feature matrix containing:

        ### Transaction Features

        - amount
        - log amount
        - currency
        - payment method
        - transaction type
        - hour
        - day of week
        - month
        - weekend indicator

        ### Customer Historical Features

        - historical transaction count
        - historical total amount
        - historical average amount
        - historical median amount
        - historical maximum amount
        - historical unique merchants
        - historical unique devices
        - historical unique IP addresses

        ### Velocity Features

        - transactions in previous 1 minute
        - transactions in previous 5 minutes
        - transactions in previous 15 minutes
        - historical amount in rolling windows
        - unique merchants in rolling windows
        - unique devices in rolling windows
        - unique IPs in rolling windows

        ### Entity History

        - account transaction count
        - card transaction count
        - merchant transaction count
        - device transaction count
        - IP transaction count

        ### Novelty Features

        - new device indicator
        - new IP indicator
        - new merchant indicator
        - new card indicator

        ### Behavioral Deviation

        - amount versus customer historical behavior
        - amount versus merchant historical behavior
        - merchant frequency
        - device frequency
        - IP frequency
        - activity-time deviation

        All historical features must use strictly earlier transactions.
        """
    )

    sections.append(
        """
        ## 21. Limitations

        This dataset is intentionally synthetic.

        The observed patterns reflect the generation logic and should not be
        interpreted as empirical evidence about real-world financial fraud.

        Important limitations include:

        1. Fraud scenarios were explicitly injected.
        2. The fraud rate is intentionally fixed at 1%.
        3. Only one synthetic year is currently available.
        4. Currency is currently USD only.
        5. Some relational patterns are deliberately strong.
        6. Fraud-ring behavior is intentionally represented.
        7. Same-timestamp transactions do not have an explicit event order.
        8. Results should be interpreted as engineering validation rather than
        real-world financial-crime research.
        """
    )

    sections.append(
        """
        ## 22. Phase 3 Conclusion

        Phase 3 establishes that the synthetic transaction dataset is suitable for
        the next stage of the Fraud Intelligence pipeline.

        The strongest signals identified are:

        ```text
        Transaction behavior
                ↓
        Temporal velocity
                ↓
        Entity behavior
                ↓
        Shared device / IP relationships
                ↓
        Graph structure

        The central design decision is to build a leakage-safe feature layer that
        combines transaction behavior, temporal behavior, historical entity
        behavior, and relational signals.
        """
    )

    print("\n=== SECTIONS GENERATED ===")

    for i, section in enumerate(sections, start=1):
        first_heading = next(
            (
                line.strip()
                for line in section.splitlines()
                if line.strip().startswith("## ")
            ),
            "(no heading)",
        )

        print(i, first_heading)
    
    report = "\n\n---\n\n".join(
        dedent(section).strip()
        for section in sections
    )

    OUTPUT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"EDA report written to: {OUTPUT_PATH}"
    )

if __name__ == "__main__":
    generate_report()