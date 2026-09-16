from __future__ import annotations

import pandas as pd
import networkx as nx


def fraud_ring_transaction_summary(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize transactions belonging to the explicit FRAUD_RING scenario."""

    fraud_ring = transactions[
        transactions["fraud_scenario"] == "FRAUD_RING"
    ].copy()

    if fraud_ring.empty:
        return pd.DataFrame(
            columns=[
                "transaction_count",
                "customer_count",
                "account_count",
                "card_count",
                "device_count",
                "ip_count",
                "merchant_count",
                "total_amount",
                "avg_amount",
            ]
        )

    return pd.DataFrame(
        {
            "transaction_count": [
                len(fraud_ring)
            ],
            "customer_count": [
                fraud_ring["customer_id"].nunique()
            ],
            "account_count": [
                fraud_ring["account_id"].nunique()
            ],
            "card_count": [
                fraud_ring["card_id"].nunique()
            ],
            "device_count": [
                fraud_ring["device_id"].nunique()
            ],
            "ip_count": [
                fraud_ring["ip_id"].nunique()
            ],
            "merchant_count": [
                fraud_ring["merchant_id"].nunique()
            ],
            "total_amount": [
                fraud_ring["amount"].sum()
            ],
            "avg_amount": [
                fraud_ring["amount"].mean()
            ],
        }
    )


def build_fraud_customer_graph(
    transactions: pd.DataFrame,
) -> nx.Graph:
    """Build a customer graph using shared device/IP relationships.

    Only fraud transactions are used.

    This is an EDA graph, not the production heterogeneous graph.
    """

    fraud = transactions[
        transactions["is_fraud"] == 1
    ].copy()

    graph = nx.Graph()

    customers = fraud["customer_id"].unique()

    graph.add_nodes_from(customers)

    for entity_column in [
        "device_id",
        "ip_id",
    ]:
        grouped = fraud.groupby(entity_column)[
            "customer_id"
        ].unique()

        for customer_ids in grouped:
            customer_ids = list(customer_ids)

            for i in range(len(customer_ids)):
                for j in range(i + 1, len(customer_ids)):
                    customer_a = customer_ids[i]
                    customer_b = customer_ids[j]

                    if graph.has_edge(
                        customer_a,
                        customer_b,
                    ):
                        graph[customer_a][customer_b][
                            "shared_entity_count"
                        ] += 1

                    else:
                        graph.add_edge(
                            customer_a,
                            customer_b,
                            shared_entity_count=1,
                        )

    return graph


def fraud_connected_components(
    transactions: pd.DataFrame,
    min_customers: int = 2,
) -> pd.DataFrame:
    """Find connected customer clusters among fraud transactions."""

    graph = build_fraud_customer_graph(
        transactions
    )

    components = []

    for component_id, nodes in enumerate(
        nx.connected_components(graph),
        start=1,
    ):
        if len(nodes) < min_customers:
            continue

        component_customers = set(nodes)

        component_transactions = transactions[
            transactions["is_fraud"].eq(1)
            & transactions["customer_id"].isin(
                component_customers
            )
        ]

        components.append(
            {
                "component_id": component_id,
                "customer_count": len(
                    component_customers
                ),
                "transaction_count": len(
                    component_transactions
                ),
                "fraud_count": int(
                    component_transactions[
                        "is_fraud"
                    ].sum()
                ),
                "device_count": component_transactions[
                    "device_id"
                ].nunique(),
                "ip_count": component_transactions[
                    "ip_id"
                ].nunique(),
                "merchant_count": component_transactions[
                    "merchant_id"
                ].nunique(),
                "total_amount": component_transactions[
                    "amount"
                ].sum(),
                "fraud_ring_transactions": int(
                    (
                        component_transactions[
                            "fraud_scenario"
                        ]
                        == "FRAUD_RING"
                    ).sum()
                ),
            }
        )

    result = pd.DataFrame(components)

    if result.empty:
        return pd.DataFrame(
            columns=[
                "component_id",
                "customer_count",
                "transaction_count",
                "fraud_count",
                "device_count",
                "ip_count",
                "merchant_count",
                "total_amount",
                "fraud_ring_transactions",
            ]
        )

    return result.sort_values(
        [
            "fraud_ring_transactions",
            "fraud_count",
            "customer_count",
        ],
        ascending=False,
    ).reset_index(drop=True)


def shared_device_fraud_connections(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Find devices connecting multiple customers through fraud transactions."""

    fraud = transactions[
        transactions["is_fraud"] == 1
    ]

    result = (
        fraud.groupby("device_id")
        .agg(
            customer_count=(
                "customer_id",
                "nunique",
            ),
            fraud_transaction_count=(
                "transaction_id",
                "count",
            ),
            fraud_ring_transaction_count=(
                "fraud_scenario",
                lambda values: (
                    values == "FRAUD_RING"
                ).sum(),
            ),
        )
        .reset_index()
    )

    return result[
        result["customer_count"] >= 2
    ].sort_values(
        [
            "fraud_ring_transaction_count",
            "customer_count",
            "fraud_transaction_count",
        ],
        ascending=False,
    )


def shared_ip_fraud_connections(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Find IPs connecting multiple customers through fraud transactions."""

    fraud = transactions[
        transactions["is_fraud"] == 1
    ]

    result = (
        fraud.groupby("ip_id")
        .agg(
            customer_count=(
                "customer_id",
                "nunique",
            ),
            fraud_transaction_count=(
                "transaction_id",
                "count",
            ),
            fraud_ring_transaction_count=(
                "fraud_scenario",
                lambda values: (
                    values == "FRAUD_RING"
                ).sum(),
            ),
        )
        .reset_index()
    )

    return result[
        result["customer_count"] >= 2
    ].sort_values(
        [
            "fraud_ring_transaction_count",
            "customer_count",
            "fraud_transaction_count",
        ],
        ascending=False,
    )


def component_fraud_ring_rate(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Measure the proportion of fraud in connected components
    that comes from the explicit FRAUD_RING scenario.
    """

    components = fraud_connected_components(
        transactions
    )

    if components.empty:
        return components

    components = components.copy()

    components["fraud_ring_pct"] = (
        components["fraud_ring_transactions"]
        / components["fraud_count"]
        * 100
    ).round(4)

    return components