from __future__ import annotations

import pandas as pd


def validate_transactions(
    transactions: pd.DataFrame,
) -> dict:
    """Validate generated transaction data."""

    duplicate_ids = int(
        transactions["transaction_id"].duplicated().sum()
    )

    null_counts = (
        transactions.isna()
        .sum()
        .loc[lambda values: values > 0]
        .to_dict()
    )

    invalid_amounts = int(
        (transactions["amount"] <= 0).sum()
    )

    invalid_labels = int(
        (~transactions["is_fraud"].isin([0, 1])).sum()
    )

    return {
        "row_count": len(transactions),
        "duplicate_transaction_ids": duplicate_ids,
        "null_columns": null_counts,
        "invalid_amounts": invalid_amounts,
        "invalid_fraud_labels": invalid_labels,
        "passed": (
            duplicate_ids == 0
            and not null_counts
            and invalid_amounts == 0
            and invalid_labels == 0
        ),
    }

def validate_relationships(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    cards: pd.DataFrame,
    devices: pd.DataFrame,
    ips: pd.DataFrame,
    merchants: pd.DataFrame,
) -> dict:
    """Validate referential integrity across generated entities."""

    customer_ids = set(customers["customer_id"])
    account_ids = set(accounts["account_id"])
    card_ids = set(cards["card_id"])
    device_ids = set(devices["device_id"])
    ip_ids = set(ips["ip_id"])
    merchant_ids = set(merchants["merchant_id"])

    account_to_customer = dict(
        zip(
            accounts["account_id"],
            accounts["customer_id"],
        )
    )

    card_to_account = dict(
        zip(
            cards["card_id"],
            cards["account_id"],
        )
    )

    invalid_customers = int(
        (~transactions["customer_id"].isin(customer_ids)).sum()
    )

    invalid_accounts = int(
        (~transactions["account_id"].isin(account_ids)).sum()
    )

    invalid_cards = int(
        (~transactions["card_id"].isin(card_ids)).sum()
    )

    invalid_devices = int(
        (~transactions["device_id"].isin(device_ids)).sum()
    )

    invalid_ips = int(
        (~transactions["ip_id"].isin(ip_ids)).sum()
    )

    invalid_merchants = int(
        (~transactions["merchant_id"].isin(merchant_ids)).sum()
    )

    account_customer_mismatches = 0

    for row in transactions[
        ["customer_id", "account_id"]
    ].itertuples(index=False):
        customer_id, account_id = row

        if account_to_customer.get(account_id) != customer_id:
            account_customer_mismatches += 1

    card_account_mismatches = 0

    for row in transactions[
        ["account_id", "card_id"]
    ].itertuples(index=False):
        account_id, card_id = row

        if card_to_account.get(card_id) != account_id:
            card_account_mismatches += 1

    passed = all(
        value == 0
        for value in [
            invalid_customers,
            invalid_accounts,
            invalid_cards,
            invalid_devices,
            invalid_ips,
            invalid_merchants,
            account_customer_mismatches,
            card_account_mismatches,
        ]
    )

    return {
        "invalid_customers": invalid_customers,
        "invalid_accounts": invalid_accounts,
        "invalid_cards": invalid_cards,
        "invalid_devices": invalid_devices,
        "invalid_ips": invalid_ips,
        "invalid_merchants": invalid_merchants,
        "account_customer_mismatches": account_customer_mismatches,
        "card_account_mismatches": card_account_mismatches,
        "passed": passed,
    }