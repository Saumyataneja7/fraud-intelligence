from __future__ import annotations

import argparse
from pathlib import Path

from fraud_intelligence.data.generator import (
    GeneratorConfig,
    SyntheticFraudGenerator,
)
from fraud_intelligence.data.validation import validate_transactions
from fraud_intelligence.utils.logging import get_logger

from fraud_intelligence.data.validation import (
    validate_relationships,
    validate_transactions,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate synthetic fraud intelligence data."
    )

    parser.add_argument(
        "--transactions",
        type=int,
        default=100_000,
        help="Number of transactions to generate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.01,
        help="Target fraud rate.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/synthetic",
        help="Directory where generated data will be stored.",
    )

    return parser.parse_args()


def main() -> None:
    """Generate and persist the synthetic dataset."""

    args = parse_args()

    if args.transactions <= 0:
        raise ValueError(
            "--transactions must be greater than zero."
        )

    if not 0 <= args.fraud_rate <= 1:
        raise ValueError(
            "--fraud-rate must be between 0 and 1."
        )

    config = GeneratorConfig(
        transactions=args.transactions,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )

    logger.info(
        "Starting synthetic fraud-data generation: "
        "transactions=%s seed=%s fraud_rate=%s",
        config.transactions,
        config.seed,
        config.fraud_rate,
    )

    generator = SyntheticFraudGenerator(config)

    # --------------------------------------------------------------
    # 1. Generate entities
    # --------------------------------------------------------------

    logger.info("Generating customers...")
    customers = generator.generate_customers()

    logger.info("Generating accounts...")
    accounts = generator.generate_accounts(
        customers
    )

    logger.info("Generating cards...")
    cards = generator.generate_cards(
        accounts
    )

    logger.info("Generating merchants...")
    merchants = generator.generate_merchants()

    logger.info("Generating devices...")
    devices = generator.generate_devices()

    logger.info("Generating IP addresses...")
    ips = generator.generate_ips()

    # --------------------------------------------------------------
    # 2. Generate relationships
    # --------------------------------------------------------------

    logger.info("Generating customer-device relationships...")
    customer_devices = generator.generate_customer_devices(
        customers,
        devices,
    )

    logger.info("Generating customer-IP relationships...")
    customer_ips = generator.generate_customer_ips(
        customers,
        ips,
    )

    logger.info("Generating customer-merchant relationships...")
    customer_merchants = generator.generate_customer_merchants(
        customers,
        merchants,
    )

    # --------------------------------------------------------------
    # 3. Generate normal transactions
    # --------------------------------------------------------------

    logger.info("Generating baseline transactions...")

    transactions = generator.generate_transactions(
        customers=customers,
        accounts=accounts,
        cards=cards,
        merchants=merchants,
        devices=devices,
        ips=ips,
        customer_devices=customer_devices,
        customer_ips=customer_ips,
        customer_merchants=customer_merchants,
    )

    # --------------------------------------------------------------
    # 4. Inject fraud
    # --------------------------------------------------------------

    logger.info("Injecting fraud scenarios...")

    transactions = generator.inject_fraud(
    transactions=transactions,
    customers=customers,
    accounts=accounts,
    cards=cards,
    devices=devices,
    ips=ips,
    merchants=merchants,
    )

    # --------------------------------------------------------------
    # 5. Validate
    # --------------------------------------------------------------

    logger.info("Validating transactions...")

    validation_result = validate_transactions(
        transactions
    )

    relationship_result = validate_relationships(
    transactions=transactions,
    customers=customers,
    accounts=accounts,
    cards=cards,
    devices=devices,
    ips=ips,
    merchants=merchants,
    )

    print("\n=== Relationship Validation ===")

    for key, value in relationship_result.items():
        print(f"{key}: {value}")

    print("===============================\n")

    if not relationship_result["passed"]:
        raise RuntimeError(
        "Generated relationship data failed validation."
    )

    logger.info(
        "Validation result: %s",
        validation_result,
    )

    if not validation_result["passed"]:
        print("\n=== Validation Failure ===")
    
        for key, value in validation_result.items():
            print(f"{key}: {value}")
        print("==========================\n")

        raise RuntimeError(
            "Generated transaction data failed validation."
        )

    # --------------------------------------------------------------
    # 6. Create output directory
    # --------------------------------------------------------------

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # 7. Write entity tables
    # --------------------------------------------------------------

    customers.to_parquet(
        output_dir / "customers.parquet",
        index=False,
    )

    accounts.to_parquet(
        output_dir / "accounts.parquet",
        index=False,
    )

    cards.to_parquet(
        output_dir / "cards.parquet",
        index=False,
    )

    merchants.to_parquet(
        output_dir / "merchants.parquet",
        index=False,
    )

    devices.to_parquet(
        output_dir / "devices.parquet",
        index=False,
    )

    ips.to_parquet(
        output_dir / "ips.parquet",
        index=False,
    )

    # --------------------------------------------------------------
    # 8. Write relationship tables
    # --------------------------------------------------------------

    customer_devices.to_parquet(
        output_dir / "customer_devices.parquet",
        index=False,
    )

    customer_ips.to_parquet(
        output_dir / "customer_ips.parquet",
        index=False,
    )

    customer_merchants.to_parquet(
        output_dir / "customer_merchants.parquet",
        index=False,
    )

    # --------------------------------------------------------------
    # 9. Write transactions
    # --------------------------------------------------------------

    transactions.to_parquet(
        output_dir / "transactions.parquet",
        index=False,
    )

    logger.info(
        "Synthetic dataset successfully written to %s",
        output_dir,
    )

    # --------------------------------------------------------------
    # 10. Print summary
    # --------------------------------------------------------------

    fraud_count = int(
        transactions["is_fraud"].sum()
    )

    actual_fraud_rate = (
        fraud_count / len(transactions)
        if len(transactions) > 0
        else 0
    )

    print("\n=== Synthetic Fraud Dataset ===")
    print(f"Transactions : {len(transactions):,}")
    print(f"Customers    : {len(customers):,}")
    print(f"Accounts     : {len(accounts):,}")
    print(f"Cards        : {len(cards):,}")
    print(f"Merchants    : {len(merchants):,}")
    print(f"Devices      : {len(devices):,}")
    print(f"IPs          : {len(ips):,}")
    print(f"Fraud        : {fraud_count:,}")
    print(f"Fraud rate   : {actual_fraud_rate:.2%}")
    print(f"Output       : {output_dir}")
    print("===============================\n")


if __name__ == "__main__":
    main()