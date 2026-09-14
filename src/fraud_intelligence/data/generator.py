from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for synthetic fraud-data generation."""

    transactions: int = 100_000
    customers: int = 20_000
    accounts: int = 25_000
    cards: int = 30_000
    devices: int = 25_000
    ips: int = 30_000
    merchants: int = 1_000

    fraud_rate: float = 0.01
    seed: int = 42

    start_date: datetime = datetime(2025, 1, 1)
    end_date: datetime = datetime(2025, 12, 31)


class SyntheticFraudGenerator:
    """Generate a relational synthetic financial fraud dataset."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # CUSTOMER GENERATION
    # ------------------------------------------------------------------

    def generate_customers(self) -> pd.DataFrame:
        """Generate customer entities."""

        n = self.config.customers

        countries = np.array(
            ["IN", "US", "GB", "SG", "AE"]
        )

        segments = np.array(
            [
                "mass_market",
                "premium",
                "business",
                "student",
            ]
        )

        return pd.DataFrame(
            {
                "customer_id": [
                    f"CUST_{i:07d}" for i in range(n)
                ],
                "age": self.rng.integers(
                    18,
                    75,
                    size=n,
                ),
                "account_age_days": self.rng.integers(
                    30,
                    3650,
                    size=n,
                ),
                "country": self.rng.choice(
                    countries,
                    size=n,
                    p=[0.55, 0.15, 0.10, 0.08, 0.12],
                ),
                "customer_segment": self.rng.choice(
                    segments,
                    size=n,
                    p=[0.55, 0.20, 0.15, 0.10],
                ),
            }
        )

    # ------------------------------------------------------------------
    # ACCOUNT GENERATION
    # ------------------------------------------------------------------

    def generate_accounts(
        self,
        customers: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate accounts while preserving:

            Account -> Customer

        Every account belongs to exactly one customer.
        """

        n = self.config.accounts

        customer_ids = customers["customer_id"].to_numpy()

        # Guarantee that every customer gets at least one account
        if n >= len(customer_ids):
            remaining = n - len(customer_ids)

            assigned_customers = np.concatenate(
                [
                    customer_ids,
                    self.rng.choice(
                        customer_ids,
                        size=remaining,
                        replace=True,
                    ),
                ]
            )

            self.rng.shuffle(assigned_customers)

        else:
            assigned_customers = self.rng.choice(
                customer_ids,
                size=n,
                replace=False,
            )

        return pd.DataFrame(
            {
                "account_id": [
                    f"ACC_{i:07d}" for i in range(n)
                ],
                "customer_id": assigned_customers,
                "account_type": self.rng.choice(
                    [
                        "checking",
                        "savings",
                        "business",
                    ],
                    size=n,
                    p=[0.55, 0.35, 0.10],
                ),
                "account_age_days": self.rng.integers(
                    30,
                    3650,
                    size=n,
                ),
            }
        )

    # ------------------------------------------------------------------
    # CARD GENERATION
    # ------------------------------------------------------------------

    def generate_cards(
            self,
            accounts: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate cards while preserving:

            Card -> Account -> Customer

        Every account receives at least one card.
        """

        account_ids = accounts["account_id"].to_numpy()

        n_accounts = len(account_ids)
        n_cards = self.config.cards

        if n_cards < n_accounts:
            raise ValueError(
                "Number of cards must be at least the number of accounts "
                "so every account can receive a card."
            )

        # Guarantee one card for every account.
        assigned_accounts = list(account_ids)

        # Allocate remaining cards probabilistically.
        remaining_cards = n_cards - n_accounts

        if remaining_cards > 0:
            additional_accounts = self.rng.choice(
                account_ids,
                size=remaining_cards,
                replace=True,
            )

            assigned_accounts.extend(
                additional_accounts.tolist()
            )

        # Shuffle so card IDs are not correlated with account IDs.
        assigned_accounts = np.asarray(
            assigned_accounts,
            dtype=object,
        )

        self.rng.shuffle(assigned_accounts)

        return pd.DataFrame(
            {
                "card_id": [
                    f"CARD_{i:07d}"
                    for i in range(n_cards)
                ],
                "account_id": assigned_accounts,
                "card_type": self.rng.choice(
                    [
                        "debit",
                        "credit",
                        "virtual",
                    ],
                    size=n_cards,
                    p=[
                        0.60,
                        0.30,
                        0.10,
                    ],
                ),
                "card_age_days": self.rng.integers(
                    7,
                    2500,
                    size=n_cards,
                ),
            }
        )

    # ------------------------------------------------------------------
    # MERCHANT GENERATION
    # ------------------------------------------------------------------

    def generate_merchants(self) -> pd.DataFrame:
        """Generate merchant entities."""

        categories = np.array(
            [
                "grocery",
                "electronics",
                "travel",
                "fashion",
                "food",
                "gaming",
                "digital_services",
                "luxury",
                "utilities",
                "other",
            ]
        )

        countries = np.array(
            ["IN", "US", "GB", "SG", "AE"]
        )

        n = self.config.merchants

        return pd.DataFrame(
            {
                "merchant_id": [
                    f"MERCH_{i:06d}" for i in range(n)
                ],
                "merchant_category": self.rng.choice(
                    categories,
                    size=n,
                ),
                "merchant_country": self.rng.choice(
                    countries,
                    size=n,
                ),
                "risk_tier": self.rng.choice(
                    [
                        "LOW",
                        "MEDIUM",
                        "HIGH",
                    ],
                    size=n,
                    p=[0.70, 0.25, 0.05],
                ),
            }
        )

    # ------------------------------------------------------------------
    # DEVICE GENERATION
    # ------------------------------------------------------------------

    def generate_devices(self) -> pd.DataFrame:
        """Generate device entities."""

        n = self.config.devices

        return pd.DataFrame(
            {
                "device_id": [
                    f"DEV_{i:07d}" for i in range(n)
                ],
                "device_type": self.rng.choice(
                    [
                        "mobile",
                        "desktop",
                        "tablet",
                    ],
                    size=n,
                    p=[0.70, 0.20, 0.10],
                ),
                "operating_system": self.rng.choice(
                    [
                        "ios",
                        "android",
                        "windows",
                        "macos",
                    ],
                    size=n,
                ),
            }
        )

    # ------------------------------------------------------------------
    # IP GENERATION
    # ------------------------------------------------------------------

    def generate_ips(self) -> pd.DataFrame:
        """Generate IP entities."""

        n = self.config.ips

        return pd.DataFrame(
            {
                "ip_id": [
                    f"IP_{i:07d}" for i in range(n)
                ],
                "country": self.rng.choice(
                    [
                        "IN",
                        "US",
                        "GB",
                        "SG",
                        "AE",
                    ],
                    size=n,
                ),
                "risk_score": np.round(
                    self.rng.beta(
                        2,
                        8,
                        size=n,
                    ),
                    4,
                ),
                "network_type": self.rng.choice(
                    [
                        "residential",
                        "mobile",
                        "corporate",
                        "vpn",
                    ],
                    size=n,
                    p=[0.55, 0.25, 0.15, 0.05],
                ),
            }
        )

    # ------------------------------------------------------------------
    # CUSTOMER -> DEVICE RELATIONSHIPS
    # ------------------------------------------------------------------

    def generate_customer_devices(
        self,
        customers: pd.DataFrame,
        devices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate normal customer-device relationships.

        Most customers get 1-2 devices.
        """

        customer_ids = customers["customer_id"].to_numpy()
        device_ids = devices["device_id"].to_numpy()

        relationships: list[tuple[str, str]] = []

        for customer_id in customer_ids:
            number_of_devices = int(
                self.rng.choice(
                    [1, 2, 3],
                    p=[0.65, 0.30, 0.05],
                )
            )

            selected_devices = self.rng.choice(
                device_ids,
                size=number_of_devices,
                replace=False,
            )

            relationships.extend(
                (customer_id, device_id)
                for device_id in selected_devices
            )

        return pd.DataFrame(
            relationships,
            columns=[
                "customer_id",
                "device_id",
            ],
        )

    # ------------------------------------------------------------------
    # CUSTOMER -> IP RELATIONSHIPS
    # ------------------------------------------------------------------

    def generate_customer_ips(
        self,
        customers: pd.DataFrame,
        ips: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate normal customer-IP relationships.

        Most customers get 1-3 usual IP addresses.
        """

        customer_ids = customers["customer_id"].to_numpy()
        ip_ids = ips["ip_id"].to_numpy()

        relationships: list[tuple[str, str]] = []

        for customer_id in customer_ids:
            number_of_ips = int(
                self.rng.choice(
                    [1, 2, 3],
                    p=[0.45, 0.40, 0.15],
                )
            )

            selected_ips = self.rng.choice(
                ip_ids,
                size=number_of_ips,
                replace=False,
            )

            relationships.extend(
                (customer_id, ip_id)
                for ip_id in selected_ips
            )

        return pd.DataFrame(
            relationships,
            columns=[
                "customer_id",
                "ip_id",
            ],
        )

    # ------------------------------------------------------------------
    # CUSTOMER -> MERCHANT RELATIONSHIPS
    # ------------------------------------------------------------------

    def generate_customer_merchants(
        self,
        customers: pd.DataFrame,
        merchants: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate customer merchant preferences.

        Each customer gets a small set of preferred merchants.
        """

        customer_ids = customers["customer_id"].to_numpy()
        merchant_ids = merchants["merchant_id"].to_numpy()

        relationships: list[tuple[str, str]] = []

        for customer_id in customer_ids:
            number_of_merchants = int(
                self.rng.integers(
                    3,
                    11,
                )
            )

            selected_merchants = self.rng.choice(
                merchant_ids,
                size=number_of_merchants,
                replace=False,
            )

            relationships.extend(
                (customer_id, merchant_id)
                for merchant_id in selected_merchants
            )

        return pd.DataFrame(
            relationships,
            columns=[
                "customer_id",
                "merchant_id",
            ],
        )

    # ------------------------------------------------------------------
    # NORMAL TRANSACTIONS
    # ------------------------------------------------------------------

    def generate_transactions(
        self,
        customers: pd.DataFrame,
        accounts: pd.DataFrame,
        cards: pd.DataFrame,
        merchants: pd.DataFrame,
        devices: pd.DataFrame,
        ips: pd.DataFrame,
        customer_devices: pd.DataFrame,
        customer_ips: pd.DataFrame,
        customer_merchants: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate normal transactions while preserving entity relationships.

        Transaction relationships:

            Customer
                |
                +-- Account
                |     |
                |     +-- Card
                |
                +-- Device
                |
                +-- IP
                |
                +-- Merchant
        """

        n = self.config.transactions

        customer_ids = customers["customer_id"].to_numpy()

        # --------------------------------------------------------------
        # Sample customers first
        # --------------------------------------------------------------

        sampled_customers = self.rng.choice(
            customer_ids,
            size=n,
            replace=True,
        )

        # --------------------------------------------------------------
        # Build lookup maps
        # --------------------------------------------------------------

        accounts_by_customer = (
            accounts
            .groupby("customer_id")["account_id"]
            .apply(list)
            .to_dict()
        )

        cards_by_account = (
            cards
            .groupby("account_id")["card_id"]
            .apply(list)
            .to_dict()
        )

        devices_by_customer = (
            customer_devices
            .groupby("customer_id")["device_id"]
            .apply(list)
            .to_dict()
        )

        ips_by_customer = (
            customer_ips
            .groupby("customer_id")["ip_id"]
            .apply(list)
            .to_dict()
        )

        merchants_by_customer = (
            customer_merchants
            .groupby("customer_id")["merchant_id"]
            .apply(list)
            .to_dict()
        )

        # --------------------------------------------------------------
        # Resolve related entities
        # --------------------------------------------------------------

        transaction_accounts: list[str] = []
        transaction_cards: list[str] = []
        transaction_devices: list[str] = []
        transaction_ips: list[str] = []
        transaction_merchants: list[str] = []

        for customer_id in sampled_customers:
            customer_accounts = accounts_by_customer[customer_id]

            account_id = self.rng.choice(
                customer_accounts
            )

            customer_cards = cards_by_account.get(
                account_id,
                [],
            )

            if customer_cards:
                card_id = self.rng.choice(
                    customer_cards
                )
            else:
                card_id = None

            customer_devices_list = devices_by_customer[
                customer_id
            ]

            customer_ips_list = ips_by_customer[
                customer_id
            ]

            customer_merchants_list = (
                merchants_by_customer[customer_id]
            )

            transaction_accounts.append(account_id)
            transaction_cards.append(card_id)
            transaction_devices.append(
                self.rng.choice(
                    customer_devices_list
                )
            )
            transaction_ips.append(
                self.rng.choice(
                    customer_ips_list
                )
            )
            transaction_merchants.append(
                self.rng.choice(
                    customer_merchants_list
                )
            )

        # --------------------------------------------------------------
        # Timestamp generation
        # --------------------------------------------------------------

        start_timestamp = int(
            pd.Timestamp(
                self.config.start_date,
                tz="UTC",
            ).timestamp()
        )

        end_timestamp = int(
            pd.Timestamp(
                self.config.end_date,
                tz="UTC",
            ).timestamp()
        )

        timestamps = pd.to_datetime(
            self.rng.integers(
                start_timestamp,
                end_timestamp,
                size=n,
            ),
            unit="s",
            utc=True,
        )

        # --------------------------------------------------------------
        # Transaction amounts
        # --------------------------------------------------------------

        amounts = np.round(
            np.exp(
                self.rng.normal(
                    loc=3.8,
                    scale=1.1,
                    size=n,
                )
            ),
            2,
        )

        # --------------------------------------------------------------
        # Transaction types
        # --------------------------------------------------------------

        transaction_types = self.rng.choice(
            [
                "purchase",
                "transfer",
                "withdrawal",
                "payment",
            ],
            size=n,
            p=[
                0.55,
                0.20,
                0.10,
                0.15,
            ],
        )

        # --------------------------------------------------------------
        # Payment methods
        # --------------------------------------------------------------

        payment_methods = self.rng.choice(
            [
                "card",
                "bank_transfer",
                "wallet",
                "online",
            ],
            size=n,
            p=[
                0.55,
                0.20,
                0.15,
                0.10,
            ],
        )

        return pd.DataFrame(
            {
                "transaction_id": [
                    f"TXN_{i:09d}"
                    for i in range(n)
                ],
                "timestamp": timestamps,
                "customer_id": sampled_customers,
                "account_id": transaction_accounts,
                "card_id": transaction_cards,
                "merchant_id": transaction_merchants,
                "device_id": transaction_devices,
                "ip_id": transaction_ips,
                "amount": amounts,
                "currency": "USD",
                "payment_method": payment_methods,
                "transaction_type": transaction_types,
                "is_fraud": np.zeros(
                    n,
                    dtype=np.int8,
                ),
                "fraud_scenario": np.full(
                    n,
                    "LEGITIMATE",
                    dtype=object,
                ),
            }
        )

    # ------------------------------------------------------------------
    # FRAUD INJECTION
    # ------------------------------------------------------------------

    def inject_fraud(
            self,
            transactions: pd.DataFrame,
            customers: pd.DataFrame,
            accounts: pd.DataFrame,
            cards: pd.DataFrame,
            devices: pd.DataFrame,
            ips: pd.DataFrame,
            merchants: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Inject controlled fraud scenarios into normal transactions.

        Fraud scenarios:

        - ACCOUNT_TAKEOVER
        - SHARED_DEVICE
        - SHARED_IP
        - MERCHANT_ABUSE
        - VELOCITY_ATTACK
        - GEO_ANOMALY
        - FRAUD_RING

        Fraud rings deliberately create shared infrastructure across
        multiple customers while preserving Customer -> Account -> Card
        referential integrity.
        """

        transactions = transactions.copy()

        n_transactions = len(transactions)

        n_fraud = int(
            round(
            n_transactions * self.config.fraud_rate
            )
        )

        if n_fraud == 0:
            return transactions

        # ------------------------------------------------------------------
        # Select fraud transactions
        # ------------------------------------------------------------------

        fraud_indices = self.rng.choice(
            transactions.index.to_numpy(),
            size=n_fraud,
            replace=False,
        )

        scenario_names = np.array(
            [
                "ACCOUNT_TAKEOVER",
                "SHARED_DEVICE",
                "SHARED_IP",
                "MERCHANT_ABUSE",
                "VELOCITY_ATTACK",
                "GEO_ANOMALY",
                "FRAUD_RING",
            ]
        )

        scenario_probabilities = np.array(
            [
                0.20,
                0.15,
                0.15,
                0.15,
                0.15,
                0.05,
                0.15,
            ]
        )

        scenarios = self.rng.choice(
            scenario_names,
            size=n_fraud,
            p=scenario_probabilities,
        )

        # ------------------------------------------------------------------
        # Identify fraud-ring transactions
        # ------------------------------------------------------------------

        ring_indices = [
            index
            for index, scenario in zip(
                fraud_indices,
                scenarios,
            )
            if scenario == "FRAUD_RING"
        ]

        # ------------------------------------------------------------------
        # Build entity lookup maps
        # ------------------------------------------------------------------

        accounts_by_customer = (
            accounts
            .groupby("customer_id")["account_id"]
            .apply(list)
            .to_dict()
        )

        cards_by_account = (
            cards
            .groupby("account_id")["card_id"]
            .apply(list)
            .to_dict()
        )

        device_ids = devices[
            "device_id"
        ].to_numpy()

        ip_ids = ips[
            "ip_id"
        ].to_numpy()

        merchant_ids = merchants[
            "merchant_id"
        ].to_numpy()

        # ------------------------------------------------------------------
        # Generate actual fraud rings
        # ------------------------------------------------------------------

        if ring_indices:

            # Number of separate fraud rings.
            #
            # A ring typically contains 4-8 customers.
            number_of_rings = max(
                1,
                min(
                    len(ring_indices) // 4,
                    10,
                ),
            )

            ring_groups = np.array_split(
                np.asarray(ring_indices),
                number_of_rings,
            )

            for ring_number, ring_group in enumerate(
                ring_groups,
                start=1,
            ):

                if len(ring_group) == 0:
                    continue

                # ----------------------------------------------------------
                # Select customers for this ring
                # ----------------------------------------------------------

                ring_size = int(
                    self.rng.integers(
                        4,
                        9,
                    )
                )

                ring_customers = self.rng.choice(
                    customers["customer_id"].to_numpy(),
                    size=ring_size,
                    replace=False,
                )

                # ----------------------------------------------------------
                # Shared infrastructure
                # ----------------------------------------------------------

                shared_device = self.rng.choice(
                    device_ids
                )

                shared_ip = self.rng.choice(
                    ip_ids
                )

                shared_merchant = self.rng.choice(
                    merchant_ids
                )

                # Create a short temporal window for the ring.
                #
                # This allows later graph/time-window features to detect
                # coordinated activity.
                ring_start = pd.Timestamp(
                    self.config.start_date,
                    tz="UTC",
                ) + pd.Timedelta(
                    days=int(
                        self.rng.integers(
                            0,
                            350,
                        )
                    )
                )

                # ----------------------------------------------------------
                # Assign fraudulent transactions to ring members
                # ----------------------------------------------------------

                for position, index in enumerate(
                    ring_group
                ):

                    customer_id = ring_customers[
                        position % len(ring_customers)
                    ]

                    # ------------------------------------------------------
                    # Preserve Customer -> Account
                    # ------------------------------------------------------

                    customer_accounts = (
                        accounts_by_customer[
                            customer_id
                        ]
                    )

                    account_id = self.rng.choice(
                        customer_accounts
                    )

                    # ------------------------------------------------------
                    # Preserve Account -> Card
                    # ------------------------------------------------------

                    customer_cards = cards_by_account.get(
                        account_id,
                        [],
                    )

                    if not customer_cards:
                        raise RuntimeError(
                            "Fraud-ring customer has no associated card."
                        )

                    card_id = self.rng.choice(
                        customer_cards
                    )

                    # ------------------------------------------------------
                    # Create transaction
                    # ------------------------------------------------------

                    transactions.at[
                        index,
                        "customer_id",
                    ] = customer_id

                    transactions.at[
                        index,
                        "account_id",
                    ] = account_id

                    transactions.at[
                        index,
                        "card_id",
                    ] = card_id

                    transactions.at[
                        index,
                        "device_id",
                    ] = shared_device

                    transactions.at[
                        index,
                        "ip_id",
                    ] = shared_ip

                    transactions.at[
                        index,
                        "merchant_id",
                    ] = shared_merchant

                    transactions.at[
                        index,
                        "timestamp",
                    ] = (
                        ring_start
                        + pd.Timedelta(
                            minutes=int(
                                self.rng.integers(
                                    0,
                                    1440,
                                )
                            )
                        )
                    )

                    transactions.at[
                        index,
                        "amount",
                    ] = round(
                        float(
                            transactions.at[
                                index,
                                "amount",
                            ]
                        )
                        * self.rng.uniform(
                            2.0,
                            7.0,
                        ),
                        2,
                    )

                    transactions.at[
                        index,
                        "transaction_type",
                    ] = "transfer"

                    transactions.at[
                        index,
                        "is_fraud",
                    ] = 1

                    transactions.at[
                        index,
                        "fraud_scenario",
                    ] = "FRAUD_RING"

        # ------------------------------------------------------------------
        # Other fraud scenarios
        # ------------------------------------------------------------------

        fraud_devices = devices[
            "device_id"
        ].to_numpy()

        fraud_ips = ips[
            "ip_id"
        ].to_numpy()

        fraud_merchants = merchants[
            "merchant_id"
        ].to_numpy()

        for index, scenario in zip(
            fraud_indices,
            scenarios,
        ):

            # FRAUD_RING was already handled above.
            if scenario == "FRAUD_RING":
                continue

            transactions.at[
                index,
                "is_fraud",
            ] = 1

            transactions.at[
                index,
                "fraud_scenario",
            ] = scenario

            # --------------------------------------------------------------
            # ACCOUNT TAKEOVER
            # --------------------------------------------------------------

            if scenario == "ACCOUNT_TAKEOVER":

                transactions.at[
                    index,
                    "device_id",
                ] = self.rng.choice(
                    fraud_devices
                )

                transactions.at[
                    index,
                    "ip_id",
                ] = self.rng.choice(
                    fraud_ips
                )

                transactions.at[
                    index,
                    "amount",
                ] = round(
                    float(
                        transactions.at[
                            index,
                            "amount",
                        ]
                    )
                    * self.rng.uniform(
                        2.0,
                        6.0,
                    ),
                    2,
                )

                transactions.at[
                    index,
                    "transaction_type",
                ] = "transfer"

            # --------------------------------------------------------------
            # SHARED DEVICE
            # --------------------------------------------------------------

            elif scenario == "SHARED_DEVICE":

                transactions.at[
                    index,
                    "device_id",
                ] = self.rng.choice(
                    fraud_devices
                )

                transactions.at[
                    index,
                    "amount",
                ] = round(
                    float(
                        transactions.at[
                            index,
                            "amount",
                        ]
                    )
                    * self.rng.uniform(
                        1.5,
                        4.0,
                    ),
                    2,
                )

            # --------------------------------------------------------------
            # SHARED IP
            # --------------------------------------------------------------

            elif scenario == "SHARED_IP":

                transactions.at[
                    index,
                    "ip_id",
                ] = self.rng.choice(
                    fraud_ips
                )

                transactions.at[
                    index,
                    "amount",
                ] = round(
                    float(
                        transactions.at[
                            index,
                            "amount",
                        ]
                    )
                    * self.rng.uniform(
                        1.5,
                        3.0,
                    ),
                    2,
                )

            # --------------------------------------------------------------
            # MERCHANT ABUSE
            # --------------------------------------------------------------

            elif scenario == "MERCHANT_ABUSE":

                transactions.at[
                    index,
                    "merchant_id",
                ] = self.rng.choice(
                fraud_merchants
                )

                transactions.at[
                    index,
                    "amount",
                ] = round(
                    float(
                        transactions.at[
                            index,
                            "amount",
                        ]
                    )
                    * self.rng.uniform(
                        3.0,
                        8.0,
                    ),
                    2,
                )

            # --------------------------------------------------------------
            # VELOCITY ATTACK
            # --------------------------------------------------------------

            elif scenario == "VELOCITY_ATTACK":

                customer_id = transactions.at[
                    index,
                    "customer_id",
                ]

                same_customer = transactions[
                    transactions["customer_id"]
                    == customer_id
                ]

                if not same_customer.empty:

                    reference_time = (
                        same_customer[
                            "timestamp"
                        ].iloc[0]
                    )

                    transactions.at[
                        index,
                        "timestamp",
                    ] = reference_time

                transactions.at[
                    index,
                    "amount",
                ] = round(
                    float(
                        transactions.at[
                            index,
                            "amount",
                        ]
                    )
                    * self.rng.uniform(
                        2.0,
                        5.0,
                    ),
                    2,
                )

            # --------------------------------------------------------------
            # GEO ANOMALY
            # --------------------------------------------------------------

            elif scenario == "GEO_ANOMALY":

                transactions.at[
                    index,
                    "ip_id",
                ] = self.rng.choice(
                    fraud_ips
                )

                transactions.at[
                    index,
                    "device_id",
                ] = self.rng.choice(
                    fraud_devices
                )

                transactions.at[
                    index,
                    "amount",
                ] = round(
                    float(
                        transactions.at[
                            index,
                            "amount",
                        ]
                    )
                    * self.rng.uniform(
                        2.0,
                        4.0,
                    ),
                    2,
                )

        return transactions