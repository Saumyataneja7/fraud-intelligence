from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureContract:
    """
    Defines the feature-engineering contract for the Fraud Intelligence
    pipeline.

    The contract explicitly separates:
    - model features
    - target columns
    - identifier columns
    - raw columns
    """

    target_columns: tuple[str, ...] = (
        "is_fraud",
        "fraud_scenario",
    )

    identifier_columns: tuple[str, ...] = (
        "transaction_id",
        "customer_id",
        "account_id",
        "card_id",
        "merchant_id",
        "device_id",
        "ip_id",
    )

    temporal_columns: tuple[str, ...] = (
        "timestamp",
    )

    raw_transaction_columns: tuple[str, ...] = (
        "amount",
        "currency",
        "payment_method",
        "transaction_type",
    )

    forbidden_feature_columns: tuple[str, ...] = field(
        default=(
            "is_fraud",
            "fraud_scenario",
        )
    )

    def validate_feature_columns(
        self,
        columns: list[str] | tuple[str, ...],
    ) -> None:
        """
        Validate that target columns are not present in the model feature set.
        """
        column_set = set(columns)

        forbidden = column_set.intersection(
            self.forbidden_feature_columns
        )

        if forbidden:
            raise ValueError(
                "Target leakage detected. "
                f"Forbidden columns found: {sorted(forbidden)}"
            )