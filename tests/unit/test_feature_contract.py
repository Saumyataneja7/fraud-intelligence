import pytest

from fraud_intelligence.features.contracts import (
    FeatureContract,
)


def test_contract_rejects_target_columns():
    contract = FeatureContract()

    with pytest.raises(ValueError):
        contract.validate_feature_columns(
            [
                "amount",
                "log_amount",
                "is_fraud",
            ]
        )


def test_contract_rejects_fraud_scenario():
    contract = FeatureContract()

    with pytest.raises(ValueError):
        contract.validate_feature_columns(
            [
                "amount",
                "fraud_scenario",
            ]
        )


def test_contract_accepts_safe_features():
    contract = FeatureContract()

    contract.validate_feature_columns(
        [
            "amount",
            "log_amount",
            "hour",
            "is_weekend",
        ]
    )