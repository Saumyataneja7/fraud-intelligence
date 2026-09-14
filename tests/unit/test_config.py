from fraud_intelligence.config.settings import (
    get_model_config,
    get_project_config,
)


def test_project_config() -> None:
    config = get_project_config()

    assert config["project"]["name"] == "fraud-intelligence"
    assert config["project"]["random_seed"] == 42


def test_model_config() -> None:
    config = get_model_config()

    assert config["target"]["name"] == "is_fraud"
    assert config["split"]["strategy"] == "temporal"
    assert config["evaluation"]["primary_metric"] == "pr_auc"