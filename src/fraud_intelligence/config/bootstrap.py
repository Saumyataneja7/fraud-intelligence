from pathlib import Path

from fraud_intelligence.config.settings import get_project_config
from fraud_intelligence.utils.logging import configure_logging


def initialize_project() -> None:
    """Create required project directories and configure logging."""

    config = get_project_config()

    directories = [
        config["data"]["raw_dir"],
        config["data"]["interim_dir"],
        config["data"]["processed_dir"],
        config["data"]["graph_dir"],
        config["models"]["classical_dir"],
        config["models"]["graph_dir"],
        config["models"]["metadata_dir"],
        config["reports"]["figures_dir"],
        config["reports"]["metrics_dir"],
        config["reports"]["data_quality_dir"],
        config["reports"]["model_card_dir"],
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    configure_logging(config["logging"]["level"])