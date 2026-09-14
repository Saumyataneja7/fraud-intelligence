from fraud_intelligence.config.bootstrap import initialize_project
from fraud_intelligence.config.settings import get_project_config
from fraud_intelligence.utils.logging import get_logger
from fraud_intelligence.utils.seeds import set_random_seed

logger = get_logger(__name__)


def main() -> None:
    """Initialize the fraud intelligence project."""

    config = get_project_config()

    initialize_project()
    set_random_seed(config["project"]["random_seed"])

    logger.info(
        "Initialized project '%s' version %s",
        config["project"]["name"],
        config["project"]["version"],
    )


if __name__ == "__main__":
    main()