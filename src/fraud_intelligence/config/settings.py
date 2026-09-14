from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root() -> Path:
    """Find the project root by locating the configs directory."""

    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if (parent / "configs").is_dir():
            return parent

    raise RuntimeError(
        "Could not locate project root. "
        "Expected a directory containing 'configs'."
    )


PROJECT_ROOT = find_project_root()


class EnvironmentSettings(BaseSettings):
    """Environment-specific application settings."""

    app_env: str = "development"
    log_level: str = "INFO"

    model_dir: Path = PROJECT_ROOT / "models"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_yaml_config(filename: str) -> dict[str, Any]:
    """Load a YAML configuration file from the project config directory."""

    config_path = PROJECT_ROOT / "configs" / filename

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a mapping in configuration file: {config_path}"
        )

    return config


def get_project_config() -> dict[str, Any]:
    """Return the main project configuration."""

    return load_yaml_config("config.yaml")


def get_model_config() -> dict[str, Any]:
    """Return the ML model configuration."""

    return load_yaml_config("model_config.yaml")


settings = EnvironmentSettings()