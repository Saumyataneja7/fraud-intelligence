from fastapi import FastAPI

from fraud_intelligence.config.settings import get_project_config
from fraud_intelligence.utils.logging import configure_logging

config = get_project_config()

configure_logging(config["logging"]["level"])


app = FastAPI(
    title="Fraud Intelligence API",
    description=(
        "Fraud investigation API combining transaction-level ML, "
        "Graph ML, and explainable AI."
    ),
    version=config["project"]["version"],
)

@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "service": "Fraud Intelligence API",
        "version": config["project"]["version"],
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Return API health status."""

    return {
        "status": "healthy",
        "service": "fraud-intelligence-api",
        "version": config["project"]["version"],
    }