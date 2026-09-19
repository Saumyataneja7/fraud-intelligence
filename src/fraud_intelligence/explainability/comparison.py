from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class ExplainabilityComparisonError(ValueError):
    """Raised when explainability comparison fails."""


@dataclass(frozen=True)
class ExplainabilityMethodComparison:
    """Structured comparison of one explainability method."""

    method: str
    model_family: str
    evidence_type: str
    scope: str
    feature_based: bool
    graph_based: bool
    directional: bool
    transaction_level: bool
    strengths: tuple[str, ...]
    limitations: tuple[str, ...]


COMPARISON_METHODS = (
    "Classical Attribution",
    "SHAP",
    "GNN Neighborhood",
    "GNN Importance",
)


def build_explainability_comparison() -> (
    tuple[ExplainabilityMethodComparison, ...]
):
    """
    Build the descriptive comparison of available
    explainability methods.

    This function does not rank methods or select a winner.
    """

    return (
        ExplainabilityMethodComparison(
            method="Classical Attribution",
            model_family="classical_ml",
            evidence_type="model feature attribution",
            scope="individual transaction",
            feature_based=True,
            graph_based=False,
            directional=True,
            transaction_level=True,
            strengths=(
                "Simple to interpret.",
                "Directly tied to classical model parameters "
                "or feature importance.",
                "Efficient to compute.",
            ),
            limitations=(
                "Does not explain graph structure.",
                "Feature importance semantics depend on the model.",
            ),
        ),
        ExplainabilityMethodComparison(
            method="SHAP",
            model_family="classical_ml",
            evidence_type="local feature attribution",
            scope="individual transaction",
            feature_based=True,
            graph_based=False,
            directional=True,
            transaction_level=True,
            strengths=(
                "Provides local feature contributions.",
                "Shows positive and negative feature effects.",
                "Provides a model-independent explanation interface "
                "for supported models.",
            ),
            limitations=(
                "Computational cost can be higher than native "
                "feature importance.",
                "Does not explain graph topology.",
            ),
        ),
        ExplainabilityMethodComparison(
            method="GNN Neighborhood",
            model_family="graph_ml",
            evidence_type="historical graph context",
            scope="transaction neighborhood",
            feature_based=False,
            graph_based=True,
            directional=False,
            transaction_level=True,
            strengths=(
                "Shows connected entities and transactions.",
                "Provides structural context around a transaction.",
                "Supports fraud-ring investigation context.",
            ),
            limitations=(
                "Shows structural context rather than causal evidence.",
                "Interpretation depends on graph construction.",
            ),
        ),
        ExplainabilityMethodComparison(
            method="GNN Importance",
            model_family="graph_ml",
            evidence_type="feature and node perturbation importance",
            scope="individual transaction",
            feature_based=True,
            graph_based=True,
            directional=False,
            transaction_level=True,
            strengths=(
                "Connects transaction features with graph context.",
                "Can identify influential neighboring nodes.",
                "Provides model-specific importance estimates.",
            ),
            limitations=(
                "Perturbation importance is not causal evidence.",
                "Results depend on the perturbation strategy.",
                "Can be computationally expensive.",
            ),
        ),
    )


def comparison_to_dataframe() -> pd.DataFrame:
    """Return the explainability comparison as a DataFrame."""

    comparisons = build_explainability_comparison()

    return pd.DataFrame(
        [
            {
                "method": item.method,
                "model_family": item.model_family,
                "evidence_type": item.evidence_type,
                "scope": item.scope,
                "feature_based": item.feature_based,
                "graph_based": item.graph_based,
                "directional": item.directional,
                "transaction_level": item.transaction_level,
                "strengths": item.strengths,
                "limitations": item.limitations,
            }
            for item in comparisons
        ]
    )


def validate_explainability_comparison(
    comparisons: tuple[
        ExplainabilityMethodComparison, ...
    ],
) -> None:
    """Validate the comparison schema and content."""

    if not comparisons:
        raise ExplainabilityComparisonError(
            "At least one explainability method is required."
        )

    methods = [
        item.method
        for item in comparisons
    ]

    if len(methods) != len(set(methods)):
        raise ExplainabilityComparisonError(
            "Explainability methods must be unique."
        )

    for item in comparisons:
        if not isinstance(
            item,
            ExplainabilityMethodComparison,
        ):
            raise ExplainabilityComparisonError(
                "All comparison entries must be "
                "ExplainabilityMethodComparison objects."
            )

        if not item.method:
            raise ExplainabilityComparisonError(
                "Method name cannot be empty."
            )

        if not item.model_family:
            raise ExplainabilityComparisonError(
                "Model family cannot be empty."
            )

        if not item.evidence_type:
            raise ExplainabilityComparisonError(
                "Evidence type cannot be empty."
            )

        if not item.scope:
            raise ExplainabilityComparisonError(
                "Scope cannot be empty."
            )

        if not item.strengths:
            raise ExplainabilityComparisonError(
                f"{item.method} must have at least one strength."
            )

        if not item.limitations:
            raise ExplainabilityComparisonError(
                f"{item.method} must have at least one limitation."
            )

        if not isinstance(item.feature_based, bool):
            raise ExplainabilityComparisonError(
                f"{item.method}.feature_based must be boolean."
            )

        if not isinstance(item.graph_based, bool):
            raise ExplainabilityComparisonError(
                f"{item.method}.graph_based must be boolean."
            )

        if not isinstance(item.directional, bool):
            raise ExplainabilityComparisonError(
                f"{item.method}.directional must be boolean."
            )

        if not isinstance(item.transaction_level, bool):
            raise ExplainabilityComparisonError(
                f"{item.method}.transaction_level must be boolean."
            )


def validate_comparison_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the tabular comparison representation."""

    if not isinstance(dataframe, pd.DataFrame):
        raise ExplainabilityComparisonError(
            "comparison must be a pandas DataFrame."
        )

    required_columns = {
        "method",
        "model_family",
        "evidence_type",
        "scope",
        "feature_based",
        "graph_based",
        "directional",
        "transaction_level",
        "strengths",
        "limitations",
    }

    missing = required_columns.difference(
        dataframe.columns
    )

    if missing:
        raise ExplainabilityComparisonError(
            "Comparison is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if dataframe.empty:
        raise ExplainabilityComparisonError(
            "Comparison DataFrame cannot be empty."
        )

    if dataframe["method"].duplicated().any():
        raise ExplainabilityComparisonError(
            "Comparison methods must be unique."
        )

    for column in (
        "feature_based",
        "graph_based",
        "directional",
        "transaction_level",
    ):
        if not dataframe[column].map(
            lambda value: isinstance(value, bool)
        ).all():
            raise ExplainabilityComparisonError(
                f"Column '{column}' must contain boolean values."
            )