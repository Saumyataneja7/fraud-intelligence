from __future__ import annotations

import pandas as pd
import pytest

from fraud_intelligence.explainability.comparison import (
    ExplainabilityComparisonError,
    ExplainabilityMethodComparison,
    build_explainability_comparison,
    comparison_to_dataframe,
    validate_comparison_dataframe,
    validate_explainability_comparison,
)


def test_four_explainability_methods_are_present() -> None:
    comparisons = build_explainability_comparison()

    assert len(comparisons) == 4

    assert {
        item.method
        for item in comparisons
    } == {
        "Classical Attribution",
        "SHAP",
        "GNN Neighborhood",
        "GNN Importance",
    }


def test_methods_are_unique() -> None:
    comparisons = build_explainability_comparison()

    methods = [
        item.method
        for item in comparisons
    ]

    assert len(methods) == len(set(methods))


def test_comparison_schema_validates() -> None:
    comparisons = build_explainability_comparison()

    validate_explainability_comparison(comparisons)


def test_classical_is_feature_based() -> None:
    comparison = next(
        item
        for item in build_explainability_comparison()
        if item.method == "Classical Attribution"
    )

    assert comparison.feature_based is True
    assert comparison.graph_based is False
    assert comparison.directional is True


def test_shap_is_local_and_directional() -> None:
    comparison = next(
        item
        for item in build_explainability_comparison()
        if item.method == "SHAP"
    )

    assert comparison.feature_based is True
    assert comparison.graph_based is False
    assert comparison.directional is True
    assert comparison.transaction_level is True


def test_gnn_neighborhood_is_graph_based() -> None:
    comparison = next(
        item
        for item in build_explainability_comparison()
        if item.method == "GNN Neighborhood"
    )

    assert comparison.feature_based is False
    assert comparison.graph_based is True
    assert comparison.transaction_level is True


def test_gnn_importance_combines_feature_and_graph_context() -> None:
    comparison = next(
        item
        for item in build_explainability_comparison()
        if item.method == "GNN Importance"
    )

    assert comparison.feature_based is True
    assert comparison.graph_based is True
    assert comparison.directional is False


def test_every_method_has_strengths_and_limitations() -> None:
    for item in build_explainability_comparison():
        assert item.strengths
        assert item.limitations


def test_dataframe_contains_expected_columns() -> None:
    dataframe = comparison_to_dataframe()

    assert list(dataframe.columns) == [
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
    ]


def test_dataframe_has_four_rows() -> None:
    dataframe = comparison_to_dataframe()

    assert len(dataframe) == 4


def test_dataframe_validates() -> None:
    dataframe = comparison_to_dataframe()

    validate_comparison_dataframe(dataframe)


def test_empty_comparison_rejected() -> None:
    with pytest.raises(
        ExplainabilityComparisonError,
        match="At least one",
    ):
        validate_explainability_comparison(())


def test_duplicate_methods_rejected() -> None:
    item = build_explainability_comparison()[0]

    with pytest.raises(
        ExplainabilityComparisonError,
        match="unique",
    ):
        validate_explainability_comparison(
            (item, item)
        )


def test_empty_method_name_rejected() -> None:
    item = ExplainabilityMethodComparison(
        method="",
        model_family="classical_ml",
        evidence_type="feature attribution",
        scope="transaction",
        feature_based=True,
        graph_based=False,
        directional=True,
        transaction_level=True,
        strengths=("simple",),
        limitations=("limited",),
    )

    with pytest.raises(
        ExplainabilityComparisonError,
        match="Method name",
    ):
        validate_explainability_comparison((item,))


def test_empty_strengths_rejected() -> None:
    item = ExplainabilityMethodComparison(
        method="Test",
        model_family="classical_ml",
        evidence_type="feature attribution",
        scope="transaction",
        feature_based=True,
        graph_based=False,
        directional=True,
        transaction_level=True,
        strengths=(),
        limitations=("limited",),
    )

    with pytest.raises(
        ExplainabilityComparisonError,
        match="strength",
    ):
        validate_explainability_comparison((item,))


def test_empty_limitations_rejected() -> None:
    item = ExplainabilityMethodComparison(
        method="Test",
        model_family="classical_ml",
        evidence_type="feature attribution",
        scope="transaction",
        feature_based=True,
        graph_based=False,
        directional=True,
        transaction_level=True,
        strengths=("simple",),
        limitations=(),
    )

    with pytest.raises(
        ExplainabilityComparisonError,
        match="limitation",
    ):
        validate_explainability_comparison((item,))


def test_missing_dataframe_column_rejected() -> None:
    dataframe = comparison_to_dataframe().drop(
        columns=["scope"]
    )

    with pytest.raises(
        ExplainabilityComparisonError,
        match="missing required columns",
    ):
        validate_comparison_dataframe(dataframe)


def test_empty_dataframe_rejected() -> None:
    dataframe = pd.DataFrame(
        columns=[
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
        ]
    )

    with pytest.raises(
        ExplainabilityComparisonError,
        match="cannot be empty",
    ):
        validate_comparison_dataframe(dataframe)


def test_duplicate_dataframe_methods_rejected() -> None:
    dataframe = comparison_to_dataframe()

    dataframe.loc[1, "method"] = dataframe.loc[0, "method"]

    with pytest.raises(
        ExplainabilityComparisonError,
        match="unique",
    ):
        validate_comparison_dataframe(dataframe)


def test_non_boolean_dataframe_flag_rejected() -> None:
    dataframe = comparison_to_dataframe()

    dataframe.loc[0, "feature_based"] = "yes"

    with pytest.raises(
        ExplainabilityComparisonError,
        match="must contain boolean",
    ):
        validate_comparison_dataframe(dataframe)