import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fraud_intelligence.explainability.contracts import (
    ExplainabilityContractError,
)
from fraud_intelligence.explainability.shap import (
    SHAPExplanation,
    SHAPExplanationError,
    SHAPFeatureAttribution,
    explain_transaction_with_shap,
    validate_shap_explanation,
)
from fraud_intelligence.ml.random_forest import (
    RandomForestConfig,
    RandomForestModel,
)


FEATURES = (
    "amount",
    "customer_txn_count_5m",
    "is_new_device",
    "amount_vs_customer_mean",
)


def make_model() -> RandomForestModel:
    X = pd.DataFrame(
        [
            [1.0, 0.0, 1.0, 0.5],
            [2.0, 1.0, 0.0, 0.2],
            [3.0, 2.0, 1.0, 0.8],
            [4.0, 3.0, 0.0, 1.2],
            [5.0, 4.0, 1.0, 1.5],
            [6.0, 5.0, 0.0, 2.0],
            [7.0, 6.0, 1.0, 2.5],
            [8.0, 7.0, 0.0, 3.0],
        ],
        columns=FEATURES,
    )

    y = pd.Series(
        [0, 0, 0, 1, 1, 1, 1, 1],
        name="is_fraud",
    )

    classifier = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
        n_jobs=1,
    )

    classifier.fit(X, y)

    return RandomForestModel(
        classifier=classifier,
        feature_columns=FEATURES,
        config=RandomForestConfig(
            n_estimators=10,
            n_jobs=1,
        ),
        class_weights={
            0: 1.0,
            1: 1.0,
        },
    )


def make_transaction() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                7.5,
                6.0,
                1.0,
                2.7,
            ]
        ],
        columns=FEATURES,
    )


def test_feature_attribution_dataclass():
    result = SHAPFeatureAttribution(
        feature="amount",
        shap_value=0.5,
        absolute_shap_value=0.5,
        rank=1,
    )

    assert result.feature == "amount"
    assert result.shap_value == 0.5
    assert result.absolute_shap_value == 0.5
    assert result.rank == 1


def test_explanation_dataclass():
    result = SHAPExplanation(
        model_name="RandomForest",
        feature_attributions=(),
        base_value=0.2,
        model_output=0.8,
    )

    assert result.model_name == "RandomForest"
    assert result.feature_count == 0
    assert result.top_feature is None
    assert result.positive_attributions == ()
    assert result.negative_attributions == ()


def test_random_forest_shap_explanation():
    model = make_model()
    X = make_transaction()

    result = explain_transaction_with_shap(
        model,
        X,
    )

    assert isinstance(
        result,
        SHAPExplanation,
    )

    assert result.model_name == "RandomForest"
    assert result.feature_count == len(FEATURES)
    assert np.isfinite(result.base_value)

    assert result.model_output is not None
    assert 0 <= result.model_output <= 1


def test_shap_features_match_model_features():
    model = make_model()
    X = make_transaction()

    result = explain_transaction_with_shap(
        model,
        X,
    )

    features = tuple(
        item.feature
        for item in result.feature_attributions
    )

    assert set(features) == set(FEATURES)


def test_shap_ranks_are_contiguous():
    model = make_model()
    X = make_transaction()

    result = explain_transaction_with_shap(
        model,
        X,
    )

    ranks = [
        item.rank
        for item in result.feature_attributions
    ]

    assert ranks == list(
        range(1, len(FEATURES) + 1)
    )


def test_shap_values_are_sorted_by_absolute_value():
    model = make_model()
    X = make_transaction()

    result = explain_transaction_with_shap(
        model,
        X,
    )

    values = [
        item.absolute_shap_value
        for item in result.feature_attributions
    ]

    assert values == sorted(
        values,
        reverse=True,
    )


def test_top_feature_is_rank_one():
    model = make_model()
    X = make_transaction()

    result = explain_transaction_with_shap(
        model,
        X,
    )

    assert result.top_feature is not None
    assert result.top_feature.rank == 1


def test_positive_and_negative_attributions_are_separated():
    model = make_model()
    X = make_transaction()

    result = explain_transaction_with_shap(
        model,
        X,
    )

    assert all(
        item.shap_value > 0
        for item in result.positive_attributions
    )

    assert all(
        item.shap_value < 0
        for item in result.negative_attributions
    )


def test_dataframe_conversion():
    model = make_model()
    X = make_transaction()

    result = explain_transaction_with_shap(
        model,
        X,
    )

    frame = result.to_dataframe()

    assert list(frame.columns) == [
        "feature",
        "shap_value",
        "absolute_shap_value",
        "rank",
    ]

    assert len(frame) == len(FEATURES)


def test_multiple_rows_rejected():
    model = make_model()

    X = pd.concat(
        [
            make_transaction(),
            make_transaction(),
        ],
        ignore_index=True,
    )

    with pytest.raises(SHAPExplanationError):
        explain_transaction_with_shap(
            model,
            X,
        )


def test_empty_dataframe_rejected():
    model = make_model()

    X = pd.DataFrame(
        columns=FEATURES,
    )

    with pytest.raises(SHAPExplanationError):
        explain_transaction_with_shap(
            model,
            X,
        )


def test_feature_order_mismatch_rejected():
    model = make_model()

    X = make_transaction()[
        [
            "is_new_device",
            "amount",
            "customer_txn_count_5m",
            "amount_vs_customer_mean",
        ]
    ]

    with pytest.raises(SHAPExplanationError):
        explain_transaction_with_shap(
            model,
            X,
        )


@pytest.mark.parametrize(
    "forbidden",
    [
        "is_fraud",
        "fraud_scenario",
        "transaction_id",
        "customer_id",
        "account_id",
        "card_id",
        "merchant_id",
        "device_id",
        "ip_id",
        "node_id",
        "timestamp",
    ],
)
def test_forbidden_feature_rejected(forbidden):
    model = make_model()

    columns = list(FEATURES)
    columns[1] = forbidden

    X = pd.DataFrame(
        [[1.0, 2.0, 1.0, 0.5]],
        columns=columns,
    )

    with pytest.raises(
        (
            SHAPExplanationError,
            ExplainabilityContractError,
        )
    ):
        explain_transaction_with_shap(
            model,
            X,
        )


def test_missing_values_rejected():
    model = make_model()

    X = make_transaction()
    X.loc[0, "amount"] = np.nan

    with pytest.raises(SHAPExplanationError):
        explain_transaction_with_shap(
            model,
            X,
        )


def test_non_finite_values_rejected():
    model = make_model()

    X = make_transaction()
    X.loc[0, "amount"] = np.inf

    with pytest.raises(SHAPExplanationError):
        explain_transaction_with_shap(
            model,
            X,
        )


def test_unsupported_model_rejected():
    with pytest.raises(SHAPExplanationError):
        explain_transaction_with_shap(
            object(),
            make_transaction(),
        )


def test_invalid_result_type_rejected():
    with pytest.raises(SHAPExplanationError):
        validate_shap_explanation(
            object()
        )


def test_invalid_base_value_rejected():
    result = SHAPExplanation(
        model_name="RandomForest",
        feature_attributions=(),
        base_value=np.nan,
    )

    with pytest.raises(SHAPExplanationError):
        validate_shap_explanation(result)


def test_invalid_model_output_rejected():
    result = SHAPExplanation(
        model_name="RandomForest",
        feature_attributions=(),
        base_value=0.2,
        model_output=1.5,
    )

    with pytest.raises(SHAPExplanationError):
        validate_shap_explanation(result)


def test_duplicate_features_rejected():
    result = SHAPExplanation(
        model_name="RandomForest",
        feature_attributions=(
            SHAPFeatureAttribution(
                feature="amount",
                shap_value=0.5,
                absolute_shap_value=0.5,
                rank=1,
            ),
            SHAPFeatureAttribution(
                feature="amount",
                shap_value=0.2,
                absolute_shap_value=0.2,
                rank=2,
            ),
        ),
        base_value=0.2,
    )

    with pytest.raises(SHAPExplanationError):
        validate_shap_explanation(result)


def test_invalid_absolute_shap_value_rejected():
    result = SHAPExplanation(
        model_name="RandomForest",
        feature_attributions=(
            SHAPFeatureAttribution(
                feature="amount",
                shap_value=0.5,
                absolute_shap_value=-0.5,
                rank=1,
            ),
        ),
        base_value=0.2,
    )

    with pytest.raises(SHAPExplanationError):
        validate_shap_explanation(result)


def test_mismatched_absolute_shap_value_rejected():
    result = SHAPExplanation(
        model_name="RandomForest",
        feature_attributions=(
            SHAPFeatureAttribution(
                feature="amount",
                shap_value=0.5,
                absolute_shap_value=0.2,
                rank=1,
            ),
        ),
        base_value=0.2,
    )

    with pytest.raises(SHAPExplanationError):
        validate_shap_explanation(result)


def test_forbidden_feature_in_result_rejected():
    result = SHAPExplanation(
        model_name="RandomForest",
        feature_attributions=(
            SHAPFeatureAttribution(
                feature="is_fraud",
                shap_value=0.5,
                absolute_shap_value=0.5,
                rank=1,
            ),
        ),
        base_value=0.2,
    )

    with pytest.raises(ExplainabilityContractError):
        validate_shap_explanation(result)