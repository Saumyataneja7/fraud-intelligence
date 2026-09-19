import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_intelligence.explainability.classical import (
    ClassicalAttributionError,
    ClassicalAttributionResult,
    FeatureAttribution,
    calculate_classical_attributions,
    validate_classical_attribution,
)
from fraud_intelligence.explainability.contracts import (
    ExplainabilityContractError,
)
from fraud_intelligence.ml.logistic_regression import (
    LogisticRegressionConfig,
    LogisticRegressionModel,
)
from fraud_intelligence.ml.random_forest import (
    RandomForestConfig,
    RandomForestModel,
)
from fraud_intelligence.ml.xgboost import (
    XGBoostConfig,
    XGBoostModel,
)


FEATURES = (
    "amount",
    "customer_txn_count_5m",
    "is_new_device",
    "amount_vs_customer_mean",
)


def make_training_data():
    X = pd.DataFrame(
        [
            [1.0, 0.0, 1.0, 0.5],
            [2.0, 1.0, 0.0, 0.2],
            [3.0, 2.0, 1.0, 0.8],
            [4.0, 3.0, 0.0, 1.2],
            [5.0, 4.0, 1.0, 1.5],
            [6.0, 5.0, 0.0, 2.0],
        ],
        columns=FEATURES,
    )

    y = pd.Series(
        [0, 0, 0, 1, 1, 1],
        name="is_fraud",
    )

    return X, y


def make_logistic_model():
    X, y = make_training_data()

    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X, y)

    return LogisticRegressionModel(
        pipeline=pipeline,
        feature_columns=FEATURES,
        config=LogisticRegressionConfig(),
        class_weights={0: 1.0, 1: 1.0},
    )


def make_random_forest_model():
    X, y = make_training_data()

    classifier = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    classifier.fit(X, y)

    return RandomForestModel(
        classifier=classifier,
        feature_columns=FEATURES,
        config=RandomForestConfig(
            n_estimators=10,
        ),
        class_weights={0: 1.0, 1: 1.0},
    )


def test_feature_attribution_dataclass():
    result = FeatureAttribution(
        feature="amount",
        attribution=0.8,
        absolute_attribution=0.8,
        rank=1,
    )

    assert result.feature == "amount"
    assert result.attribution == 0.8
    assert result.absolute_attribution == 0.8
    assert result.rank == 1


def test_result_dataclass():
    result = ClassicalAttributionResult(
        model_name="LogisticRegression",
        model_family="classical_ml",
        feature_attributions=(),
    )

    assert result.model_name == "LogisticRegression"
    assert result.model_family == "classical_ml"
    assert result.feature_count == 0
    assert result.top_feature is None


def test_logistic_attributions():
    model = make_logistic_model()

    result = calculate_classical_attributions(
        model=model,
        feature_names=FEATURES,
    )

    assert isinstance(
        result,
        ClassicalAttributionResult,
    )

    assert result.model_name == "LogisticRegression"
    assert result.model_family == "classical_ml"
    assert result.feature_count == len(FEATURES)


def test_logistic_attributions_are_signed():
    model = make_logistic_model()

    result = calculate_classical_attributions(
        model=model,
        feature_names=FEATURES,
    )

    values = [
        item.attribution
        for item in result.feature_attributions
    ]

    assert any(
        value != 0
        for value in values
    )


def test_random_forest_attributions():
    model = make_random_forest_model()

    result = calculate_classical_attributions(
        model=model,
        feature_names=FEATURES,
    )

    assert result.model_name == "RandomForest"
    assert result.model_family == "classical_ml"
    assert result.feature_count == len(FEATURES)


def test_attributions_sorted_by_absolute_value():
    model = make_logistic_model()

    result = calculate_classical_attributions(
        model=model,
        feature_names=FEATURES,
    )

    absolute_values = [
        item.absolute_attribution
        for item in result.feature_attributions
    ]

    assert absolute_values == sorted(
        absolute_values,
        reverse=True,
    )


def test_attribution_ranks_are_contiguous():
    model = make_logistic_model()

    result = calculate_classical_attributions(
        model=model,
        feature_names=FEATURES,
    )

    ranks = [
        item.rank
        for item in result.feature_attributions
    ]

    assert ranks == list(
        range(1, len(FEATURES) + 1)
    )


def test_top_feature_is_rank_one():
    model = make_logistic_model()

    result = calculate_classical_attributions(
        model=model,
        feature_names=FEATURES,
    )

    assert result.top_feature is not None
    assert result.top_feature.rank == 1


def test_dataframe_conversion():
    model = make_logistic_model()

    result = calculate_classical_attributions(
        model=model,
        feature_names=FEATURES,
    )

    frame = result.to_dataframe()

    assert list(frame.columns) == [
        "feature",
        "attribution",
        "absolute_attribution",
        "rank",
    ]

    assert len(frame) == len(FEATURES)


def test_model_feature_columns_are_authoritative():
    model = make_logistic_model()

    result = calculate_classical_attributions(
        model=model
    )

    assert [
        item.feature
        for item in result.feature_attributions
    ]


def test_mismatched_feature_names_rejected():
    model = make_logistic_model()

    with pytest.raises(ClassicalAttributionError):
        calculate_classical_attributions(
            model=model,
            feature_names=(
                "amount",
                "wrong_feature",
                "is_new_device",
                "amount_vs_customer_mean",
            ),
        )


def test_duplicate_features_rejected():
    model = make_logistic_model()

    with pytest.raises(ClassicalAttributionError):
        calculate_classical_attributions(
            model=model,
            feature_names=(
                "amount",
                "amount",
                "is_new_device",
                "amount_vs_customer_mean",
            ),
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
    model = make_logistic_model()

    features = (
        "amount",
        forbidden,
        "is_new_device",
        "amount_vs_customer_mean",
    )

    with pytest.raises(ExplainabilityContractError):
        calculate_classical_attributions(
            model=model,
            feature_names=features,
        )


def test_empty_feature_names_rejected():
    model = make_logistic_model()

    with pytest.raises(ClassicalAttributionError):
        calculate_classical_attributions(
            model=model,
            feature_names=(),
        )


def test_unsupported_model_rejected():
    with pytest.raises(ClassicalAttributionError):
        calculate_classical_attributions(
            model=object(),
            feature_names=FEATURES,
        )


def test_invalid_result_type_rejected():
    with pytest.raises(ClassicalAttributionError):
        validate_classical_attribution(object())


def test_invalid_model_family_rejected():
    result = ClassicalAttributionResult(
        model_name="Test",
        model_family="gnn",
        feature_attributions=(),
    )

    with pytest.raises(ClassicalAttributionError):
        validate_classical_attribution(result)


def test_duplicate_attribution_features_rejected():
    result = ClassicalAttributionResult(
        model_name="Test",
        model_family="classical_ml",
        feature_attributions=(
            FeatureAttribution(
                feature="amount",
                attribution=0.5,
                absolute_attribution=0.5,
                rank=1,
            ),
            FeatureAttribution(
                feature="amount",
                attribution=0.2,
                absolute_attribution=0.2,
                rank=2,
            ),
        ),
    )

    with pytest.raises(ClassicalAttributionError):
        validate_classical_attribution(result)


def test_negative_absolute_attribution_rejected():
    result = ClassicalAttributionResult(
        model_name="Test",
        model_family="classical_ml",
        feature_attributions=(
            FeatureAttribution(
                feature="amount",
                attribution=0.5,
                absolute_attribution=-0.5,
                rank=1,
            ),
        ),
    )

    with pytest.raises(ClassicalAttributionError):
        validate_classical_attribution(result)


def test_mismatched_absolute_attribution_rejected():
    result = ClassicalAttributionResult(
        model_name="Test",
        model_family="classical_ml",
        feature_attributions=(
            FeatureAttribution(
                feature="amount",
                attribution=0.5,
                absolute_attribution=0.2,
                rank=1,
            ),
        ),
    )

    with pytest.raises(ClassicalAttributionError):
        validate_classical_attribution(result)