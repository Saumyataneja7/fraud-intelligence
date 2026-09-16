from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_intelligence.ml.imbalance import (
    build_class_imbalance_strategy,
)
from fraud_intelligence.ml.logistic_regression import (
    LogisticRegressionConfig,
    LogisticRegressionError,
    LogisticRegressionModel,
    build_logistic_regression_pipeline,
    load_logistic_regression,
    predict_logistic_regression,
    predict_logistic_regression_proba,
    save_logistic_regression,
    train_logistic_regression,
)


def make_training_data(
    rows: int = 100,
    features: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(rows, features)),
        columns=[f"feature_{i}" for i in range(features)],
    )

    y = pd.Series(
        [0] * (rows - 10) + [1] * 10,
        name="is_fraud",
    )

    return X, y


def test_pipeline_contains_scaler_and_logistic_regression() -> None:
    class_weights = {0: 0.5555555556, 1: 5.0}

    pipeline = build_logistic_regression_pipeline(class_weights)

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(
        pipeline.named_steps["classifier"],
        LogisticRegression,
    )


def test_pipeline_uses_class_weights() -> None:
    X, y = make_training_data()

    strategy = build_class_imbalance_strategy(y)

    pipeline = build_logistic_regression_pipeline(
        strategy.sklearn_class_weight
    )

    classifier = pipeline.named_steps["classifier"]

    assert classifier.class_weight == strategy.sklearn_class_weight


def test_model_trains_successfully() -> None:
    X, y = make_training_data()

    model = train_logistic_regression(X, y)

    assert isinstance(model, LogisticRegressionModel)
    assert isinstance(model.pipeline, Pipeline)
    assert tuple(X.columns) == model.feature_columns


def test_model_uses_expected_feature_columns() -> None:
    X, y = make_training_data()

    model = train_logistic_regression(X, y)

    assert model.feature_columns == tuple(X.columns)
    assert len(model.feature_columns) == X.shape[1]


def test_prediction_returns_binary_values() -> None:
    X, y = make_training_data()

    model = train_logistic_regression(X, y)

    predictions = predict_logistic_regression(model, X)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_probability_prediction_returns_values_between_zero_and_one() -> None:
    X, y = make_training_data()

    model = train_logistic_regression(X, y)

    probabilities = predict_logistic_regression_proba(model, X)

    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == len(X)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_feature_column_mismatch_raises_error() -> None:
    X, y = make_training_data()

    model = train_logistic_regression(X, y)

    X_bad = X.rename(
        columns={"feature_0": "different_feature"}
    )

    with pytest.raises(
        LogisticRegressionError,
        match="Prediction features do not match training features",
    ):
        predict_logistic_regression(model, X_bad)


def test_probability_feature_column_mismatch_raises_error() -> None:
    X, y = make_training_data()

    model = train_logistic_regression(X, y)

    X_bad = X.rename(
        columns={"feature_0": "different_feature"}
    )

    with pytest.raises(
        LogisticRegressionError,
        match="Prediction features do not match training features",
    ):
        predict_logistic_regression_proba(model, X_bad)


def test_missing_prediction_values_raise_error() -> None:
    X, y = make_training_data()

    model = train_logistic_regression(X, y)

    X_bad = X.copy()
    X_bad.loc[0, "feature_0"] = np.nan

    with pytest.raises(
        LogisticRegressionError,
        match="Prediction features contain missing values",
    ):
        predict_logistic_regression(model, X_bad)


def test_missing_training_values_raise_error() -> None:
    X, y = make_training_data()

    X.loc[0, "feature_0"] = np.nan

    with pytest.raises(
        LogisticRegressionError,
        match="X_train contains missing values",
    ):
        train_logistic_regression(X, y)


def test_single_class_training_target_raises_error() -> None:
    X, _ = make_training_data()

    y = pd.Series([0] * len(X), name="is_fraud")

    with pytest.raises(
        LogisticRegressionError,
        match="y_train must contain both classes",
    ):
        train_logistic_regression(X, y)


def test_mismatched_training_rows_raise_error() -> None:
    X, y = make_training_data()

    y = y.iloc[:-1]

    with pytest.raises(
        LogisticRegressionError,
        match="same number of rows",
    ):
        train_logistic_regression(X, y)


def test_model_can_be_saved_and_loaded(tmp_path) -> None:
    X, y = make_training_data()

    model = train_logistic_regression(X, y)

    path = tmp_path / "logistic_regression.joblib"

    saved_path = save_logistic_regression(model, path)

    assert saved_path == path
    assert path.exists()

    loaded_model = load_logistic_regression(path)

    assert isinstance(loaded_model, LogisticRegressionModel)
    assert loaded_model.feature_columns == model.feature_columns


def test_loaded_model_produces_same_predictions(tmp_path) -> None:
    X, y = make_training_data()

    model = train_logistic_regression(X, y)

    path = tmp_path / "logistic_regression.joblib"

    save_logistic_regression(model, path)

    loaded_model = load_logistic_regression(path)

    original_predictions = predict_logistic_regression(
        model,
        X,
    )

    loaded_predictions = predict_logistic_regression(
        loaded_model,
        X,
    )

    np.testing.assert_array_equal(
        original_predictions,
        loaded_predictions,
    )


def test_loaded_model_produces_same_probabilities(tmp_path) -> None:
    X, y = make_training_data()

    model = train_logistic_regression(X, y)

    path = tmp_path / "logistic_regression.joblib"

    save_logistic_regression(model, path)

    loaded_model = load_logistic_regression(path)

    original_probabilities = predict_logistic_regression_proba(
        model,
        X,
    )

    loaded_probabilities = predict_logistic_regression_proba(
        loaded_model,
        X,
    )

    np.testing.assert_allclose(
        original_probabilities,
        loaded_probabilities,
    )


def test_custom_config_is_preserved() -> None:
    X, y = make_training_data()

    config = LogisticRegressionConfig(
        C=0.5,
        max_iter=500,
        random_state=123,
        solver="lbfgs",
    )

    model = train_logistic_regression(
        X,
        y,
        config=config,
    )

    classifier = model.pipeline.named_steps["classifier"]

    assert model.config == config
    assert classifier.C == 0.5
    assert classifier.max_iter == 500
    assert classifier.random_state == 123
    assert classifier.solver == "lbfgs"


def test_training_uses_training_target_for_class_weights() -> None:
    X, y = make_training_data()

    expected_strategy = build_class_imbalance_strategy(y)

    model = train_logistic_regression(X, y)

    assert model.class_weights == expected_strategy.sklearn_class_weight


def test_model_does_not_modify_training_features() -> None:
    X, y = make_training_data()

    original_X = X.copy()

    train_logistic_regression(X, y)

    pd.testing.assert_frame_equal(X, original_X)