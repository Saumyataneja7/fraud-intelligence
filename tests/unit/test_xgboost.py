from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from fraud_intelligence.ml.imbalance import (
    build_class_imbalance_strategy,
)
from fraud_intelligence.ml.xgboost import (
    XGBoostConfig,
    XGBoostError,
    XGBoostModel,
    build_xgboost_classifier,
    load_xgboost,
    predict_xgboost,
    predict_xgboost_proba,
    save_xgboost,
    train_xgboost,
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


def test_classifier_is_xgboost() -> None:
    classifier = build_xgboost_classifier(
        scale_pos_weight=9.0
    )

    assert isinstance(
        classifier,
        XGBClassifier,
    )


def test_classifier_uses_scale_pos_weight() -> None:
    X, y = make_training_data()

    strategy = build_class_imbalance_strategy(y)

    classifier = build_xgboost_classifier(
        strategy.xgboost_scale_pos_weight
    )

    assert (
        classifier.scale_pos_weight
        == strategy.xgboost_scale_pos_weight
    )


def test_model_trains_successfully() -> None:
    X, y = make_training_data()

    model = train_xgboost(X, y)

    assert isinstance(model, XGBoostModel)
    assert isinstance(
        model.classifier,
        XGBClassifier,
    )


def test_model_preserves_feature_columns() -> None:
    X, y = make_training_data()

    model = train_xgboost(X, y)

    assert model.feature_columns == tuple(X.columns)


def test_prediction_returns_binary_values() -> None:
    X, y = make_training_data()

    model = train_xgboost(X, y)

    predictions = predict_xgboost(
        model,
        X,
    )

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_probability_prediction_returns_values_between_zero_and_one() -> None:
    X, y = make_training_data()

    model = train_xgboost(X, y)

    probabilities = predict_xgboost_proba(
        model,
        X,
    )

    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == len(X)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_feature_column_mismatch_raises_error() -> None:
    X, y = make_training_data()

    model = train_xgboost(X, y)

    X_bad = X.rename(
        columns={
            "feature_0": "different_feature"
        }
    )

    with pytest.raises(
        XGBoostError,
        match="Prediction features do not match training features",
    ):
        predict_xgboost(model, X_bad)


def test_probability_feature_column_mismatch_raises_error() -> None:
    X, y = make_training_data()

    model = train_xgboost(X, y)

    X_bad = X.rename(
        columns={
            "feature_0": "different_feature"
        }
    )

    with pytest.raises(
        XGBoostError,
        match="Prediction features do not match training features",
    ):
        predict_xgboost_proba(model, X_bad)


def test_missing_training_values_raise_error() -> None:
    X, y = make_training_data()

    X.loc[0, "feature_0"] = np.nan

    with pytest.raises(
        XGBoostError,
        match="X_train contains missing values",
    ):
        train_xgboost(X, y)


def test_missing_prediction_values_raise_error() -> None:
    X, y = make_training_data()

    model = train_xgboost(X, y)

    X_bad = X.copy()
    X_bad.loc[0, "feature_0"] = np.nan

    with pytest.raises(
        XGBoostError,
        match="Prediction features contain missing values",
    ):
        predict_xgboost(model, X_bad)


def test_single_class_training_target_raises_error() -> None:
    X, _ = make_training_data()

    y = pd.Series(
        [0] * len(X),
        name="is_fraud",
    )

    with pytest.raises(
        XGBoostError,
        match="y_train must contain both classes",
    ):
        train_xgboost(X, y)


def test_mismatched_training_rows_raise_error() -> None:
    X, y = make_training_data()

    y = y.iloc[:-1]

    with pytest.raises(
        XGBoostError,
        match="same number of rows",
    ):
        train_xgboost(X, y)


def test_model_can_be_saved_and_loaded(tmp_path) -> None:
    X, y = make_training_data()

    model = train_xgboost(X, y)

    path = tmp_path / "xgboost.joblib"

    saved_path = save_xgboost(
        model,
        path,
    )

    assert saved_path == path
    assert path.exists()

    loaded_model = load_xgboost(path)

    assert isinstance(
        loaded_model,
        XGBoostModel,
    )

    assert (
        loaded_model.feature_columns
        == model.feature_columns
    )


def test_loaded_model_produces_same_predictions(tmp_path) -> None:
    X, y = make_training_data()

    model = train_xgboost(X, y)

    path = tmp_path / "xgboost.joblib"

    save_xgboost(model, path)

    loaded_model = load_xgboost(path)

    original_predictions = predict_xgboost(
        model,
        X,
    )

    loaded_predictions = predict_xgboost(
        loaded_model,
        X,
    )

    np.testing.assert_array_equal(
        original_predictions,
        loaded_predictions,
    )


def test_loaded_model_produces_same_probabilities(tmp_path) -> None:
    X, y = make_training_data()

    model = train_xgboost(X, y)

    path = tmp_path / "xgboost.joblib"

    save_xgboost(model, path)

    loaded_model = load_xgboost(path)

    original_probabilities = predict_xgboost_proba(
        model,
        X,
    )

    loaded_probabilities = predict_xgboost_proba(
        loaded_model,
        X,
    )

    np.testing.assert_allclose(
        original_probabilities,
        loaded_probabilities,
    )


def test_custom_config_is_preserved() -> None:
    X, y = make_training_data()

    config = XGBoostConfig(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2.0,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=123,
        n_jobs=1,
        eval_metric="logloss",
    )

    model = train_xgboost(
        X,
        y,
        config=config,
    )

    classifier = model.classifier

    assert model.config == config
    assert classifier.n_estimators == 50
    assert classifier.max_depth == 4
    assert classifier.learning_rate == 0.1
    assert classifier.subsample == 0.9
    assert classifier.colsample_bytree == 0.9
    assert classifier.min_child_weight == 2.0
    assert classifier.reg_alpha == 0.1
    assert classifier.reg_lambda == 2.0
    assert classifier.random_state == 123
    assert classifier.n_jobs == 1
    assert classifier.eval_metric == "logloss"


def test_training_uses_training_target_for_scale_pos_weight() -> None:
    X, y = make_training_data()

    expected_strategy = (
        build_class_imbalance_strategy(y)
    )

    model = train_xgboost(X, y)

    assert (
        model.scale_pos_weight
        == expected_strategy.xgboost_scale_pos_weight
    )


def test_model_does_not_modify_training_features() -> None:
    X, y = make_training_data()

    original_X = X.copy()

    train_xgboost(X, y)

    pd.testing.assert_frame_equal(
        X,
        original_X,
    )