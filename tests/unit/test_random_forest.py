from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fraud_intelligence.ml.imbalance import (
    build_class_imbalance_strategy,
)
from fraud_intelligence.ml.random_forest import (
    RandomForestConfig,
    RandomForestError,
    RandomForestModel,
    build_random_forest_classifier,
    load_random_forest,
    predict_random_forest,
    predict_random_forest_proba,
    save_random_forest,
    train_random_forest,
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


def test_classifier_is_random_forest() -> None:
    class_weights = {
        0: 0.5555555556,
        1: 5.0,
    }

    classifier = build_random_forest_classifier(
        class_weights
    )

    assert isinstance(
        classifier,
        RandomForestClassifier,
    )


def test_classifier_uses_class_weights() -> None:
    X, y = make_training_data()

    strategy = build_class_imbalance_strategy(y)

    classifier = build_random_forest_classifier(
        strategy.sklearn_class_weight
    )

    assert (
        classifier.class_weight
        == strategy.sklearn_class_weight
    )


def test_model_trains_successfully() -> None:
    X, y = make_training_data()

    model = train_random_forest(X, y)

    assert isinstance(model, RandomForestModel)
    assert isinstance(
        model.classifier,
        RandomForestClassifier,
    )


def test_model_preserves_feature_columns() -> None:
    X, y = make_training_data()

    model = train_random_forest(X, y)

    assert model.feature_columns == tuple(X.columns)


def test_prediction_returns_binary_values() -> None:
    X, y = make_training_data()

    model = train_random_forest(X, y)

    predictions = predict_random_forest(
        model,
        X,
    )

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_probability_prediction_returns_values_between_zero_and_one() -> None:
    X, y = make_training_data()

    model = train_random_forest(X, y)

    probabilities = predict_random_forest_proba(
        model,
        X,
    )

    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == len(X)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_feature_column_mismatch_raises_error() -> None:
    X, y = make_training_data()

    model = train_random_forest(X, y)

    X_bad = X.rename(
        columns={
            "feature_0": "different_feature"
        }
    )

    with pytest.raises(
        RandomForestError,
        match="Prediction features do not match training features",
    ):
        predict_random_forest(model, X_bad)


def test_probability_feature_column_mismatch_raises_error() -> None:
    X, y = make_training_data()

    model = train_random_forest(X, y)

    X_bad = X.rename(
        columns={
            "feature_0": "different_feature"
        }
    )

    with pytest.raises(
        RandomForestError,
        match="Prediction features do not match training features",
    ):
        predict_random_forest_proba(model, X_bad)


def test_missing_training_values_raise_error() -> None:
    X, y = make_training_data()

    X.loc[0, "feature_0"] = np.nan

    with pytest.raises(
        RandomForestError,
        match="X_train contains missing values",
    ):
        train_random_forest(X, y)


def test_missing_prediction_values_raise_error() -> None:
    X, y = make_training_data()

    model = train_random_forest(X, y)

    X_bad = X.copy()
    X_bad.loc[0, "feature_0"] = np.nan

    with pytest.raises(
        RandomForestError,
        match="Prediction features contain missing values",
    ):
        predict_random_forest(model, X_bad)


def test_single_class_training_target_raises_error() -> None:
    X, _ = make_training_data()

    y = pd.Series(
        [0] * len(X),
        name="is_fraud",
    )

    with pytest.raises(
        RandomForestError,
        match="y_train must contain both classes",
    ):
        train_random_forest(X, y)


def test_mismatched_training_rows_raise_error() -> None:
    X, y = make_training_data()

    y = y.iloc[:-1]

    with pytest.raises(
        RandomForestError,
        match="same number of rows",
    ):
        train_random_forest(X, y)


def test_model_can_be_saved_and_loaded(tmp_path) -> None:
    X, y = make_training_data()

    model = train_random_forest(X, y)

    path = tmp_path / "random_forest.joblib"

    saved_path = save_random_forest(
        model,
        path,
    )

    assert saved_path == path
    assert path.exists()

    loaded_model = load_random_forest(path)

    assert isinstance(
        loaded_model,
        RandomForestModel,
    )

    assert (
        loaded_model.feature_columns
        == model.feature_columns
    )


def test_loaded_model_produces_same_predictions(tmp_path) -> None:
    X, y = make_training_data()

    model = train_random_forest(X, y)

    path = tmp_path / "random_forest.joblib"

    save_random_forest(model, path)

    loaded_model = load_random_forest(path)

    original_predictions = predict_random_forest(
        model,
        X,
    )

    loaded_predictions = predict_random_forest(
        loaded_model,
        X,
    )

    np.testing.assert_array_equal(
        original_predictions,
        loaded_predictions,
    )


def test_loaded_model_produces_same_probabilities(tmp_path) -> None:
    X, y = make_training_data()

    model = train_random_forest(X, y)

    path = tmp_path / "random_forest.joblib"

    save_random_forest(model, path)

    loaded_model = load_random_forest(path)

    original_probabilities = (
        predict_random_forest_proba(model, X)
    )

    loaded_probabilities = (
        predict_random_forest_proba(
            loaded_model,
            X,
        )
    )

    np.testing.assert_allclose(
        original_probabilities,
        loaded_probabilities,
    )


def test_custom_config_is_preserved() -> None:
    X, y = make_training_data()

    config = RandomForestConfig(
        n_estimators=50,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=123,
        n_jobs=1,
    )

    model = train_random_forest(
        X,
        y,
        config=config,
    )

    classifier = model.classifier

    assert model.config == config
    assert classifier.n_estimators == 50
    assert classifier.max_depth == 8
    assert classifier.min_samples_split == 4
    assert classifier.min_samples_leaf == 2
    assert classifier.random_state == 123
    assert classifier.n_jobs == 1


def test_training_uses_training_target_for_class_weights() -> None:
    X, y = make_training_data()

    expected_strategy = (
        build_class_imbalance_strategy(y)
    )

    model = train_random_forest(X, y)

    assert (
        model.class_weights
        == expected_strategy.sklearn_class_weight
    )


def test_model_does_not_modify_training_features() -> None:
    X, y = make_training_data()

    original_X = X.copy()

    train_random_forest(X, y)

    pd.testing.assert_frame_equal(
        X,
        original_X,
    )