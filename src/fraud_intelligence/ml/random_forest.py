from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from fraud_intelligence.ml.imbalance import (
    build_class_imbalance_strategy,
)


class RandomForestError(ValueError):
    """Raised when Random Forest preparation or training fails."""


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | int | float | None = "sqrt"
    random_state: int = 42
    n_jobs: int = -1


@dataclass(frozen=True)
class RandomForestModel:
    classifier: RandomForestClassifier
    feature_columns: tuple[str, ...]
    config: RandomForestConfig
    class_weights: dict[int, float]


def _validate_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    if not isinstance(X_train, pd.DataFrame):
        raise RandomForestError(
            "X_train must be a pandas DataFrame."
        )

    if not isinstance(y_train, pd.Series):
        raise RandomForestError(
            "y_train must be a pandas Series."
        )

    if X_train.empty:
        raise RandomForestError(
            "X_train cannot be empty."
        )

    if len(X_train) != len(y_train):
        raise RandomForestError(
            "X_train and y_train must have the same number of rows."
        )

    if X_train.isna().any().any():
        raise RandomForestError(
            "X_train contains missing values."
        )

    if y_train.isna().any():
        raise RandomForestError(
            "y_train contains missing values."
        )

    if not set(y_train.unique()).issubset({0, 1}):
        raise RandomForestError(
            "y_train must contain only binary values 0 and 1."
        )

    if y_train.nunique() != 2:
        raise RandomForestError(
            "y_train must contain both classes."
        )


def build_random_forest_classifier(
    class_weights: dict[int, float],
    config: RandomForestConfig | None = None,
) -> RandomForestClassifier:
    config = config or RandomForestConfig()

    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        class_weight=class_weights,
    )


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: RandomForestConfig | None = None,
) -> RandomForestModel:
    _validate_training_data(X_train, y_train)

    strategy = build_class_imbalance_strategy(y_train)

    classifier = build_random_forest_classifier(
        class_weights=strategy.sklearn_class_weight,
        config=config,
    )

    classifier.fit(X_train, y_train)

    return RandomForestModel(
        classifier=classifier,
        feature_columns=tuple(X_train.columns),
        config=config or RandomForestConfig(),
        class_weights=strategy.sklearn_class_weight,
    )


def _validate_prediction_data(
    model: RandomForestModel,
    X: pd.DataFrame,
) -> None:
    if not isinstance(X, pd.DataFrame):
        raise RandomForestError(
            "X must be a pandas DataFrame."
        )

    if tuple(X.columns) != model.feature_columns:
        raise RandomForestError(
            "Prediction features do not match training features."
        )

    if X.isna().any().any():
        raise RandomForestError(
            "Prediction features contain missing values."
        )


def predict_random_forest(
    model: RandomForestModel,
    X: pd.DataFrame,
) -> np.ndarray:
    _validate_prediction_data(model, X)

    return model.classifier.predict(X)


def predict_random_forest_proba(
    model: RandomForestModel,
    X: pd.DataFrame,
) -> np.ndarray:
    _validate_prediction_data(model, X)

    return model.classifier.predict_proba(X)[:, 1]


def save_random_forest(
    model: RandomForestModel,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

    return path


def load_random_forest(
    path: str | Path,
) -> RandomForestModel:
    path = Path(path)

    if not path.exists():
        raise RandomForestError(
            f"Model artifact does not exist: {path}"
        )

    model = joblib.load(path)

    if not isinstance(model, RandomForestModel):
        raise RandomForestError(
            "Invalid Random Forest model artifact."
        )

    return model