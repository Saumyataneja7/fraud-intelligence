from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_intelligence.ml.imbalance import (
    build_class_imbalance_strategy,
)


class LogisticRegressionError(ValueError):
    """Raised when Logistic Regression preparation or training fails."""


@dataclass(frozen=True)
class LogisticRegressionConfig:
    C: float = 1.0
    max_iter: int = 1000
    random_state: int = 42
    solver: str = "lbfgs"


@dataclass(frozen=True)
class LogisticRegressionModel:
    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    config: LogisticRegressionConfig
    class_weights: dict[int, float]


def _validate_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    if not isinstance(X_train, pd.DataFrame):
        raise LogisticRegressionError("X_train must be a pandas DataFrame.")

    if not isinstance(y_train, pd.Series):
        raise LogisticRegressionError("y_train must be a pandas Series.")

    if X_train.empty:
        raise LogisticRegressionError("X_train cannot be empty.")

    if len(X_train) != len(y_train):
        raise LogisticRegressionError(
            "X_train and y_train must have the same number of rows."
        )

    if X_train.isna().any().any():
        raise LogisticRegressionError(
            "X_train contains missing values."
        )

    if y_train.isna().any():
        raise LogisticRegressionError(
            "y_train contains missing values."
        )

    if not set(y_train.unique()).issubset({0, 1}):
        raise LogisticRegressionError(
            "y_train must contain only binary values 0 and 1."
        )

    if y_train.nunique() != 2:
        raise LogisticRegressionError(
            "y_train must contain both classes."
        )


def build_logistic_regression_pipeline(
    class_weights: dict[int, float],
    config: LogisticRegressionConfig | None = None,
) -> Pipeline:
    config = config or LogisticRegressionConfig()

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=config.C,
                    max_iter=config.max_iter,
                    random_state=config.random_state,
                    solver=config.solver,
                    class_weight=class_weights,
                ),
            ),
        ]
    )


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: LogisticRegressionConfig | None = None,
) -> LogisticRegressionModel:
    _validate_training_data(X_train, y_train)

    strategy = build_class_imbalance_strategy(y_train)

    pipeline = build_logistic_regression_pipeline(
        class_weights=strategy.sklearn_class_weight,
        config=config,
    )

    pipeline.fit(X_train, y_train)

    return LogisticRegressionModel(
        pipeline=pipeline,
        feature_columns=tuple(X_train.columns),
        config=config or LogisticRegressionConfig(),
        class_weights=strategy.sklearn_class_weight,
    )


def predict_logistic_regression(
    model: LogisticRegressionModel,
    X: pd.DataFrame,
) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise LogisticRegressionError("X must be a pandas DataFrame.")

    if tuple(X.columns) != model.feature_columns:
        raise LogisticRegressionError(
            "Prediction features do not match training features."
        )

    if X.isna().any().any():
        raise LogisticRegressionError(
            "Prediction features contain missing values."
        )

    return model.pipeline.predict(X)


def predict_logistic_regression_proba(
    model: LogisticRegressionModel,
    X: pd.DataFrame,
) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise LogisticRegressionError("X must be a pandas DataFrame.")

    if tuple(X.columns) != model.feature_columns:
        raise LogisticRegressionError(
            "Prediction features do not match training features."
        )

    if X.isna().any().any():
        raise LogisticRegressionError(
            "Prediction features contain missing values."
        )

    return model.pipeline.predict_proba(X)[:, 1]


def save_logistic_regression(
    model: LogisticRegressionModel,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

    return path


def load_logistic_regression(
    path: str | Path,
) -> LogisticRegressionModel:
    path = Path(path)

    if not path.exists():
        raise LogisticRegressionError(
            f"Model artifact does not exist: {path}"
        )

    model = joblib.load(path)

    if not isinstance(model, LogisticRegressionModel):
        raise LogisticRegressionError(
            "Invalid Logistic Regression model artifact."
        )

    return model