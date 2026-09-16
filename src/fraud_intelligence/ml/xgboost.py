from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraud_intelligence.ml.imbalance import (
    build_class_imbalance_strategy,
)


class XGBoostError(ValueError):
    """Raised when XGBoost preparation or training fails."""


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    eval_metric: str = "logloss"


@dataclass(frozen=True)
class XGBoostModel:
    classifier: XGBClassifier
    feature_columns: tuple[str, ...]
    config: XGBoostConfig
    scale_pos_weight: float


def _validate_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    if not isinstance(X_train, pd.DataFrame):
        raise XGBoostError(
            "X_train must be a pandas DataFrame."
        )

    if not isinstance(y_train, pd.Series):
        raise XGBoostError(
            "y_train must be a pandas Series."
        )

    if X_train.empty:
        raise XGBoostError(
            "X_train cannot be empty."
        )

    if len(X_train) != len(y_train):
        raise XGBoostError(
            "X_train and y_train must have the same number of rows."
        )

    if X_train.isna().any().any():
        raise XGBoostError(
            "X_train contains missing values."
        )

    if y_train.isna().any():
        raise XGBoostError(
            "y_train contains missing values."
        )

    if not set(y_train.unique()).issubset({0, 1}):
        raise XGBoostError(
            "y_train must contain only binary values 0 and 1."
        )

    if y_train.nunique() != 2:
        raise XGBoostError(
            "y_train must contain both classes."
        )


def build_xgboost_classifier(
    scale_pos_weight: float,
    config: XGBoostConfig | None = None,
) -> XGBClassifier:
    config = config or XGBoostConfig()

    return XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        eval_metric=config.eval_metric,
        scale_pos_weight=scale_pos_weight,
    )


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: XGBoostConfig | None = None,
) -> XGBoostModel:
    _validate_training_data(X_train, y_train)

    strategy = build_class_imbalance_strategy(y_train)

    classifier = build_xgboost_classifier(
        scale_pos_weight=strategy.xgboost_scale_pos_weight,
        config=config,
    )

    classifier.fit(X_train, y_train)

    return XGBoostModel(
        classifier=classifier,
        feature_columns=tuple(X_train.columns),
        config=config or XGBoostConfig(),
        scale_pos_weight=strategy.xgboost_scale_pos_weight,
    )


def _validate_prediction_data(
    model: XGBoostModel,
    X: pd.DataFrame,
) -> None:
    if not isinstance(X, pd.DataFrame):
        raise XGBoostError(
            "X must be a pandas DataFrame."
        )

    if tuple(X.columns) != model.feature_columns:
        raise XGBoostError(
            "Prediction features do not match training features."
        )

    if X.isna().any().any():
        raise XGBoostError(
            "Prediction features contain missing values."
        )


def predict_xgboost(
    model: XGBoostModel,
    X: pd.DataFrame,
) -> np.ndarray:
    _validate_prediction_data(model, X)

    return model.classifier.predict(X)


def predict_xgboost_proba(
    model: XGBoostModel,
    X: pd.DataFrame,
) -> np.ndarray:
    _validate_prediction_data(model, X)

    return model.classifier.predict_proba(X)[:, 1]


def save_xgboost(
    model: XGBoostModel,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

    return path


def load_xgboost(
    path: str | Path,
) -> XGBoostModel:
    path = Path(path)

    if not path.exists():
        raise XGBoostError(
            f"Model artifact does not exist: {path}"
        )

    model = joblib.load(path)

    if not isinstance(model, XGBoostModel):
        raise XGBoostError(
            "Invalid XGBoost model artifact."
        )

    return model