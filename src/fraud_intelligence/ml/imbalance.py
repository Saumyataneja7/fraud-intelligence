from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


TARGET_COLUMN = "is_fraud"


class ClassImbalanceError(ValueError):
    """Raised when class imbalance configuration is invalid."""


@dataclass(frozen=True)
class ClassDistribution:
    """Summary of the binary fraud target distribution."""

    total_rows: int
    negative_count: int
    positive_count: int
    negative_rate: float
    positive_rate: float

    @property
    def imbalance_ratio(self) -> float:
        """Return negative-to-positive class ratio."""

        if self.positive_count == 0:
            return float("inf")

        return self.negative_count / self.positive_count


@dataclass(frozen=True)
class ClassImbalanceStrategy:
    """Class imbalance strategy derived from training data."""

    class_weights: dict[int, float]
    scale_pos_weight: float
    distribution: ClassDistribution

    @property
    def sklearn_class_weight(self) -> dict[int, float]:
        """Return class weights suitable for sklearn estimators."""

        return dict(self.class_weights)

    @property
    def xgboost_scale_pos_weight(self) -> float:
        """Return positive-class weight for XGBoost-style models."""

        return self.scale_pos_weight


def _validate_target(target: pd.Series) -> None:
    """Validate that the target is a non-empty binary series."""

    if not isinstance(target, pd.Series):
        raise ClassImbalanceError(
            "Target must be provided as a pandas Series."
        )

    if target.empty:
        raise ClassImbalanceError(
            "Target cannot be empty."
        )

    if target.isna().any():
        raise ClassImbalanceError(
            "Target contains missing values."
        )

    unique_values = set(target.unique())

    if not unique_values.issubset({0, 1}):
        raise ClassImbalanceError(
            "Target must contain only binary 0/1 values. "
            f"Found: {sorted(unique_values)}"
        )

    if len(unique_values) < 2:
        raise ClassImbalanceError(
            "Target must contain both negative and positive classes."
        )


def calculate_class_distribution(
    target: pd.Series,
) -> ClassDistribution:
    """Calculate the class distribution for a target series."""

    _validate_target(target)

    negative_count = int((target == 0).sum())
    positive_count = int((target == 1).sum())
    total_rows = len(target)

    return ClassDistribution(
        total_rows=total_rows,
        negative_count=negative_count,
        positive_count=positive_count,
        negative_rate=negative_count / total_rows,
        positive_rate=positive_count / total_rows,
    )


def calculate_class_weights(
    target: pd.Series,
) -> dict[int, float]:
    """
    Calculate balanced class weights from a training target.

    The calculation follows sklearn's balanced class-weight formula.
    """

    _validate_target(target)

    classes = np.array([0, 1])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=target.to_numpy(),
    )

    return {
        int(class_value): float(weight)
        for class_value, weight in zip(classes, weights)
    }


def calculate_scale_pos_weight(
    target: pd.Series,
) -> float:
    """
    Calculate XGBoost-style positive-class weight.

    scale_pos_weight = negative_count / positive_count
    """

    distribution = calculate_class_distribution(target)

    if distribution.positive_count == 0:
        raise ClassImbalanceError(
            "Cannot calculate scale_pos_weight without positive samples."
        )

    return distribution.imbalance_ratio


def build_class_imbalance_strategy(
    y_train: pd.Series,
) -> ClassImbalanceStrategy:
    """
    Build the class imbalance strategy using training data only.

    Validation and test targets must never be used to calculate weights.
    """

    distribution = calculate_class_distribution(y_train)

    class_weights = calculate_class_weights(y_train)

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    return ClassImbalanceStrategy(
        class_weights=class_weights,
        scale_pos_weight=scale_pos_weight,
        distribution=distribution,
    )


def build_class_imbalance_summary(
    strategy: ClassImbalanceStrategy,
) -> dict:
    """Build serializable metadata describing the imbalance strategy."""

    distribution = strategy.distribution

    return {
        "strategy": "training-set class weighting",
        "target_column": TARGET_COLUMN,
        "total_training_rows": distribution.total_rows,
        "negative_count": distribution.negative_count,
        "positive_count": distribution.positive_count,
        "negative_rate": distribution.negative_rate,
        "positive_rate": distribution.positive_rate,
        "imbalance_ratio": distribution.imbalance_ratio,
        "class_weights": {
            str(key): value
            for key, value in strategy.class_weights.items()
        },
        "scale_pos_weight": strategy.scale_pos_weight,
        "validation_rebalanced": False,
        "test_rebalanced": False,
    }