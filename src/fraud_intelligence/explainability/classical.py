from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_intelligence.explainability.contracts import (
    ExplainabilityContractError,
)
from fraud_intelligence.ml.logistic_regression import (
    LogisticRegressionModel,
)
from fraud_intelligence.ml.random_forest import (
    RandomForestModel,
)
from fraud_intelligence.ml.xgboost import (
    XGBoostModel,
)


class ClassicalAttributionError(ValueError):
    """Raised when classical model attribution fails."""


@dataclass(frozen=True)
class FeatureAttribution:
    """Attribution for one model feature."""

    feature: str
    attribution: float
    absolute_attribution: float
    rank: int


@dataclass(frozen=True)
class ClassicalAttributionResult:
    """Global feature attribution for a classical ML model."""

    model_name: str
    model_family: str
    feature_attributions: tuple[FeatureAttribution, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_attributions)

    @property
    def top_feature(self) -> FeatureAttribution | None:
        if not self.feature_attributions:
            return None

        return self.feature_attributions[0]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "feature": item.feature,
                    "attribution": item.attribution,
                    "absolute_attribution": item.absolute_attribution,
                    "rank": item.rank,
                }
                for item in self.feature_attributions
            ]
        )


FORBIDDEN_FEATURES = frozenset(
    {
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
    }
)


def _validate_feature_names(
    feature_names: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    names = tuple(feature_names)

    if not names:
        raise ClassicalAttributionError(
            "feature_names cannot be empty."
        )

    if any(
        not isinstance(name, str) or not name
        for name in names
    ):
        raise ClassicalAttributionError(
            "feature_names must contain non-empty strings."
        )

    if len(set(names)) != len(names):
        raise ClassicalAttributionError(
            "feature_names must not contain duplicates."
        )

    violations = [
        name
        for name in names
        if name in FORBIDDEN_FEATURES
    ]

    if violations:
        raise ExplainabilityContractError(
            "Classical explanations cannot use forbidden "
            f"columns: {violations}."
        )

    return names


def _validate_model_feature_names(
    model_feature_names: tuple[str, ...],
    supplied_feature_names: tuple[str, ...],
) -> None:
    if model_feature_names != supplied_feature_names:
        raise ClassicalAttributionError(
            "feature_names do not match the model's "
            "training feature_columns."
        )


def _validate_importances(
    importances: np.ndarray,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    values = np.asarray(
        importances,
        dtype=float,
    )

    if values.ndim != 1:
        raise ClassicalAttributionError(
            "Feature importances must be one-dimensional."
        )

    if len(values) != len(feature_names):
        raise ClassicalAttributionError(
            "Number of feature importances must match "
            "number of feature names."
        )

    if not np.isfinite(values).all():
        raise ClassicalAttributionError(
            "Feature importances must contain only finite values."
        )

    return values


def _build_result(
    *,
    model_name: str,
    model_family: str,
    feature_names: tuple[str, ...],
    attributions: np.ndarray,
) -> ClassicalAttributionResult:
    order = np.argsort(
        -np.abs(attributions),
        kind="mergesort",
    )

    results: list[FeatureAttribution] = []

    for rank, index in enumerate(order, start=1):
        value = float(attributions[index])

        results.append(
            FeatureAttribution(
                feature=feature_names[index],
                attribution=value,
                absolute_attribution=abs(value),
                rank=rank,
            )
        )

    return ClassicalAttributionResult(
        model_name=model_name,
        model_family=model_family,
        feature_attributions=tuple(results),
    )


def _logistic_attributions(
    model: LogisticRegressionModel,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    _validate_model_feature_names(
        model.feature_columns,
        feature_names,
    )

    classifier = model.pipeline.named_steps.get(
        "classifier"
    )

    if classifier is None:
        raise ClassicalAttributionError(
            "Logistic Regression pipeline does not contain "
            "a 'classifier' step."
        )

    if not hasattr(classifier, "coef_"):
        raise ClassicalAttributionError(
            "Logistic Regression classifier is not fitted."
        )

    coefficients = np.asarray(
        classifier.coef_,
        dtype=float,
    )

    if coefficients.ndim != 2:
        raise ClassicalAttributionError(
            "Expected a two-dimensional coefficient matrix."
        )

    if coefficients.shape[0] != 1:
        raise ClassicalAttributionError(
            "Expected binary Logistic Regression."
        )

    return _validate_importances(
        coefficients[0],
        feature_names,
    )


def _random_forest_attributions(
    model: RandomForestModel,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    _validate_model_feature_names(
        model.feature_columns,
        feature_names,
    )

    classifier = model.classifier

    if not hasattr(
        classifier,
        "feature_importances_",
    ):
        raise ClassicalAttributionError(
            "Random Forest classifier is not fitted."
        )

    return _validate_importances(
        classifier.feature_importances_,
        feature_names,
    )


def _xgboost_attributions(
    model: XGBoostModel,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    _validate_model_feature_names(
        model.feature_columns,
        feature_names,
    )

    classifier = model.classifier

    if not hasattr(
        classifier,
        "feature_importances_",
    ):
        raise ClassicalAttributionError(
            "XGBoost classifier is not fitted."
        )

    return _validate_importances(
        classifier.feature_importances_,
        feature_names,
    )


def calculate_classical_attributions(
    model: (
        LogisticRegressionModel
        | RandomForestModel
        | XGBoostModel
    ),
    feature_names: list[str] | tuple[str, ...] | None = None,
) -> ClassicalAttributionResult:
    """
    Calculate global model-native feature attribution.

    Logistic Regression:
        signed classifier coefficients.

    Random Forest:
        native feature_importances_.

    XGBoost:
        native feature_importances_.

    The model's frozen feature_columns are authoritative.
    """
    if isinstance(model, LogisticRegressionModel):
        model_features = model.feature_columns

        names = _validate_feature_names(
            feature_names
            if feature_names is not None
            else model_features
        )

        values = _logistic_attributions(
            model,
            names,
        )

        return _build_result(
            model_name="LogisticRegression",
            model_family="classical_ml",
            feature_names=names,
            attributions=values,
        )

    if isinstance(model, RandomForestModel):
        model_features = model.feature_columns

        names = _validate_feature_names(
            feature_names
            if feature_names is not None
            else model_features
        )

        values = _random_forest_attributions(
            model,
            names,
        )

        return _build_result(
            model_name="RandomForest",
            model_family="classical_ml",
            feature_names=names,
            attributions=values,
        )

    if isinstance(model, XGBoostModel):
        model_features = model.feature_columns

        names = _validate_feature_names(
            feature_names
            if feature_names is not None
            else model_features
        )

        values = _xgboost_attributions(
            model,
            names,
        )

        return _build_result(
            model_name="XGBoost",
            model_family="classical_ml",
            feature_names=names,
            attributions=values,
        )

    raise ClassicalAttributionError(
        "Unsupported classical model type: "
        f"{type(model).__name__}."
    )


def validate_classical_attribution(
    result: ClassicalAttributionResult,
) -> None:
    if not isinstance(
        result,
        ClassicalAttributionResult,
    ):
        raise ClassicalAttributionError(
            "result must be a ClassicalAttributionResult."
        )

    if not result.model_name:
        raise ClassicalAttributionError(
            "model_name cannot be empty."
        )

    if result.model_family != "classical_ml":
        raise ClassicalAttributionError(
            "model_family must be 'classical_ml'."
        )

    seen_features: set[str] = set()
    seen_ranks: set[int] = set()

    for item in result.feature_attributions:
        if not item.feature:
            raise ClassicalAttributionError(
                "Attribution feature cannot be empty."
            )

        if item.feature in seen_features:
            raise ClassicalAttributionError(
                "Duplicate feature in attribution result."
            )

        if item.rank in seen_ranks:
            raise ClassicalAttributionError(
                "Duplicate attribution rank."
            )

        if item.rank < 1:
            raise ClassicalAttributionError(
                "Attribution rank must be at least 1."
            )

        if not np.isfinite(item.attribution):
            raise ClassicalAttributionError(
                "Attribution must be finite."
            )

        if not np.isfinite(
            item.absolute_attribution
        ):
            raise ClassicalAttributionError(
                "Absolute attribution must be finite."
            )

        if item.absolute_attribution < 0:
            raise ClassicalAttributionError(
                "Absolute attribution cannot be negative."
            )

        if not np.isclose(
            item.absolute_attribution,
            abs(item.attribution),
        ):
            raise ClassicalAttributionError(
                "Absolute attribution must equal the absolute "
                "value of attribution."
            )

        seen_features.add(item.feature)
        seen_ranks.add(item.rank)

    expected_ranks = set(
        range(
            1,
            len(result.feature_attributions) + 1,
        )
    )

    if seen_ranks != expected_ranks:
        raise ClassicalAttributionError(
            "Attribution ranks must be contiguous."
        )