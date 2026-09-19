from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from fraud_intelligence.explainability.contracts import (
    ExplainabilityContractError,
)
from fraud_intelligence.ml.random_forest import (
    RandomForestModel,
)
from fraud_intelligence.ml.xgboost import (
    XGBoostModel,
)


class SHAPExplanationError(ValueError):
    """Raised when SHAP explanation preparation fails."""


SUPPORTED_SHAP_MODELS = (
    RandomForestModel,
    XGBoostModel,
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


@dataclass(frozen=True)
class SHAPFeatureAttribution:
    """SHAP attribution for one feature."""

    feature: str
    shap_value: float
    absolute_shap_value: float
    rank: int


@dataclass(frozen=True)
class SHAPExplanation:
    """Local SHAP explanation for one transaction."""

    model_name: str
    feature_attributions: tuple[SHAPFeatureAttribution, ...]
    base_value: float
    model_output: float | None = None

    @property
    def feature_count(self) -> int:
        return len(self.feature_attributions)

    @property
    def top_feature(
        self,
    ) -> SHAPFeatureAttribution | None:
        if not self.feature_attributions:
            return None

        return self.feature_attributions[0]

    @property
    def positive_attributions(
        self,
    ) -> tuple[SHAPFeatureAttribution, ...]:
        return tuple(
            item
            for item in self.feature_attributions
            if item.shap_value > 0
        )

    @property
    def negative_attributions(
        self,
    ) -> tuple[SHAPFeatureAttribution, ...]:
        return tuple(
            item
            for item in self.feature_attributions
            if item.shap_value < 0
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "feature": item.feature,
                    "shap_value": item.shap_value,
                    "absolute_shap_value": (
                        item.absolute_shap_value
                    ),
                    "rank": item.rank,
                }
                for item in self.feature_attributions
            ]
        )


def _validate_model(
    model: object,
) -> None:
    if not isinstance(
        model,
        SUPPORTED_SHAP_MODELS,
    ):
        raise SHAPExplanationError(
            "SHAP TreeExplainer supports only "
            "RandomForestModel and XGBoostModel."
        )


def _validate_features(
    model: RandomForestModel | XGBoostModel,
    X: pd.DataFrame,
) -> None:
    if not isinstance(X, pd.DataFrame):
        raise SHAPExplanationError(
            "X must be a pandas DataFrame."
        )

    if X.empty:
        raise SHAPExplanationError(
            "X cannot be empty."
        )

    if len(X) != 1:
        raise SHAPExplanationError(
            "SHAP transaction explanation expects exactly "
            "one transaction row."
        )

    if tuple(X.columns) != model.feature_columns:
        raise SHAPExplanationError(
            "Explanation features do not match the model's "
            "training feature_columns."
        )

    if X.isna().any().any():
        raise SHAPExplanationError(
            "Explanation features contain missing values."
        )

    forbidden = [
        column
        for column in X.columns
        if column in FORBIDDEN_FEATURES
    ]

    if forbidden:
        raise ExplainabilityContractError(
            "SHAP explanations cannot use forbidden "
            f"columns: {forbidden}."
        )

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise SHAPExplanationError(
            "Explanation features must contain only finite values."
        )


def _extract_single_shap_row(
    shap_values: object,
    feature_count: int,
) -> np.ndarray:
    """
    Normalize SHAP output across supported SHAP versions.

    Binary classifiers may return:
    - ndarray of shape (n_samples, n_features)
    - ndarray with an additional output dimension
    - list containing class-specific arrays
    """
    if isinstance(shap_values, list):
        if not shap_values:
            raise SHAPExplanationError(
                "SHAP returned an empty explanation."
            )

        values = np.asarray(
            shap_values[-1],
            dtype=float,
        )
    else:
        values = np.asarray(
            shap_values,
            dtype=float,
        )

    if values.ndim == 1:
        row = values

    elif values.ndim == 2:
        if values.shape[0] != 1:
            raise SHAPExplanationError(
                "Expected SHAP output for exactly one row."
            )

        row = values[0]

    elif values.ndim == 3:
        if values.shape[0] != 1:
            raise SHAPExplanationError(
                "Expected SHAP output for exactly one row."
            )

        if values.shape[2] < 1:
            raise SHAPExplanationError(
                "SHAP output contains no model outputs."
            )

        row = values[0, :, -1]

    else:
        raise SHAPExplanationError(
            "Unsupported SHAP output shape: "
            f"{values.shape}."
        )

    row = np.asarray(
        row,
        dtype=float,
    )

    if row.ndim != 1:
        raise SHAPExplanationError(
            "Normalized SHAP values must be one-dimensional."
        )

    if len(row) != feature_count:
        raise SHAPExplanationError(
            "Number of SHAP values does not match "
            "number of model features."
        )

    if not np.isfinite(row).all():
        raise SHAPExplanationError(
            "SHAP values must contain only finite values."
        )

    return row


def _extract_base_value(
    expected_value: object,
) -> float:
    values = np.asarray(
        expected_value,
        dtype=float,
    ).reshape(-1)

    if values.size == 0:
        raise SHAPExplanationError(
            "SHAP returned an empty base value."
        )

    value = float(values[-1])

    if not np.isfinite(value):
        raise SHAPExplanationError(
            "SHAP base value must be finite."
        )

    return value


def _build_explanation(
    *,
    model_name: str,
    feature_names: tuple[str, ...],
    shap_values: np.ndarray,
    base_value: float,
    model_output: float | None,
) -> SHAPExplanation:
    order = np.argsort(
        -np.abs(shap_values),
        kind="mergesort",
    )

    attributions: list[SHAPFeatureAttribution] = []

    for rank, index in enumerate(
        order,
        start=1,
    ):
        value = float(shap_values[index])

        attributions.append(
            SHAPFeatureAttribution(
                feature=feature_names[index],
                shap_value=value,
                absolute_shap_value=abs(value),
                rank=rank,
            )
        )

    return SHAPExplanation(
        model_name=model_name,
        feature_attributions=tuple(attributions),
        base_value=base_value,
        model_output=model_output,
    )


def explain_transaction_with_shap(
    model: RandomForestModel | XGBoostModel,
    X: pd.DataFrame,
) -> SHAPExplanation:
    """
    Generate a local SHAP explanation for one transaction.

    SHAP TreeExplainer is used against the frozen Phase 5
    tree-based model classifier.
    """
    _validate_model(model)
    _validate_features(model, X)

    if isinstance(model, RandomForestModel):
        classifier = model.classifier
        model_name = "RandomForest"

    else:
        classifier = model.classifier
        model_name = "XGBoost"

    try:
        explainer = shap.TreeExplainer(
            classifier
        )

        explanation = explainer.shap_values(X)

        values = _extract_single_shap_row(
            explanation,
            feature_count=len(model.feature_columns),
        )

        base_value = _extract_base_value(
            explainer.expected_value
        )

    except SHAPExplanationError:
        raise

    except Exception as exc:
        raise SHAPExplanationError(
            "SHAP explanation failed."
        ) from exc

    model_output = None

    try:
        model_output = float(
            classifier.predict_proba(X)[0, 1]
        )
    except Exception:
        model_output = None

    return _build_explanation(
        model_name=model_name,
        feature_names=model.feature_columns,
        shap_values=values,
        base_value=base_value,
        model_output=model_output,
    )


def validate_shap_explanation(
    result: SHAPExplanation,
) -> None:
    if not isinstance(
        result,
        SHAPExplanation,
    ):
        raise SHAPExplanationError(
            "result must be a SHAPExplanation."
        )

    if not result.model_name:
        raise SHAPExplanationError(
            "model_name cannot be empty."
        )

    if not np.isfinite(result.base_value):
        raise SHAPExplanationError(
            "base_value must be finite."
        )

    if result.model_output is not None:
        if not np.isfinite(result.model_output):
            raise SHAPExplanationError(
                "model_output must be finite."
            )

        if not 0 <= result.model_output <= 1:
            raise SHAPExplanationError(
                "model_output must be between 0 and 1."
            )

    seen_features: set[str] = set()
    seen_ranks: set[int] = set()

    for item in result.feature_attributions:
        if not item.feature:
            raise SHAPExplanationError(
                "Feature name cannot be empty."
            )

        if item.feature in FORBIDDEN_FEATURES:
            raise ExplainabilityContractError(
                "SHAP result contains a forbidden feature: "
                f"{item.feature}."
            )

        if item.feature in seen_features:
            raise SHAPExplanationError(
                "Duplicate feature in SHAP explanation."
            )

        if item.rank in seen_ranks:
            raise SHAPExplanationError(
                "Duplicate rank in SHAP explanation."
            )

        if item.rank < 1:
            raise SHAPExplanationError(
                "SHAP rank must be at least 1."
            )

        if not np.isfinite(item.shap_value):
            raise SHAPExplanationError(
                "SHAP value must be finite."
            )

        if not np.isfinite(
            item.absolute_shap_value
        ):
            raise SHAPExplanationError(
                "Absolute SHAP value must be finite."
            )

        if item.absolute_shap_value < 0:
            raise SHAPExplanationError(
                "Absolute SHAP value cannot be negative."
            )

        if not np.isclose(
            item.absolute_shap_value,
            abs(item.shap_value),
        ):
            raise SHAPExplanationError(
                "Absolute SHAP value must equal the "
                "absolute value of shap_value."
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
        raise SHAPExplanationError(
            "SHAP ranks must be contiguous."
        )