from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from api.contracts.explanation import (
    ExplanationResponse,
    FeatureAttributionResponse,
    GraphFindingResponse,
)
from api.services.prediction import (
    FEATURE_DATASET_PATH,
    MODEL_PATH,
)

from fraud_intelligence.explainability.classical import (
    calculate_classical_attributions,
)
from fraud_intelligence.explainability.human_readable import (
    build_human_readable_explanation,
)
from fraud_intelligence.explainability.shap import (
    explain_transaction_with_shap,
)
from fraud_intelligence.explainability.transaction import (
    PredictionEvidence,
    assemble_transaction_explanation,
)
from fraud_intelligence.ml.xgboost import (
    load_xgboost,
    predict_xgboost_proba,
)


class ExplanationServiceError(RuntimeError):
    """Raised when transaction explanation cannot be completed."""


@dataclass(frozen=True)
class ExplanationService:
    """
    API adapter for the frozen Phase 8 explainability layer.

    The service performs only orchestration and contract mapping.
    Explanation logic remains owned by Phase 8.
    """

    feature_dataset_path: Path = FEATURE_DATASET_PATH
    model_path: Path = MODEL_PATH

    def _load_features(self) -> pd.DataFrame:
        if not self.feature_dataset_path.exists():
            raise ExplanationServiceError(
                f"Feature dataset not found: "
                f"{self.feature_dataset_path}"
            )

        try:
            return pd.read_parquet(
                self.feature_dataset_path
            )
        except Exception as exc:
            raise ExplanationServiceError(
                "Failed to load feature dataset."
            ) from exc

    def _load_model(self):
        if not self.model_path.exists():
            raise ExplanationServiceError(
                f"Model artifact not found: {self.model_path}"
            )

        try:
            return load_xgboost(self.model_path)
        except Exception as exc:
            raise ExplanationServiceError(
                "Failed to load XGBoost model."
            ) from exc

    def _load_transaction(
        self,
        transaction_id: str,
        features: pd.DataFrame,
    ) -> pd.Series:
        if not transaction_id:
            raise ValueError(
                "transaction_id must be non-empty."
            )

        if "transaction_id" not in features.columns:
            raise ExplanationServiceError(
                "Feature dataset does not contain "
                "transaction_id."
            )

        matches = features[
            features["transaction_id"].astype(str)
            == str(transaction_id)
        ]

        if matches.empty:
            raise KeyError(
                f"Transaction {transaction_id!r} was not found."
            )

        if len(matches) > 1:
            raise ExplanationServiceError(
                f"Multiple rows found for transaction: "
                f"{transaction_id}"
            )

        return matches.iloc[0]

    def explain(
        self,
        transaction_id: str,
    ) -> ExplanationResponse:
        """
        Build a human-readable explanation for one transaction.

        Phase 8 remains the source of truth for explanation logic.
        """

        features = self._load_features()

        transaction = self._load_transaction(
            transaction_id,
            features,
        )

        model = self._load_model()

        feature_columns = tuple(model.feature_columns)

        missing_features = [
            column
            for column in feature_columns
            if column not in transaction.index
        ]

        if missing_features:
            raise ExplanationServiceError(
                "Feature dataset is missing model features: "
                + ", ".join(missing_features)
            )

        X = transaction.loc[
            list(feature_columns)
        ].to_frame().T

        try:
            probability = float(
                predict_xgboost_proba(
                    model,
                    X,
                )[0]
            )
        except Exception as exc:
            raise ExplanationServiceError(
                "Model prediction failed."
            ) from exc

        prediction_label = int(
            probability >= 0.5
        )

        timestamp = transaction["timestamp"]

        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)

        prediction = PredictionEvidence(
            transaction_id=str(transaction_id),
            probability=probability,
            prediction=prediction_label,
            threshold=0.5,
            model_name="XGBoost",
        )

        try:
            classical_attribution = (
                calculate_classical_attributions(
                    model
                )
            )

            shap_explanation = (
                explain_transaction_with_shap(
                    model,
                    X,
                )
            )

            transaction_explanation = (
                assemble_transaction_explanation(
                    transaction_id=str(transaction_id),
                    timestamp=timestamp,
                    prediction=prediction,
                    classical_attribution=(
                        classical_attribution
                    ),
                    shap_explanation=(
                        shap_explanation
                    ),
                    graph_neighborhood=None,
                    gnn_importance=None,
                )
            )

            human_readable = (
                build_human_readable_explanation(
                    transaction_explanation
                )
            )

        except Exception as exc:
            raise ExplanationServiceError(
                "Unable to build transaction explanation."
            ) from exc

        feature_attributions = []

        # SHAP is transaction-specific and therefore is the
        # authoritative local attribution exposed by this endpoint.
        if human_readable.reasons:
            for reason in human_readable.reasons:
                feature_attributions.append(
                    FeatureAttributionResponse(
                        feature=reason.feature,
                        attribution=reason.attribution,
                        absolute_attribution=abs(
                            reason.attribution
                        ),
                        rank=reason.rank,
                    )
                )

        graph_findings = [
            GraphFindingResponse(
                finding_type=finding.category,
                description=finding.text,
            )
            for finding in human_readable.graph_findings
        ]

        return ExplanationResponse(
            transaction_id=human_readable.transaction_id,
            prediction_probability=probability,
            prediction_label=prediction_label,
            feature_attributions=feature_attributions,
            graph_findings=graph_findings,
            summary=human_readable.summary,
        )
