from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from api.contracts.prediction import PredictionResponse
from fraud_intelligence.ml.xgboost import (
    load_xgboost,
    predict_xgboost_proba,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "feature_dataset.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classical"
    / "xgboost_model.joblib"
)


class PredictionServiceError(RuntimeError):
    """Raised when prediction cannot be performed."""


@dataclass(frozen=True)
class PredictionService:
    """Service responsible for transaction prediction."""

    feature_dataset_path: Path = FEATURE_DATASET_PATH
    model_path: Path = MODEL_PATH

    def _load_features(self) -> pd.DataFrame:
        if not self.feature_dataset_path.exists():
            raise PredictionServiceError(
                f"Feature dataset not found: "
                f"{self.feature_dataset_path}"
            )

        try:
            return pd.read_parquet(
                self.feature_dataset_path
            )
        except Exception as exc:
            raise PredictionServiceError(
                "Failed to load feature dataset."
            ) from exc

    def _load_model(self):
        if not self.model_path.exists():
            raise PredictionServiceError(
                f"Model artifact not found: {self.model_path}"
            )

        try:
            return load_xgboost(self.model_path)
        except Exception as exc:
            raise PredictionServiceError(
                "Failed to load XGBoost model."
            ) from exc

    def predict(
        self,
        transaction_id: str,
    ) -> PredictionResponse:
        if not transaction_id:
            raise PredictionServiceError(
                "transaction_id must not be empty."
            )

        features = self._load_features()

        if "transaction_id" not in features.columns:
            raise PredictionServiceError(
                "Feature dataset does not contain "
                "transaction_id."
            )

        transaction = features[
            features["transaction_id"] == transaction_id
        ]

        if transaction.empty:
            raise KeyError(
                f"Transaction not found: {transaction_id}"
            )

        if len(transaction) > 1:
            raise PredictionServiceError(
                f"Multiple rows found for transaction: "
                f"{transaction_id}"
            )

        model = self._load_model()

        feature_columns = model.feature_columns

        missing_features = [
            column
            for column in feature_columns
            if column not in transaction.columns
        ]

        if missing_features:
            raise PredictionServiceError(
                "Feature dataset is missing model features: "
                + ", ".join(missing_features)
            )

        X = transaction.loc[
            :,
            list(feature_columns),
        ]

        try:
            probability = float(
                predict_xgboost_proba(
                    model,
                    X,
                )[0]
            )
        except Exception as exc:
            raise PredictionServiceError(
                "Model prediction failed."
            ) from exc

        prediction = int(probability >= 0.5)

        return PredictionResponse(
            transaction_id=transaction_id,
            prediction_probability=probability,
            prediction_label=prediction,
            model_name="xgboost",
            model_family="classical",
        )