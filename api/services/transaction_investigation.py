from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from api.contracts.transaction import (
    SuspicionSignalResponse,
    TransactionInvestigationResponse,
    TransactionRelationshipResponse,
)
from api.contracts.common import EntityReferenceResponse
from api.services.prediction import (
    MODEL_PATH,
    FEATURE_DATASET_PATH,
    PredictionService,
    PredictionServiceError,
)

from fraud_intelligence.intelligence.entity_relationships import (
    build_direct_transaction_relationships,
)
from fraud_intelligence.intelligence.fraud_ring_candidates import (
    detect_fraud_ring_candidates,
)
from fraud_intelligence.intelligence.fraud_ring_evidence import (
    assemble_fraud_ring_evidence,
)
from fraud_intelligence.intelligence.ring_network_scoring import (
    score_fraud_ring_candidates,
)
from fraud_intelligence.intelligence.suspicious_transaction import (
    assemble_suspicious_transaction_intelligence,
)
from fraud_intelligence.intelligence.transaction_investigation import (
    assemble_transaction_investigation_view,
)


class TransactionInvestigationServiceError(RuntimeError):
    """Raised when transaction investigation cannot be completed."""


@dataclass(frozen=True)
class TransactionInvestigationService:
    """Service adapter for the frozen Phase 9 investigation layer."""

    feature_dataset_path: Path = FEATURE_DATASET_PATH
    prediction_service: PredictionService = PredictionService()

    def _load_transaction(
        self,
        transaction_id: str,
    ) -> pd.Series:
        if not transaction_id:
            raise ValueError(
                "transaction_id must be non-empty."
            )

        try:
            dataframe = pd.read_parquet(
                self.feature_dataset_path
            )
        except Exception as exc:
            raise TransactionInvestigationServiceError(
                "Unable to load the feature dataset."
            ) from exc

        if "transaction_id" not in dataframe.columns:
            raise TransactionInvestigationServiceError(
                "Feature dataset does not contain transaction_id."
            )

        matches = dataframe[
            dataframe["transaction_id"].astype(str)
            == str(transaction_id)
        ]

        if matches.empty:
            raise KeyError(
                f"Transaction {transaction_id!r} was not found."
            )

        return matches.iloc[0]

    def investigate(
        self,
        transaction_id: str,
    ) -> TransactionInvestigationResponse:
        """
        Build the complete transaction investigation response.

        Phase 9 remains the source of truth for investigation logic.
        This service only adapts persisted transaction data and model
        predictions into the frozen intelligence interfaces.
        """

        transaction = self._load_transaction(transaction_id)

        try:
            prediction = self.prediction_service.predict(
                transaction_id
            )

            timestamp = transaction["timestamp"]

            # ------------------------------------------------------------------
            # Suspicious transaction intelligence
            # ------------------------------------------------------------------

            transaction_features = {
                key: value
                for key, value in transaction.to_dict().items()
                if key
                not in {
                    "transaction_id",
                    "timestamp",
                    "is_fraud",
                    "fraud_scenario",
                }
            }

            entity_fields = {
                "customer": "customer_id",
                "account": "account_id",
                "card": "card_id",
                "merchant": "merchant_id",
                "device": "device_id",
                "ip": "ip_id",
            }

            entity_counts = {
                entity_type: 1
                for entity_type, field_name in entity_fields.items()
                if field_name in transaction.index
                and pd.notna(transaction[field_name])
            }

            suspicious_transaction = (
                assemble_suspicious_transaction_intelligence(
                    transaction_id=str(transaction_id),
                    timestamp=timestamp,
                    prediction_probability=(
                        prediction.prediction_probability
                    ),
                    prediction_label=prediction.prediction_label,
                    transaction_features=transaction_features,
                    entity_counts=entity_counts,
                    context_timestamps=None,
                )
            )

            # ------------------------------------------------------------------
            # Direct transaction relationships
            # ------------------------------------------------------------------

            relationships = build_direct_transaction_relationships(
                transaction_id=str(transaction_id),
                transaction_timestamp=timestamp,
                transaction=transaction.to_dict(),
            )

            # ------------------------------------------------------------------
            # Structural fraud-ring candidates
            # ------------------------------------------------------------------

            candidate_result = detect_fraud_ring_candidates(
                target_transaction_id=str(transaction_id),
                relationships=relationships,
            )

            candidates = candidate_result.candidates

            # ------------------------------------------------------------------
            # Structural network scores
            # ------------------------------------------------------------------

            network_scores = score_fraud_ring_candidates(
                candidates
            )

            # ------------------------------------------------------------------
            # Ring evidence
            # ------------------------------------------------------------------

            score_by_id = {
                score.candidate_id: score
                for score in network_scores
            }

            ring_evidence = tuple(
                assemble_fraud_ring_evidence(
                    candidate=candidate,
                    network_score=score_by_id[
                        candidate.candidate_id
                    ],
                    target_transaction_id=str(transaction_id),
                )
                for candidate in candidates
                if candidate.candidate_id in score_by_id
            )

            # ------------------------------------------------------------------
            # Frozen Phase 9 transaction investigation view
            # ------------------------------------------------------------------

            investigation_view = (
                assemble_transaction_investigation_view(
                    transaction_id=str(transaction_id),
                    timestamp=timestamp,
                    suspicious_transaction=suspicious_transaction,
                    relationships=relationships,
                    candidates=candidates,
                    network_scores=network_scores,
                    ring_evidence=ring_evidence,
                )
            )

        except KeyError:
            raise
        except PredictionServiceError:
            raise
        except Exception as exc:
            raise TransactionInvestigationServiceError(
                "Unable to build transaction investigation."
            ) from exc

        # ----------------------------------------------------------------------
        # API contract mapping
        #
        # The Phase 10.1 response contract is frozen. It intentionally exposes
        # the core investigation fields currently defined there.
        # ----------------------------------------------------------------------

        return TransactionInvestigationResponse(
            transaction_id=investigation_view.transaction_id,
            timestamp=investigation_view.timestamp,
            prediction_probability=(
                investigation_view.prediction_probability
            ),
            prediction_label=investigation_view.prediction_label,
            suspicion_signals=[
                SuspicionSignalResponse(
                    name=signal.name,
                    value=signal.value,
                    description=signal.description,
                    source=signal.source,
                )
                for signal in investigation_view.suspicion_signals
            ],
            related_entities=[
                EntityReferenceResponse(
                    entity_type=entity.entity_type,
                    entity_id=entity.entity_id,
                )
                for entity in investigation_view.related_entities
            ],
            relationships=[
                TransactionRelationshipResponse(
                    relationship_type=relationship.relationship_type,
                    source=EntityReferenceResponse(
                        entity_type=relationship.source.entity_type,
                        entity_id=relationship.source.entity_id,
                    ),
                    target=EntityReferenceResponse(
                        entity_type=relationship.target.entity_type,
                        entity_id=relationship.target.entity_id,
                    ),
                    timestamp=relationship.timestamp,
                )
                for relationship in investigation_view.relationships
            ],
            candidate_ring_ids=list(
                investigation_view.candidate_ring_ids
            ),
            investigation_summary=(
                investigation_view.investigation_summary
            ),
            temporal_rule=investigation_view.temporal_rule,
        )