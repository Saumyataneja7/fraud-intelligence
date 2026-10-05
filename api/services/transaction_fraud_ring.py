from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from api.contracts.common import EntityReferenceResponse
from api.contracts.fraud_ring import (
    FraudRingInvestigationResponse,
    RingEvidenceResponse,
    RingNetworkScoreResponse,
    RingTransactionResponse,
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
from fraud_intelligence.intelligence.ring_investigation import (
    assemble_fraud_ring_investigation_view,
)
from fraud_intelligence.intelligence.ring_network_scoring import (
    score_fraud_ring_candidate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "feature_dataset.parquet"
)


class TransactionFraudRingServiceError(RuntimeError):
    """Raised when transaction-scoped fraud-ring investigation fails."""


@dataclass(frozen=True)
class TransactionFraudRingService:
    feature_dataset_path: Path = FEATURE_DATASET_PATH

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
            raise TransactionFraudRingServiceError(
                "Unable to load the feature dataset."
            ) from exc

        if "transaction_id" not in dataframe.columns:
            raise TransactionFraudRingServiceError(
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
    ) -> FraudRingInvestigationResponse:
        transaction = self._load_transaction(
            transaction_id
        )

        relationships = build_direct_transaction_relationships(
            transaction_id=str(transaction_id),
            transaction_timestamp=transaction["timestamp"],
            transaction=transaction.to_dict(),
        )

        candidate_result = detect_fraud_ring_candidates(
            target_transaction_id=str(transaction_id),
            relationships=relationships,
        )

        if not candidate_result.candidates:
            raise KeyError(
                f"No candidate fraud ring found for "
                f"transaction {transaction_id!r}."
            )

        candidate = candidate_result.candidates[0]

        network_score = score_fraud_ring_candidate(
            candidate,
        )

        evidence = assemble_fraud_ring_evidence(
            candidate=candidate,
            network_score=network_score,
            target_transaction_id=str(transaction_id),
        )

        view = assemble_fraud_ring_investigation_view(
            candidate=candidate,
            network_score=network_score,
            evidence=evidence,
        )

        return self._to_response(view)

    @staticmethod
    def _to_response(view) -> FraudRingInvestigationResponse:
        evidence_items = (
            view.evidence.evidence_items
            if view.evidence is not None
            else ()
        )

        return FraudRingInvestigationResponse(
            candidate_id=view.candidate_id,
            entities=[
                EntityReferenceResponse(
                    entity_type=entity.entity_type,
                    entity_id=entity.entity_id,
                )
                for entity in view.entities
            ],
            relationships=[
                {
                    "relationship_type": relationship.relationship_type,
                    "source": {
                        "entity_type": relationship.source.entity_type,
                        "entity_id": relationship.source.entity_id,
                    },
                    "target": {
                        "entity_type": relationship.target.entity_type,
                        "entity_id": relationship.target.entity_id,
                    },
                    "timestamp": relationship.timestamp,
                }
                for relationship in view.relationships
            ],
            transactions=[
                RingTransactionResponse(
                    transaction_id=transaction.transaction_id,
                    timestamp=transaction.timestamp,
                )
                for transaction in view.transactions
            ],
            network_score=RingNetworkScoreResponse(
                candidate_id=view.network_score.candidate_id,
                entity_count=view.network_score.entity_count,
                relationship_count=view.network_score.relationship_count,
                relationship_density=(
                    view.network_score.relationship_density
                ),
                entity_type_count=(
                    view.network_score.entity_type_count
                ),
                relationship_type_count=(
                    view.network_score.relationship_type_count
                ),
                non_transaction_entity_count=(
                    view.network_score.non_transaction_entity_count
                ),
                non_transaction_connectivity=(
                    view.network_score.non_transaction_connectivity
                ),
                structural_score=(
                    view.network_score.structural_score
                ),
            ),
            evidence=[
                RingEvidenceResponse(
                    evidence_type=item.evidence_type,
                    description=item.description,
                    entity_types=list(item.entity_types),
                    relationship_types=list(
                        item.relationship_types
                    ),
                    strength=item.strength,
                )
                for item in evidence_items
            ],
            entity_type_counts=dict(
                view.entity_type_counts
            ),
            relationship_type_counts=dict(
                view.relationship_type_counts
            ),
            investigation_summary=view.investigation_summary,
            temporal_rule=view.temporal_rule,
        )
