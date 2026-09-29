from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from api.contracts.fraud_ring import (
    FraudRingInvestigationResponse,
    RingEvidenceResponse,
    RingNetworkScoreResponse,
    RingTransactionResponse,
)
from api.contracts.common import EntityReferenceResponse
from fraud_intelligence.intelligence.entity_relationships import (
    build_direct_transaction_relationships,
)
from fraud_intelligence.intelligence.fraud_ring_candidates import (
    detect_connected_candidates,
)
from fraud_intelligence.intelligence.fraud_ring_evidence import (
    assemble_fraud_ring_evidence,
)
from fraud_intelligence.intelligence.ring_investigation import (
    FraudRingInvestigationView,
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


class FraudRingInvestigationServiceError(RuntimeError):
    """Raised when fraud-ring investigation cannot be completed."""


@dataclass(frozen=True)
class FraudRingInvestigationService:
    feature_dataset_path: Path = FEATURE_DATASET_PATH

    def _load_dataset(self) -> pd.DataFrame:
        try:
            return pd.read_parquet(self.feature_dataset_path)
        except Exception as exc:
            raise FraudRingInvestigationServiceError(
                "Unable to load the feature dataset."
            ) from exc


    def _find_candidate(
        self,
        ring_id: str,
        relationships,
    ):
        candidates = detect_connected_candidates(
            relationships=relationships,
        )

        for candidate in candidates:
            if candidate.candidate_id == ring_id:
                return candidate

        raise KeyError(ring_id)

    def investigate(
        self,
        ring_id: str,
    ) -> FraudRingInvestigationResponse:
        ring_id = str(ring_id).strip()

        if not ring_id:
            raise ValueError("ring_id must be non-empty.")

        dataframe = self._load_dataset()

        relationships = self._build_relationships(dataframe)

        candidate = self._find_candidate(
            ring_id,
            relationships,
        )

        network_score = score_fraud_ring_candidate(
            candidate,
        )

        transaction_ids = sorted(
            entity.entity_id
            for entity in candidate.entities
            if entity.entity_type == "transaction"
        )

        # Evidence requires a target transaction belonging to
        # the candidate. Use the deterministic first transaction
        # only as the investigation context; this does not create
        # a prediction or alter the candidate.
        target_transaction_id = transaction_ids[0]

        evidence = assemble_fraud_ring_evidence(
            candidate=candidate,
            network_score=network_score,
            target_transaction_id=target_transaction_id,
        )

        view = assemble_fraud_ring_investigation_view(
            candidate=candidate,
            network_score=network_score,
            evidence=evidence,
        )

        return self._to_response(view)

    @staticmethod
    def _build_relationships(
        dataframe: pd.DataFrame,
    ):
        relationships = []

        for row in dataframe.to_dict(orient="records"):
            relationships.extend(
                build_direct_transaction_relationships(
                    transaction_id=str(row["transaction_id"]),
                    transaction_timestamp=row["timestamp"],
                    transaction=row,
                )
            )

        return tuple(relationships)