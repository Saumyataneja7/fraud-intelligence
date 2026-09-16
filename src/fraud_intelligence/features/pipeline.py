from __future__ import annotations

import pandas as pd

from fraud_intelligence.features.behavioral_features import (
    add_behavioral_deviation_features,
)
from fraud_intelligence.features.entity_features import (
    add_entity_historical_features,
)
from fraud_intelligence.features.historical_features import (
    add_customer_historical_features,
)
from fraud_intelligence.features.novelty_features import (
    add_novelty_features,
)
from fraud_intelligence.features.temporal_features import (
    add_temporal_features,
)
from fraud_intelligence.features.transaction_features import (
    add_transaction_features,
)
from fraud_intelligence.features.velocity_features import (
    add_velocity_features,
)


def build_phase_41_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build Phase 4.1 transaction and temporal features.

    This function is retained for backward compatibility with the
    Phase 4.1 feature pipeline.
    """

    result = add_transaction_features(
        transactions
    )

    result = add_temporal_features(
        result
    )

    return result


def build_feature_dataset(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete Phase 4 feature dataset.

    Feature stages:

    4.1 Transaction + Temporal
    4.2 Customer Historical
    4.3 Velocity
    4.4 Entity Historical
    4.5 Novelty
    4.6 Behavioral Deviation

    The input dataframe is not modified.

    Historical features are calculated using only information
    available strictly before each transaction timestamp.
    """

    result = transactions.copy()

    # ---------------------------------------------------------
    # Phase 4.1
    # Transaction + temporal features
    # ---------------------------------------------------------

    result = add_transaction_features(
        result
    )

    result = add_temporal_features(
        result
    )

    # ---------------------------------------------------------
    # Phase 4.2
    # Customer historical features
    # ---------------------------------------------------------

    result = add_customer_historical_features(
        result
    )

    # ---------------------------------------------------------
    # Phase 4.3
    # Customer velocity features
    # ---------------------------------------------------------

    result = add_velocity_features(
        result
    )

    # ---------------------------------------------------------
    # Phase 4.4
    # Entity historical features
    # ---------------------------------------------------------

    result = add_entity_historical_features(
        result
    )

    # ---------------------------------------------------------
    # Phase 4.5
    # Novelty features
    # ---------------------------------------------------------

    result = add_novelty_features(
        result
    )

    # ---------------------------------------------------------
    # Phase 4.6
    # Behavioral deviation features
    # ---------------------------------------------------------

    result = add_behavioral_deviation_features(
        result
    )

    return result


__all__ = [
    "build_phase_41_features",
    "build_feature_dataset",
]