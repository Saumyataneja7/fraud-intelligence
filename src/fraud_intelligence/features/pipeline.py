from __future__ import annotations

import pandas as pd

from fraud_intelligence.features.temporal_features import (
    add_temporal_features,
)
from fraud_intelligence.features.transaction_features import (
    add_transaction_features,
)


def build_phase_41_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the Phase 4.1 feature set.

    Current feature groups:
    - transaction features
    - temporal features

    Historical and relational features are intentionally excluded until
    later Phase 4 stages.
    """
    result = add_transaction_features(
        transactions
    )

    result = add_temporal_features(
        result
    )

    return result