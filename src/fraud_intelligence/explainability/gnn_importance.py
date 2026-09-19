from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch

from fraud_intelligence.explainability.contracts import (
    GNNExplanationContract,
    GNNExplanationContractError,
)
from fraud_intelligence.explainability.graph_neighborhood import (
    GNNNeighborhoodExplanation,
)

FORBIDDEN_IMPORTANCE_FEATURES = frozenset(
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


class GNNImportanceError(ValueError):
    """Raised when GNN importance calculation fails."""


@dataclass(frozen=True)
class GNNFeatureImportance:
    """Importance of one transaction feature."""

    feature: str
    importance: float
    rank: int


@dataclass(frozen=True)
class GNNNodeImportance:
    """Importance of one graph node."""

    node_type: str
    node_index: int
    importance: float
    rank: int


@dataclass(frozen=True)
class GNNImportanceResult:
    """Feature and graph-node importance for one transaction."""

    transaction_node_index: int
    baseline_score: float
    feature_importances: tuple[GNNFeatureImportance, ...]
    node_importances: tuple[GNNNodeImportance, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_importances)

    @property
    def node_count(self) -> int:
        return len(self.node_importances)

    @property
    def top_feature(
        self,
    ) -> GNNFeatureImportance | None:
        if not self.feature_importances:
            return None

        return self.feature_importances[0]

    @property
    def top_node(
        self,
    ) -> GNNNodeImportance | None:
        if not self.node_importances:
            return None

        return self.node_importances[0]


def _validate_score(
    score: float,
    name: str,
) -> float:
    value = float(score)

    if not np.isfinite(value):
        raise GNNImportanceError(
            f"{name} must be finite."
        )

    if not 0.0 <= value <= 1.0:
        raise GNNImportanceError(
            f"{name} must be between 0 and 1."
        )

    return value


def _validate_feature_frame(
    X: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> None:
    if not isinstance(X, pd.DataFrame):
        raise GNNImportanceError(
            "X must be a pandas DataFrame."
        )

    if len(X) != 1:
        raise GNNImportanceError(
            "GNN feature importance expects exactly "
            "one transaction row."
        )

    if tuple(X.columns) != feature_columns:
        raise GNNImportanceError(
            "X columns do not match the approved "
            "GNN feature columns."
        )

    if X.isna().any().any():
        raise GNNImportanceError(
            "GNN features cannot contain missing values."
        )

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise GNNImportanceError(
            "GNN features must contain finite values."
        )


def _validate_feature_names(
    feature_columns: tuple[str, ...],
    contract: GNNExplanationContract,
) -> None:
    forbidden = sorted(
        set(feature_columns)
        & FORBIDDEN_IMPORTANCE_FEATURES
    )

    if forbidden:
        raise GNNExplanationContractError(
            "GNN importance cannot use forbidden "
            f"columns: {forbidden}."
        )

    if len(set(feature_columns)) != len(feature_columns):
        raise GNNImportanceError(
            "Feature columns must be unique."
        )


def _rank_feature_importances(
    feature_columns: tuple[str, ...],
    importances: np.ndarray,
) -> tuple[GNNFeatureImportance, ...]:
    if len(importances) != len(feature_columns):
        raise GNNImportanceError(
            "Feature importance count does not match "
            "feature count."
        )

    values = np.asarray(
        importances,
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise GNNImportanceError(
            "Feature importance values must be finite."
        )

    if (values < 0).any():
        raise GNNImportanceError(
            "Feature importance values cannot be negative."
        )

    order = np.argsort(
        -values,
        kind="mergesort",
    )

    return tuple(
        GNNFeatureImportance(
            feature=feature_columns[index],
            importance=float(values[index]),
            rank=rank,
        )
        for rank, index in enumerate(
            order,
            start=1,
        )
    )


def _rank_node_importances(
    neighborhood: GNNNeighborhoodExplanation,
    importances: dict[
        tuple[str, int],
        float,
    ],
) -> tuple[GNNNodeImportance, ...]:
    values = []

    for node in neighborhood.neighbor_nodes:
        key = (
            node.node_type,
            node.node_index,
        )

        importance = float(
            importances.get(key, 0.0)
        )

        if not np.isfinite(importance):
            raise GNNImportanceError(
                "Node importance values must be finite."
            )

        if importance < 0:
            raise GNNImportanceError(
                "Node importance values cannot be negative."
            )

        values.append(
            (
                node.node_type,
                node.node_index,
                importance,
            )
        )

    values.sort(
        key=lambda item: (
            -item[2],
            item[0],
            item[1],
        )
    )

    return tuple(
        GNNNodeImportance(
            node_type=node_type,
            node_index=node_index,
            importance=importance,
            rank=rank,
        )
        for rank, (
            node_type,
            node_index,
            importance,
        ) in enumerate(
            values,
            start=1,
        )
    )


def calculate_feature_perturbation_importance(
    *,
    X: pd.DataFrame,
    feature_columns: tuple[str, ...],
    predict_score: Callable[
        [pd.DataFrame],
        float,
    ],
    contract: GNNExplanationContract | None = None,
) -> tuple[GNNFeatureImportance, ...]:
    """
    Calculate local feature importance through one-feature
    perturbation.

    Each feature is replaced with zero and the change from
    the baseline prediction score is measured.

    This is an attribution heuristic, not a causal estimate.
    """
    if contract is None:
        contract = GNNExplanationContract()

    _validate_feature_frame(
        X,
        feature_columns,
    )

    _validate_feature_names(
        feature_columns,
        contract,
    )

    baseline = _validate_score(
        predict_score(X.copy()),
        "baseline_score",
    )

    importances = np.zeros(
        len(feature_columns),
        dtype=float,
    )

    for index, feature in enumerate(
        feature_columns
    ):
        perturbed = X.copy()

        perturbed.loc[
            perturbed.index[0],
            feature,
        ] = 0.0

        perturbed_score = _validate_score(
            predict_score(perturbed),
            "perturbed_score",
        )

        importances[index] = abs(
            baseline - perturbed_score
        )

    return _rank_feature_importances(
        feature_columns,
        importances,
    )


def calculate_node_perturbation_importance(
    *,
    neighborhood: GNNNeighborhoodExplanation,
    baseline_score: float,
    perturb_node: Callable[
        [str, int],
        float,
    ],
    contract: GNNExplanationContract | None = None,
) -> tuple[GNNNodeImportance, ...]:
    """
    Calculate local node importance through node ablation.

    `perturb_node(node_type, node_index)` must return the
    prediction score after that node's contribution has been
    removed/ablated by the caller.

    Importance is the absolute change from the baseline score.
    """
    if contract is None:
        contract = GNNExplanationContract()

    validate_gnn_importance_neighborhood(
        neighborhood,
        contract,
    )

    baseline = _validate_score(
        baseline_score,
        "baseline_score",
    )

    importances: dict[
        tuple[str, int],
        float,
    ] = {}

    for node in neighborhood.neighbor_nodes:
        perturbed_score = _validate_score(
            perturb_node(
                node.node_type,
                node.node_index,
            ),
            "perturbed_score",
        )

        importances[
            (
                node.node_type,
                node.node_index,
            )
        ] = abs(
            baseline - perturbed_score
        )

    return _rank_node_importances(
        neighborhood,
        importances,
    )


def build_gnn_importance_result(
    *,
    neighborhood: GNNNeighborhoodExplanation,
    baseline_score: float,
    feature_importances: tuple[
        GNNFeatureImportance,
        ...,
    ],
    node_importances: tuple[
        GNNNodeImportance,
        ...,
    ],
    contract: GNNExplanationContract | None = None,
) -> GNNImportanceResult:
    if contract is None:
        contract = GNNExplanationContract()

    validate_gnn_importance_neighborhood(
        neighborhood,
        contract,
    )

    baseline = _validate_score(
        baseline_score,
        "baseline_score",
    )

    result = GNNImportanceResult(
        transaction_node_index=(
            neighborhood.transaction_node_index
        ),
        baseline_score=baseline,
        feature_importances=feature_importances,
        node_importances=node_importances,
    )

    validate_gnn_importance_result(
        result,
        contract,
    )

    return result


def validate_gnn_importance_neighborhood(
    neighborhood: GNNNeighborhoodExplanation,
    contract: GNNExplanationContract | None = None,
) -> None:
    if contract is None:
        contract = GNNExplanationContract()

    if not isinstance(
        neighborhood,
        GNNNeighborhoodExplanation,
    ):
        raise GNNImportanceError(
            "neighborhood must be a "
            "GNNNeighborhoodExplanation."
        )

    if neighborhood.transaction_node_index < 0:
        raise GNNImportanceError(
            "Transaction node index cannot be negative."
        )

    if neighborhood.transaction_timestamp.tzinfo is None:
        raise GNNImportanceError(
            "Transaction timestamp must be timezone-aware."
        )

    for node in neighborhood.nodes:
        contract.validate_hops(node.hop)

    for edge in neighborhood.edges:
        contract.validate_hops(edge.hop)

        if edge.edge_type not in (
            contract.allowed_edge_types
        ):
            raise GNNExplanationContractError(
                f"Unknown edge type: {edge.edge_type}."
            )


def validate_gnn_importance_result(
    result: GNNImportanceResult,
    contract: GNNExplanationContract | None = None,
) -> None:
    if contract is None:
        contract = GNNExplanationContract()

    if not isinstance(
        result,
        GNNImportanceResult,
    ):
        raise GNNImportanceError(
            "result must be a GNNImportanceResult."
        )

    _validate_score(
        result.baseline_score,
        "baseline_score",
    )

    feature_names = set()
    feature_ranks = set()

    for item in result.feature_importances:
        if item.feature in feature_names:
            raise GNNImportanceError(
                "Duplicate feature importance."
            )

        if item.feature in (
            contract.forbidden_context
        ):
            raise GNNExplanationContractError(
                f"Forbidden feature: {item.feature}."
            )

        if item.feature in (
            contract.forbidden_node_metadata
        ):
            raise GNNExplanationContractError(
                f"Forbidden metadata feature: {item.feature}."
            )

        if item.rank < 1:
            raise GNNImportanceError(
                "Feature rank must be at least 1."
            )

        if item.rank in feature_ranks:
            raise GNNImportanceError(
                "Duplicate feature rank."
            )

        if not np.isfinite(item.importance):
            raise GNNImportanceError(
                "Feature importance must be finite."
            )

        if item.importance < 0:
            raise GNNImportanceError(
                "Feature importance cannot be negative."
            )

        feature_names.add(item.feature)
        feature_ranks.add(item.rank)

    expected_feature_ranks = set(
        range(
            1,
            len(result.feature_importances) + 1,
        )
    )

    if feature_ranks != expected_feature_ranks:
        raise GNNImportanceError(
            "Feature ranks must be contiguous."
        )

    node_keys = set()
    node_ranks = set()

    for item in result.node_importances:
        key = (
            item.node_type,
            item.node_index,
        )

        if key in node_keys:
            raise GNNImportanceError(
                "Duplicate node importance."
            )

        if item.node_type not in (
            contract.allowed_node_types
        ):
            raise GNNExplanationContractError(
                f"Unknown node type: {item.node_type}."
            )

        if item.node_index < 0:
            raise GNNImportanceError(
                "Node index cannot be negative."
            )

        if item.rank < 1:
            raise GNNImportanceError(
                "Node rank must be at least 1."
            )

        if item.rank in node_ranks:
            raise GNNImportanceError(
                "Duplicate node rank."
            )

        if not np.isfinite(item.importance):
            raise GNNImportanceError(
                "Node importance must be finite."
            )

        if item.importance < 0:
            raise GNNImportanceError(
                "Node importance cannot be negative."
            )

        node_keys.add(key)
        node_ranks.add(item.rank)

    expected_node_ranks = set(
        range(
            1,
            len(result.node_importances) + 1,
        )
    )

    if node_ranks != expected_node_ranks:
        raise GNNImportanceError(
            "Node ranks must be contiguous."
        )