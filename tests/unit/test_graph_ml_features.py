import numpy as np
import pandas as pd
import pytest
import torch

from fraud_intelligence.graph_ml.features import (
    GraphFeaturePreparationError,
    GraphFeatureSet,
    align_features_to_split,
    get_graph_feature_columns,
    prepare_transaction_node_features,
    validate_transaction_node_features,
)


def make_feature_dataset(n: int = 10) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2025-01-01",
        periods=n,
        freq="min",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "transaction_id": [
                f"TXN_{i:03d}"
                for i in range(n)
            ],
            "timestamp": timestamps,
            "is_fraud": [0] * n,
            "fraud_scenario": [""] * n,
            "feature_a": np.arange(n, dtype=float),
            "feature_b": np.arange(n, dtype=float) + 10,
        }
    )


def make_transaction_nodes(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_id": np.arange(n, dtype=np.int64),
            "transaction_id": [
                f"TXN_{i:03d}"
                for i in range(n)
            ],
        }
    )


def test_get_graph_feature_columns_excludes_targets_and_identifiers() -> None:
    dataset = make_feature_dataset()

    columns = get_graph_feature_columns(dataset)

    assert columns == (
        "feature_a",
        "feature_b",
    )


def test_prepare_returns_graph_feature_set() -> None:
    dataset = make_feature_dataset()
    nodes = make_transaction_nodes()

    result = prepare_transaction_node_features(
        dataset,
        nodes,
    )

    assert isinstance(result, GraphFeatureSet)
    assert result.node_type == "transaction"
    assert result.node_count == 10
    assert result.feature_count == 2


def test_features_are_float32() -> None:
    result = prepare_transaction_node_features(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    assert result.features.dtype == torch.float32


def test_feature_matrix_has_expected_shape() -> None:
    result = prepare_transaction_node_features(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    assert result.features.shape == (10, 2)


def test_node_ids_are_preserved() -> None:
    result = prepare_transaction_node_features(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    assert result.node_ids.tolist() == list(range(10))


def test_transaction_ids_are_preserved() -> None:
    result = prepare_transaction_node_features(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    assert result.transaction_ids.tolist() == [
        f"TXN_{i:03d}"
        for i in range(10)
    ]


def test_feature_values_are_aligned_to_transaction_nodes() -> None:
    dataset = make_feature_dataset()
    nodes = make_transaction_nodes()

    result = prepare_transaction_node_features(
        dataset,
        nodes,
    )

    assert result.features[:, 0].tolist() == list(
        map(float, range(10))
    )

    assert result.features[:, 1].tolist() == [
        float(i + 10)
        for i in range(10)
    ]


def test_unsorted_nodes_are_sorted_by_graph_node_id() -> None:
    dataset = make_feature_dataset()

    nodes = make_transaction_nodes().iloc[
        [5, 2, 9, 0, 1, 7, 3, 8, 4, 6]
    ].reset_index(drop=True)

    result = prepare_transaction_node_features(
        dataset,
        nodes,
    )

    assert result.node_ids.tolist() == list(range(10))


def test_missing_transaction_node_is_rejected() -> None:
    dataset = make_feature_dataset()

    nodes = make_transaction_nodes().iloc[:-1].copy()

    with pytest.raises(GraphFeaturePreparationError):
        prepare_transaction_node_features(
            dataset,
            nodes,
        )


def test_unknown_transaction_node_is_rejected() -> None:
    dataset = make_feature_dataset()
    nodes = make_transaction_nodes()

    nodes.loc[0, "transaction_id"] = "TXN_UNKNOWN"

    with pytest.raises(GraphFeaturePreparationError):
        prepare_transaction_node_features(
            dataset,
            nodes,
        )


def test_duplicate_transaction_ids_are_rejected() -> None:
    dataset = make_feature_dataset()
    dataset.loc[1, "transaction_id"] = dataset.loc[0, "transaction_id"]

    with pytest.raises(GraphFeaturePreparationError):
        prepare_transaction_node_features(
            dataset,
            make_transaction_nodes(),
        )


def test_target_columns_are_not_features() -> None:
    dataset = make_feature_dataset()

    columns = get_graph_feature_columns(dataset)

    assert "is_fraud" not in columns
    assert "fraud_scenario" not in columns


def test_identifiers_are_not_features() -> None:
    dataset = make_feature_dataset()

    columns = get_graph_feature_columns(dataset)

    forbidden = {
        "transaction_id",
        "timestamp",
        "node_id",
        "is_fraud",
        "fraud_scenario",
    }

    assert not forbidden.intersection(columns)


def test_nan_features_are_rejected() -> None:
    dataset = make_feature_dataset()
    dataset.loc[0, "feature_a"] = np.nan

    with pytest.raises(GraphFeaturePreparationError):
        prepare_transaction_node_features(
            dataset,
            make_transaction_nodes(),
        )


def test_infinite_features_are_rejected() -> None:
    dataset = make_feature_dataset()
    dataset.loc[0, "feature_a"] = np.inf

    with pytest.raises(GraphFeaturePreparationError):
        prepare_transaction_node_features(
            dataset,
            make_transaction_nodes(),
        )


def test_feature_validation_passes() -> None:
    result = prepare_transaction_node_features(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    validate_transaction_node_features(result)


def test_feature_validation_rejects_wrong_node_type() -> None:
    result = prepare_transaction_node_features(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    invalid = GraphFeatureSet(
        node_type="customer",
        node_ids=result.node_ids,
        transaction_ids=result.transaction_ids,
        timestamps=result.timestamps,
        features=result.features,
        feature_columns=result.feature_columns,
    )

    with pytest.raises(GraphFeaturePreparationError):
        validate_transaction_node_features(invalid)


def test_split_alignment_preserves_original_node_ids() -> None:
    feature_set = prepare_transaction_node_features(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    split = make_feature_dataset().iloc[:4].copy()

    result = align_features_to_split(
        feature_set,
        split,
    )

    assert result.transaction_ids.tolist() == [
        "TXN_000",
        "TXN_001",
        "TXN_002",
        "TXN_003",
    ]

    assert result.node_ids.tolist() == [
        0,
        1,
        2,
        3,
    ]


def test_split_alignment_does_not_renumber_nodes() -> None:
    feature_set = prepare_transaction_node_features(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    split = make_feature_dataset().iloc[[2, 5, 8]].copy()

    result = align_features_to_split(
        feature_set,
        split,
    )

    assert result.node_ids.tolist() == [
        2,
        5,
        8,
    ]


def test_split_alignment_preserves_feature_columns() -> None:
    feature_set = prepare_transaction_node_features(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    split = make_feature_dataset().iloc[:5].copy()

    result = align_features_to_split(
        feature_set,
        split,
    )

    assert result.feature_columns == (
        "feature_a",
        "feature_b",
    )


def test_split_alignment_rejects_missing_transaction() -> None:
    feature_set = prepare_transaction_node_features(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    split = make_feature_dataset().iloc[:3].copy()
    split.loc[0, "transaction_id"] = "TXN_UNKNOWN"

    with pytest.raises(GraphFeaturePreparationError):
        align_features_to_split(
            feature_set,
            split,
        )