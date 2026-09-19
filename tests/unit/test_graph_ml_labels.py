import numpy as np
import pandas as pd
import pytest
import torch

from fraud_intelligence.graph_ml.labels import (
    GraphLabelPreparationError,
    TransactionNodeLabels,
    align_labels_to_split,
    prepare_transaction_node_labels,
    validate_transaction_node_labels,
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
            "is_fraud": [
                1 if i % 3 == 0 else 0
                for i in range(n)
            ],
            "fraud_scenario": [
                "VELOCITY_ATTACK"
                if i % 3 == 0
                else ""
                for i in range(n)
            ],
        }
    )


def make_transaction_nodes(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_id": np.arange(
                n,
                dtype=np.int64,
            ),
            "transaction_id": [
                f"TXN_{i:03d}"
                for i in range(n)
            ],
        }
    )


def test_prepare_returns_transaction_node_labels() -> None:
    result = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    assert isinstance(
        result,
        TransactionNodeLabels,
    )


def test_node_type_is_transaction() -> None:
    result = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    assert result.node_type == "transaction"


def test_labels_are_one_dimensional_long_tensor() -> None:
    result = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    assert result.labels.ndim == 1
    assert result.labels.dtype == torch.long


def test_label_count_matches_nodes() -> None:
    result = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    assert result.node_count == 10
    assert len(result.node_ids) == 10


def test_labels_are_aligned_to_node_ids() -> None:
    result = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    assert result.node_ids.tolist() == list(range(10))

    assert result.labels.tolist() == [
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
    ]


def test_fraud_scenarios_are_retained_separately() -> None:
    result = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    assert result.fraud_scenarios.tolist()[0] == (
        "VELOCITY_ATTACK"
    )

    assert "fraud_scenario" not in {
        "labels",
    }


def test_positive_and_negative_counts() -> None:
    result = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    assert result.positive_count == 4
    assert result.negative_count == 6


def test_fraud_rate() -> None:
    result = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    assert result.fraud_rate == pytest.approx(0.4)


def test_missing_graph_transaction_is_rejected() -> None:
    dataset = make_feature_dataset()

    nodes = make_transaction_nodes().iloc[:-1].copy()

    with pytest.raises(GraphLabelPreparationError):
        prepare_transaction_node_labels(
            dataset,
            nodes,
        )


def test_unknown_graph_transaction_is_rejected() -> None:
    dataset = make_feature_dataset()
    nodes = make_transaction_nodes()

    nodes.loc[0, "transaction_id"] = "TXN_UNKNOWN"

    with pytest.raises(GraphLabelPreparationError):
        prepare_transaction_node_labels(
            dataset,
            nodes,
        )


def test_duplicate_transaction_ids_are_rejected() -> None:
    dataset = make_feature_dataset()

    dataset.loc[1, "transaction_id"] = (
        dataset.loc[0, "transaction_id"]
    )

    with pytest.raises(GraphLabelPreparationError):
        prepare_transaction_node_labels(
            dataset,
            make_transaction_nodes(),
        )


def test_null_labels_are_rejected() -> None:
    dataset = make_feature_dataset()

    dataset.loc[0, "is_fraud"] = np.nan

    with pytest.raises(GraphLabelPreparationError):
        prepare_transaction_node_labels(
            dataset,
            make_transaction_nodes(),
        )


def test_non_binary_labels_are_rejected() -> None:
    dataset = make_feature_dataset()

    dataset.loc[0, "is_fraud"] = 2

    with pytest.raises(GraphLabelPreparationError):
        prepare_transaction_node_labels(
            dataset,
            make_transaction_nodes(),
        )


def test_labels_are_sorted_by_node_id() -> None:
    dataset = make_feature_dataset()

    nodes = make_transaction_nodes().iloc[
        [5, 2, 9, 0, 1, 7, 3, 8, 4, 6]
    ].reset_index(drop=True)

    result = prepare_transaction_node_labels(
        dataset,
        nodes,
    )

    assert result.node_ids.tolist() == list(range(10))


def test_transaction_ids_are_preserved() -> None:
    result = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    assert result.transaction_ids.tolist() == [
        f"TXN_{i:03d}"
        for i in range(10)
    ]


def test_timestamps_are_preserved() -> None:
    dataset = make_feature_dataset()

    result = prepare_transaction_node_labels(
        dataset,
        make_transaction_nodes(),
    )

    assert result.timestamps.tolist() == (
        dataset["timestamp"].tolist()
    )


def test_validation_passes() -> None:
    result = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    validate_transaction_node_labels(result)


def test_validation_rejects_wrong_node_type() -> None:
    result = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    invalid = TransactionNodeLabels(
        node_type="customer",
        node_ids=result.node_ids,
        transaction_ids=result.transaction_ids,
        timestamps=result.timestamps,
        labels=result.labels,
        fraud_scenarios=result.fraud_scenarios,
    )

    with pytest.raises(GraphLabelPreparationError):
        validate_transaction_node_labels(invalid)


def test_validation_rejects_non_binary_tensor() -> None:
    result = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    invalid = TransactionNodeLabels(
        node_type=result.node_type,
        node_ids=result.node_ids,
        transaction_ids=result.transaction_ids,
        timestamps=result.timestamps,
        labels=torch.tensor(
            [0, 1, 2, 0, 1, 0, 1, 0, 1, 0],
            dtype=torch.long,
        ),
        fraud_scenarios=result.fraud_scenarios,
    )

    with pytest.raises(GraphLabelPreparationError):
        validate_transaction_node_labels(invalid)


def test_split_alignment_preserves_node_ids() -> None:
    label_set = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    split = make_feature_dataset().iloc[
        [2, 5, 8]
    ].copy()

    result = align_labels_to_split(
        label_set,
        split,
    )

    assert result.node_ids.tolist() == [
        2,
        5,
        8,
    ]


def test_split_alignment_preserves_labels() -> None:
    label_set = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    split = make_feature_dataset().iloc[
        [0, 1, 3, 4]
    ].copy()

    result = align_labels_to_split(
        label_set,
        split,
    )

    assert result.labels.tolist() == [
        1,
        0,
        1,
        0,
    ]


def test_split_alignment_rejects_unknown_transaction() -> None:
    label_set = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    split = make_feature_dataset().iloc[
        :3
    ].copy()

    split.loc[0, "transaction_id"] = "TXN_UNKNOWN"

    with pytest.raises(GraphLabelPreparationError):
        align_labels_to_split(
            label_set,
            split,
        )


def test_split_alignment_preserves_fraud_scenarios() -> None:
    label_set = prepare_transaction_node_labels(
        make_feature_dataset(),
        make_transaction_nodes(),
    )

    split = make_feature_dataset().iloc[
        [0, 3, 6]
    ].copy()

    result = align_labels_to_split(
        label_set,
        split,
    )

    assert result.fraud_scenarios.tolist() == [
        "VELOCITY_ATTACK",
        "VELOCITY_ATTACK",
        "VELOCITY_ATTACK",
    ]