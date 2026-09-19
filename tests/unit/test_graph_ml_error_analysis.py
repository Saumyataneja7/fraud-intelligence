import pytest
import torch

from fraud_intelligence.graph_ml.error_analysis import (
    GNNErrorAnalysisError,
    GNNErrorAnalysisResult,
    GNNErrorCounts,
    analyze_gnn_errors,
    build_gnn_error_frame,
    calculate_gnn_error_counts,
    extract_false_negatives,
    extract_false_positives,
    summarize_gnn_errors_by_amount_band,
    summarize_gnn_errors_by_column,
)


def make_ids():
    return [
        "tx_001",
        "tx_002",
        "tx_003",
        "tx_004",
        "tx_005",
    ]


def make_labels():
    return torch.tensor(
        [0, 0, 1, 1, 1],
        dtype=torch.long,
    )


def make_probabilities():
    return torch.tensor(
        [0.10, 0.80, 0.20, 0.90, 0.60],
        dtype=torch.float32,
    )


def make_scenarios():
    return [
        None,
        None,
        "ACCOUNT_TAKEOVER",
        "SHARED_DEVICE",
        "FRAUD_RING",
    ]


def make_amounts():
    return [
        10.0,
        40.0,
        75.0,
        300.0,
        1200.0,
    ]


def test_error_counts_type():
    result = calculate_gnn_error_counts(
        make_labels(),
        make_probabilities(),
    )

    assert isinstance(
        result,
        GNNErrorCounts,
    )


def test_error_counts_are_correct():
    result = calculate_gnn_error_counts(
        make_labels(),
        make_probabilities(),
    )

    assert result.true_negatives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.true_positives == 2


def test_error_counts_total():
    result = calculate_gnn_error_counts(
        make_labels(),
        make_probabilities(),
    )

    assert result.total == 5


def test_error_frame_type():
    result = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
    )

    assert len(result) == 5


def test_error_frame_columns():
    result = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
    )

    assert list(result.columns) == [
        "transaction_id",
        "actual_label",
        "predicted_probability",
        "predicted_label",
        "error_type",
    ]


def test_error_frame_prediction_labels():
    result = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
    )

    assert result["predicted_label"].tolist() == [
        0,
        1,
        0,
        1,
        1,
    ]


def test_error_types():
    result = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
    )

    assert result["error_type"].tolist() == [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "true_positive",
    ]


def test_scenarios_are_optional():
    result = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
        fraud_scenarios=make_scenarios(),
    )

    assert "fraud_scenario" in result.columns


def test_amounts_are_optional():
    result = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
        amounts=make_amounts(),
    )

    assert "amount" in result.columns


def test_analyze_result_type():
    result = analyze_gnn_errors(
        make_ids(),
        make_labels(),
        make_probabilities(),
    )

    assert isinstance(
        result,
        GNNErrorAnalysisResult,
    )


def test_analyze_result_properties():
    result = analyze_gnn_errors(
        make_ids(),
        make_labels(),
        make_probabilities(),
    )

    assert result.false_positive_count == 1
    assert result.false_negative_count == 1
    assert result.error_count == 2


def test_split_name_preserved():
    result = analyze_gnn_errors(
        make_ids(),
        make_labels(),
        make_probabilities(),
        split_name="validation",
    )

    assert result.split_name == "validation"


def test_threshold_is_preserved():
    result = analyze_gnn_errors(
        make_ids(),
        make_labels(),
        make_probabilities(),
        threshold=0.7,
    )

    assert result.threshold == 0.7


def test_extract_false_positives():
    frame = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
    )

    result = extract_false_positives(frame)

    assert len(result) == 1
    assert result.iloc[0]["transaction_id"] == "tx_002"


def test_extract_false_negatives():
    frame = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
    )

    result = extract_false_negatives(frame)

    assert len(result) == 1
    assert result.iloc[0]["transaction_id"] == "tx_003"


def test_summary_by_error_type_column():
    frame = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
        fraud_scenarios=make_scenarios(),
    )

    result = summarize_gnn_errors_by_column(
        frame,
        "fraud_scenario",
    )

    assert "false_positive" in result.columns
    assert "false_negative" in result.columns


def test_summary_by_scenario():
    frame = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
        fraud_scenarios=make_scenarios(),
    )

    result = summarize_gnn_errors_by_column(
        frame,
        "fraud_scenario",
    )

    assert result.loc[
        "ACCOUNT_TAKEOVER",
        "false_negative",
    ] == 1


def test_summary_handles_missing_error_types():
    labels = torch.tensor(
        [1, 1],
        dtype=torch.long,
    )

    probabilities = torch.tensor(
        [0.9, 0.8],
        dtype=torch.float32,
    )

    frame = build_gnn_error_frame(
        ["tx_a", "tx_b"],
        labels,
        probabilities,
        fraud_scenarios=[
            "FRAUD_RING",
            "FRAUD_RING",
        ],
    )

    result = summarize_gnn_errors_by_column(
        frame,
        "fraud_scenario",
    )

    assert result.loc[
        "FRAUD_RING",
        "false_positive",
    ] == 0

    assert result.loc[
        "FRAUD_RING",
        "false_negative",
    ] == 0


def test_amount_band_summary():
    frame = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
        amounts=make_amounts(),
    )

    result = summarize_gnn_errors_by_amount_band(
        frame,
    )

    assert "false_positive" in result.columns
    assert "false_negative" in result.columns


def test_custom_amount_bins():
    frame = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
        amounts=make_amounts(),
    )

    result = summarize_gnn_errors_by_amount_band(
        frame,
        bins=[0, 100, 500, float("inf")],
    )

    assert len(result) == 3


def test_duplicate_transaction_ids_rejected():
    with pytest.raises(GNNErrorAnalysisError):
        build_gnn_error_frame(
            ["tx_001", "tx_001"],
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            torch.tensor(
                [0.1, 0.9],
                dtype=torch.float32,
            ),
        )


def test_mismatched_transaction_ids_rejected():
    with pytest.raises(GNNErrorAnalysisError):
        build_gnn_error_frame(
            ["tx_001"],
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            torch.tensor(
                [0.1, 0.9],
                dtype=torch.float32,
            ),
        )


def test_mismatched_scenarios_rejected():
    with pytest.raises(GNNErrorAnalysisError):
        build_gnn_error_frame(
            make_ids(),
            make_labels(),
            make_probabilities(),
            fraud_scenarios=["FRAUD_RING"],
        )


def test_mismatched_amounts_rejected():
    with pytest.raises(GNNErrorAnalysisError):
        build_gnn_error_frame(
            make_ids(),
            make_labels(),
            make_probabilities(),
            amounts=[10.0],
        )


def test_invalid_threshold_rejected():
    with pytest.raises(GNNErrorAnalysisError):
        build_gnn_error_frame(
            make_ids(),
            make_labels(),
            make_probabilities(),
            threshold=1.5,
        )


def test_invalid_label_dtype_rejected():
    with pytest.raises(GNNErrorAnalysisError):
        calculate_gnn_error_counts(
            torch.tensor(
                [0, 1],
                dtype=torch.int64,
            ).float(),
            torch.tensor(
                [0.1, 0.9],
                dtype=torch.float32,
            ),
        )


def test_nonbinary_labels_rejected():
    with pytest.raises(GNNErrorAnalysisError):
        calculate_gnn_error_counts(
            torch.tensor(
                [0, 2],
                dtype=torch.long,
            ),
            torch.tensor(
                [0.1, 0.9],
                dtype=torch.float32,
            ),
        )


def test_invalid_probabilities_rejected():
    with pytest.raises(GNNErrorAnalysisError):
        calculate_gnn_error_counts(
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            torch.tensor(
                [-0.1, 1.1],
                dtype=torch.float32,
            ),
        )


def test_nan_probabilities_rejected():
    with pytest.raises(GNNErrorAnalysisError):
        calculate_gnn_error_counts(
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            torch.tensor(
                [float("nan"), 0.9],
                dtype=torch.float32,
            ),
        )


def test_extract_missing_columns_rejected():
    with pytest.raises(GNNErrorAnalysisError):
        extract_false_positives(
            make_ids_dataframe()
        )


def test_missing_scenario_column_rejected():
    frame = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
    )

    with pytest.raises(GNNErrorAnalysisError):
        summarize_gnn_errors_by_column(
            frame,
            "fraud_scenario",
        )


def test_missing_amount_column_rejected():
    frame = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
    )

    with pytest.raises(GNNErrorAnalysisError):
        summarize_gnn_errors_by_amount_band(
            frame,
        )


def test_invalid_amount_bins_rejected():
    frame = build_gnn_error_frame(
        make_ids(),
        make_labels(),
        make_probabilities(),
        amounts=make_amounts(),
    )

    with pytest.raises(GNNErrorAnalysisError):
        summarize_gnn_errors_by_amount_band(
            frame,
            bins=[0],
        )


def make_ids_dataframe():
    import pandas as pd

    return pd.DataFrame(
        {
            "transaction_id": ["tx_001"],
        }
    )