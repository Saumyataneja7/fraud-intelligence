import pandas as pd
import pytest

from fraud_intelligence.graph_ml.comparison import (
    GNNComparisonError,
    ModelComparisonResult,
    compare_classical_and_gnn,
    compare_ranking_results,
    validate_model_comparison,
    validate_ranking_comparison,
)
from fraud_intelligence.graph_ml.evaluation import (
    GNNEvaluationResult,
)
from fraud_intelligence.graph_ml.ranking import (
    GNNRankingMetrics,
    GNNRankingResult,
)
from fraud_intelligence.ml.evaluation import (
    ClassificationMetrics,
    EvaluationResult,
)


# ===========================================================================
# Frozen Phase 5 classical ML fixtures
# ===========================================================================


def make_classification_metrics(
    precision: float = 0.50,
    recall: float = 0.60,
    f1: float = 0.55,
    pr_auc: float = 0.65,
    roc_auc: float = 0.80,
    true_negatives: int = 850,
    false_positives: int = 100,
    false_negatives: int = 50,
    true_positives: int = 100,
) -> ClassificationMetrics:
    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_positives=true_positives,
    )


def make_evaluation_result(
    model_name: str = "XGBoost",
    split_name: str = "test",
    metrics: ClassificationMetrics | None = None,
) -> EvaluationResult:
    if metrics is None:
        metrics = make_classification_metrics()

    return EvaluationResult(
        model_name=model_name,
        split_name=split_name,
        metrics=metrics,
    )


# ===========================================================================
# Frozen Phase 7.7 GNN fixture
# ===========================================================================


def make_gnn_result(
    precision: float = 0.45,
    recall: float = 0.70,
    f1: float = 0.55,
    pr_auc: float = 0.68,
    roc_auc: float = 0.82,
    predicted_positive: int = 120,
    actual_positive: int = 150,
) -> GNNEvaluationResult:
    return GNNEvaluationResult(
        split_name="test",
        threshold=0.50,
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        true_negative=1000,
        false_positive=100,
        false_negative=30,
        true_positive=120,
        support=1150,
        predicted_positive=predicted_positive,
        actual_positive=actual_positive,
    )


# ===========================================================================
# Frozen Phase 7.9 GNN ranking fixtures
# ===========================================================================


def make_gnn_ranking_result() -> GNNRankingResult:
    return GNNRankingResult(
        split_name="test",
        metrics=(
            GNNRankingMetrics(
                split_name="test",
                k=10,
                precision_at_k=0.50,
                recall_at_k=0.10,
                fraud_count_at_k=5,
                total_fraud=50,
                support=100,
            ),
            GNNRankingMetrics(
                split_name="test",
                k=50,
                precision_at_k=0.40,
                recall_at_k=0.30,
                fraud_count_at_k=20,
                total_fraud=50,
                support=100,
            ),
        ),
    )


def make_single_k_gnn_ranking_result() -> GNNRankingResult:
    return GNNRankingResult(
        split_name="test",
        metrics=(
            GNNRankingMetrics(
                split_name="test",
                k=10,
                precision_at_k=0.50,
                recall_at_k=0.10,
                fraud_count_at_k=5,
                total_fraud=50,
                support=100,
            ),
        ),
    )


# ===========================================================================
# Model comparison tests
# ===========================================================================


def test_model_comparison_type():
    result = compare_classical_and_gnn(
        classical_results=[
            make_evaluation_result(
                "LogisticRegression",
            ),
            make_evaluation_result(
                "RandomForest",
            ),
            make_evaluation_result(
                "XGBoost",
            ),
        ],
        gnn_result=make_gnn_result(),
        split_name="test",
    )

    assert isinstance(
        result,
        ModelComparisonResult,
    )


def test_model_comparison_contains_all_models():
    result = compare_classical_and_gnn(
        classical_results=[
            make_evaluation_result(
                "LogisticRegression",
            ),
            make_evaluation_result(
                "RandomForest",
            ),
            make_evaluation_result(
                "XGBoost",
            ),
        ],
        gnn_result=make_gnn_result(),
        split_name="test",
    )

    assert result.model_names == (
        "LogisticRegression",
        "RandomForest",
        "XGBoost",
        "GraphSAGE",
    )


def test_model_comparison_row_count():
    result = compare_classical_and_gnn(
        classical_results=[
            make_evaluation_result(
                "LogisticRegression",
            ),
            make_evaluation_result(
                "RandomForest",
            ),
        ],
        gnn_result=make_gnn_result(),
        split_name="test",
    )

    assert result.row_count == 3


def test_model_comparison_schema():
    result = compare_classical_and_gnn(
        classical_results=[
            make_evaluation_result(
                "XGBoost",
            ),
        ],
        gnn_result=make_gnn_result(),
        split_name="test",
    )

    assert list(result.metrics.columns) == [
        "model",
        "model_family",
        "split",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "predicted_positives",
        "actual_positives",
    ]


def test_model_family_is_recorded():
    result = compare_classical_and_gnn(
        classical_results=[
            make_evaluation_result(
                "XGBoost",
            ),
        ],
        gnn_result=make_gnn_result(),
        split_name="test",
    )

    assert result.metrics.iloc[0]["model_family"] == (
        "classical_ml"
    )

    assert result.metrics.iloc[1]["model_family"] == "gnn"


def test_split_is_recorded():
    result = compare_classical_and_gnn(
        classical_results=[
            make_evaluation_result(
                "XGBoost",
            ),
        ],
        gnn_result=make_gnn_result(),
        split_name="validation",
    )

    assert (
        result.metrics["split"].unique().tolist()
        == ["validation"]
    )


def test_metrics_are_copied():
    result = compare_classical_and_gnn(
        classical_results=[
            make_evaluation_result(
                model_name="XGBoost",
                metrics=make_classification_metrics(
                    precision=0.91,
                    recall=0.42,
                    f1=0.58,
                    pr_auc=0.73,
                    roc_auc=0.91,
                    true_negatives=890,
                    false_positives=70,
                    false_negatives=58,
                    true_positives=42,
                ),
            ),
        ],
        gnn_result=make_gnn_result(),
        split_name="test",
    )

    row = result.metrics.iloc[0]

    assert row["precision"] == pytest.approx(0.91)
    assert row["recall"] == pytest.approx(0.42)
    assert row["f1"] == pytest.approx(0.58)
    assert row["pr_auc"] == pytest.approx(0.73)
    assert row["roc_auc"] == pytest.approx(0.91)

    assert row["predicted_positives"] == 112
    assert row["actual_positives"] == 100


def test_gnn_metrics_are_copied():
    result = compare_classical_and_gnn(
        classical_results=[
            make_evaluation_result(
                "XGBoost",
            ),
        ],
        gnn_result=make_gnn_result(
            precision=0.31,
            recall=0.77,
            f1=0.44,
            pr_auc=0.59,
            roc_auc=0.86,
            predicted_positive=200,
            actual_positive=150,
        ),
        split_name="test",
    )

    row = result.metrics.iloc[1]

    assert row["model"] == "GraphSAGE"
    assert row["precision"] == pytest.approx(0.31)
    assert row["recall"] == pytest.approx(0.77)
    assert row["f1"] == pytest.approx(0.44)
    assert row["pr_auc"] == pytest.approx(0.59)
    assert row["roc_auc"] == pytest.approx(0.86)
    assert row["predicted_positives"] == 200
    assert row["actual_positives"] == 150


def test_empty_classical_results_rejected():
    with pytest.raises(GNNComparisonError):
        compare_classical_and_gnn(
            classical_results=[],
            gnn_result=make_gnn_result(),
            split_name="test",
        )


def test_empty_split_name_rejected():
    with pytest.raises(GNNComparisonError):
        compare_classical_and_gnn(
            classical_results=[
                make_evaluation_result(
                    "XGBoost",
                ),
            ],
            gnn_result=make_gnn_result(),
            split_name="",
        )


def test_invalid_gnn_result_rejected():
    with pytest.raises(GNNComparisonError):
        compare_classical_and_gnn(
            classical_results=[
                make_evaluation_result(
                    "XGBoost",
                ),
            ],
            gnn_result="invalid",
            split_name="test",
        )


def test_invalid_classical_result_rejected():
    with pytest.raises(GNNComparisonError):
        compare_classical_and_gnn(
            classical_results=["invalid"],
            gnn_result=make_gnn_result(),
            split_name="test",
        )


# ===========================================================================
# Model comparison validation
# ===========================================================================


def test_validation_schema_accepts_valid_frame():
    frame = pd.DataFrame(
        {
            "model": ["XGBoost", "GraphSAGE"],
            "model_family": [
                "classical_ml",
                "gnn",
            ],
            "split": ["test", "test"],
            "precision": [0.2, 0.3],
            "recall": [0.4, 0.5],
            "f1": [0.3, 0.4],
            "pr_auc": [0.5, 0.6],
            "roc_auc": [0.8, 0.9],
            "predicted_positives": [200, 100],
            "actual_positives": [150, 150],
        }
    )

    validate_model_comparison(frame)


def test_invalid_metric_range_rejected():
    frame = pd.DataFrame(
        {
            "model": ["XGBoost"],
            "model_family": ["classical_ml"],
            "split": ["test"],
            "precision": [1.5],
            "recall": [0.5],
            "f1": [0.4],
            "pr_auc": [0.5],
            "roc_auc": [0.8],
            "predicted_positives": [10],
            "actual_positives": [10],
        }
    )

    with pytest.raises(GNNComparisonError):
        validate_model_comparison(frame)


def test_empty_comparison_rejected():
    with pytest.raises(GNNComparisonError):
        validate_model_comparison(
            pd.DataFrame()
        )


def test_negative_prediction_count_rejected():
    frame = pd.DataFrame(
        {
            "model": ["XGBoost"],
            "model_family": ["classical_ml"],
            "split": ["test"],
            "precision": [0.5],
            "recall": [0.5],
            "f1": [0.5],
            "pr_auc": [0.5],
            "roc_auc": [0.5],
            "predicted_positives": [-1],
            "actual_positives": [10],
        }
    )

    with pytest.raises(GNNComparisonError):
        validate_model_comparison(frame)


# ===========================================================================
# Ranking comparison tests
# ===========================================================================


def test_ranking_comparison_type():
    result = compare_ranking_results(
        classical_ranking=None,
        gnn_ranking=make_gnn_ranking_result(),
        split_name="test",
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_ranking_comparison_contains_gnn():
    result = compare_ranking_results(
        classical_ranking=None,
        gnn_ranking=make_gnn_ranking_result(),
        split_name="test",
    )

    assert result["model"].tolist() == [
        "GraphSAGE",
        "GraphSAGE",
    ]


def test_ranking_comparison_contains_k():
    result = compare_ranking_results(
        classical_ranking=None,
        gnn_ranking=make_gnn_ranking_result(),
        split_name="test",
    )

    assert result["k"].tolist() == [
        10,
        50,
    ]


def test_ranking_comparison_schema():
    result = compare_ranking_results(
        classical_ranking=None,
        gnn_ranking=make_single_k_gnn_ranking_result(),
        split_name="test",
    )

    assert list(result.columns) == [
        "model",
        "model_family",
        "split",
        "k",
        "precision_at_k",
        "recall_at_k",
    ]


def test_classical_ranking_can_be_included():
    classical = pd.DataFrame(
        {
            "model": ["XGBoost"],
            "k": [10],
            "precision_at_k": [0.6],
            "recall_at_k": [0.1],
        }
    )

    result = compare_ranking_results(
        classical_ranking=classical,
        gnn_ranking=make_single_k_gnn_ranking_result(),
        split_name="test",
    )

    assert len(result) == 2

    assert result["model"].tolist() == [
        "XGBoost",
        "GraphSAGE",
    ]


def test_invalid_classical_ranking_rejected():
    with pytest.raises(GNNComparisonError):
        compare_ranking_results(
            classical_ranking=pd.DataFrame(
                {"model": ["XGBoost"]}
            ),
            gnn_ranking=make_single_k_gnn_ranking_result(),
            split_name="test",
        )


def test_invalid_ranking_k_rejected():
    frame = pd.DataFrame(
        {
            "model": ["GraphSAGE"],
            "model_family": ["gnn"],
            "split": ["test"],
            "k": [0],
            "precision_at_k": [0.5],
            "recall_at_k": [0.2],
        }
    )

    with pytest.raises(GNNComparisonError):
        validate_ranking_comparison(frame)


def test_invalid_ranking_metric_rejected():
    frame = pd.DataFrame(
        {
            "model": ["GraphSAGE"],
            "model_family": ["gnn"],
            "split": ["test"],
            "k": [10],
            "precision_at_k": [1.2],
            "recall_at_k": [0.2],
        }
    )

    with pytest.raises(GNNComparisonError):
        validate_ranking_comparison(frame)