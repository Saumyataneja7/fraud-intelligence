from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fraud_intelligence.ml.dataset import prepare_model_dataset
from fraud_intelligence.ml.evaluation import (
    evaluate_model,
    evaluation_results_to_dataframe,
)
from fraud_intelligence.ml.logistic_regression import (
    predict_logistic_regression,
    predict_logistic_regression_proba,
    train_logistic_regression,
)
from fraud_intelligence.ml.random_forest import (
    predict_random_forest,
    predict_random_forest_proba,
    train_random_forest,
)
from fraud_intelligence.ml.splits import (
    load_and_split_feature_dataset,
)
from fraud_intelligence.ml.xgboost import (
    predict_xgboost,
    predict_xgboost_proba,
    train_xgboost,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "feature_dataset.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "classical_model_metrics.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "classical_model_metrics.json"
)


def _ensure_output_directory() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def _evaluate_logistic_regression(
    model_dataset,
):
    model = train_logistic_regression(
        model_dataset.X_train,
        model_dataset.y_train,
    )

    validation_pred = predict_logistic_regression(
        model,
        model_dataset.X_validation,
    )

    validation_proba = (
        predict_logistic_regression_proba(
            model,
            model_dataset.X_validation,
        )
    )

    test_pred = predict_logistic_regression(
        model,
        model_dataset.X_test,
    )

    test_proba = (
        predict_logistic_regression_proba(
            model,
            model_dataset.X_test,
        )
    )

    validation_result = evaluate_model(
        model_name="logistic_regression",
        split_name="validation",
        y_true=model_dataset.y_validation,
        y_pred=validation_pred,
        y_proba=validation_proba,
    )

    test_result = evaluate_model(
        model_name="logistic_regression",
        split_name="test",
        y_true=model_dataset.y_test,
        y_pred=test_pred,
        y_proba=test_proba,
    )

    return validation_result, test_result


def _evaluate_random_forest(
    model_dataset,
):
    model = train_random_forest(
        model_dataset.X_train,
        model_dataset.y_train,
    )

    validation_pred = predict_random_forest(
        model,
        model_dataset.X_validation,
    )

    validation_proba = (
        predict_random_forest_proba(
            model,
            model_dataset.X_validation,
        )
    )

    test_pred = predict_random_forest(
        model,
        model_dataset.X_test,
    )

    test_proba = (
        predict_random_forest_proba(
            model,
            model_dataset.X_test,
        )
    )

    validation_result = evaluate_model(
        model_name="random_forest",
        split_name="validation",
        y_true=model_dataset.y_validation,
        y_pred=validation_pred,
        y_proba=validation_proba,
    )

    test_result = evaluate_model(
        model_name="random_forest",
        split_name="test",
        y_true=model_dataset.y_test,
        y_pred=test_pred,
        y_proba=test_proba,
    )

    return validation_result, test_result


def _evaluate_xgboost(
    model_dataset,
):
    model = train_xgboost(
        model_dataset.X_train,
        model_dataset.y_train,
    )

    validation_pred = predict_xgboost(
        model,
        model_dataset.X_validation,
    )

    validation_proba = (
        predict_xgboost_proba(
            model,
            model_dataset.X_validation,
        )
    )

    test_pred = predict_xgboost(
        model,
        model_dataset.X_test,
    )

    test_proba = (
        predict_xgboost_proba(
            model,
            model_dataset.X_test,
        )
    )

    validation_result = evaluate_model(
        model_name="xgboost",
        split_name="validation",
        y_true=model_dataset.y_validation,
        y_pred=validation_pred,
        y_proba=validation_proba,
    )

    test_result = evaluate_model(
        model_name="xgboost",
        split_name="test",
        y_true=model_dataset.y_test,
        y_pred=test_pred,
        y_proba=test_proba,
    )

    return validation_result, test_result


def main() -> None:
    print("=" * 72)
    print("Phase 5.7 — Classical Model Evaluation")
    print("=" * 72)

    _ensure_output_directory()

    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Feature dataset not found: "
            f"{FEATURE_DATASET_PATH}"
        )

    print("\n[1/4] Loading temporal splits...")

    splits = load_and_split_feature_dataset(
        FEATURE_DATASET_PATH
    )

    print(
        f"Train:      {len(splits.train):,} rows"
    )
    print(
        f"Validation: {len(splits.validation):,} rows"
    )
    print(
        f"Test:       {len(splits.test):,} rows"
    )

    print("\n[2/4] Preparing model datasets...")

    model_dataset = prepare_model_dataset(
        splits
    )

    print(
        f"Model features: "
        f"{len(model_dataset.feature_columns)}"
    )

    print("\n[3/4] Training and evaluating models...")

    results = []

    print("\n  Logistic Regression...")
    lr_validation, lr_test = (
        _evaluate_logistic_regression(
            model_dataset
        )
    )
    results.extend(
        [
            lr_validation,
            lr_test,
        ]
    )

    print("  Random Forest...")
    rf_validation, rf_test = (
        _evaluate_random_forest(
            model_dataset
        )
    )
    results.extend(
        [
            rf_validation,
            rf_test,
        ]
    )

    print("  XGBoost...")
    xgb_validation, xgb_test = (
        _evaluate_xgboost(
            model_dataset
        )
    )
    results.extend(
        [
            xgb_validation,
            xgb_test,
        ]
    )

    print("\n[4/4] Saving evaluation artifacts...")

    metrics_df = (
        evaluation_results_to_dataframe(
            results
        )
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    summary = {
        "feature_dataset": str(
            FEATURE_DATASET_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "feature_count": len(
            model_dataset.feature_columns
        ),
        "models": [
            "logistic_regression",
            "random_forest",
            "xgboost",
        ],
        "splits": [
            "validation",
            "test",
        ],
        "results": [
            {
                "model_name": result.model_name,
                "split_name": result.split_name,
                "metrics": {
                    "precision": (
                        result.metrics.precision
                    ),
                    "recall": (
                        result.metrics.recall
                    ),
                    "f1": result.metrics.f1,
                    "pr_auc": (
                        result.metrics.pr_auc
                    ),
                    "roc_auc": (
                        result.metrics.roc_auc
                    ),
                    "true_negatives": (
                        result.metrics.true_negatives
                    ),
                    "false_positives": (
                        result.metrics.false_positives
                    ),
                    "false_negatives": (
                        result.metrics.false_negatives
                    ),
                    "true_positives": (
                        result.metrics.true_positives
                    ),
                },
            }
            for result in results
        ],
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nEvaluation complete.")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")

    print("\nResults:")
    print(
        metrics_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()