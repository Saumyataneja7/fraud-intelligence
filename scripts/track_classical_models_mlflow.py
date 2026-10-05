from __future__ import annotations

import json
import subprocess
from pathlib import Path

import mlflow

from fraud_intelligence.ml.dataset import prepare_model_dataset
from fraud_intelligence.ml.evaluation import evaluate_model
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
from fraud_intelligence.ml.splits import load_and_split_feature_dataset
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

EVALUATION_DIR = PROJECT_ROOT / "reports" / "model_evaluation"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "fraud-intelligence-classical-ml"


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _flatten_metrics(result) -> dict[str, float]:
    return {
        "precision": float(result.metrics.precision),
        "recall": float(result.metrics.recall),
        "f1": float(result.metrics.f1),
        "pr_auc": float(result.metrics.pr_auc),
        "roc_auc": float(result.metrics.roc_auc),
        "true_negatives": float(result.metrics.true_negatives),
        "false_positives": float(result.metrics.false_positives),
        "false_negatives": float(result.metrics.false_negatives),
        "true_positives": float(result.metrics.true_positives),
    }


def _log_model_metadata(model, model_name: str) -> None:
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("feature_count", len(model.feature_columns))
    mlflow.log_param(
        "feature_columns",
        json.dumps(list(model.feature_columns)),
    )

    for key, value in vars(model.config).items():
        if value is not None:
            mlflow.log_param(f"model_config_{key}", value)

    if hasattr(model, "class_weights"):
        mlflow.log_param(
            "class_weights",
            json.dumps(model.class_weights),
        )

    if hasattr(model, "scale_pos_weight"):
        mlflow.log_param(
            "scale_pos_weight",
            float(model.scale_pos_weight),
        )


def _log_common_metadata(model_dataset, git_commit: str) -> None:
    mlflow.log_params(
        {
            "dataset_name": "fraud-intelligence-synthetic",
            "dataset_version": "1.0.0",
            "feature_dataset": str(
                FEATURE_DATASET_PATH.relative_to(PROJECT_ROOT)
            ),
            "feature_count": len(model_dataset.feature_columns),
            "train_rows": len(model_dataset.X_train),
            "validation_rows": len(model_dataset.X_validation),
            "test_rows": len(model_dataset.X_test),
            "git_commit": git_commit,
        }
    )


def _log_evaluation_artifacts() -> None:
    if EVALUATION_DIR.exists():
        mlflow.log_artifacts(
            str(EVALUATION_DIR),
            artifact_path="evaluation_reports",
        )


def _track_model(
    model_name: str,
    model_dataset,
    train_fn,
    predict_fn,
    predict_proba_fn,
    git_commit: str,
) -> None:
    with mlflow.start_run(run_name=model_name):
        model = train_fn(
            model_dataset.X_train,
            model_dataset.y_train,
        )

        validation_pred = predict_fn(
            model,
            model_dataset.X_validation,
        )
        validation_proba = predict_proba_fn(
            model,
            model_dataset.X_validation,
        )

        test_pred = predict_fn(
            model,
            model_dataset.X_test,
        )
        test_proba = predict_proba_fn(
            model,
            model_dataset.X_test,
        )

        validation_result = evaluate_model(
            model_name=model_name,
            split_name="validation",
            y_true=model_dataset.y_validation,
            y_pred=validation_pred,
            y_proba=validation_proba,
        )

        test_result = evaluate_model(
            model_name=model_name,
            split_name="test",
            y_true=model_dataset.y_test,
            y_pred=test_pred,
            y_proba=test_proba,
        )

        _log_common_metadata(model_dataset, git_commit)
        _log_model_metadata(model, model_name)

        mlflow.log_metrics(
            {
                f"validation_{key}": value
                for key, value in _flatten_metrics(validation_result).items()
            }
        )

        mlflow.log_metrics(
            {
                f"test_{key}": value
                for key, value in _flatten_metrics(test_result).items()
            }
        )

        _log_evaluation_artifacts()

        print(
            f"Tracked {model_name}: "
            f"test_pr_auc={test_result.metrics.pr_auc:.6f}, "
            f"test_f1={test_result.metrics.f1:.6f}"
        )


def main() -> None:
    print("=" * 72)
    print("Phase 12.2 — MLflow Experiment Tracking")
    print("=" * 72)

    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_DATASET_PATH}"
        )

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB_PATH}"
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\n[1/3] Loading temporal splits...")

    splits = load_and_split_feature_dataset(
        FEATURE_DATASET_PATH
    )

    print(f"Train:      {len(splits.train):,} rows")
    print(f"Validation: {len(splits.validation):,} rows")
    print(f"Test:       {len(splits.test):,} rows")

    print("\n[2/3] Preparing model dataset...")

    model_dataset = prepare_model_dataset(splits)

    print(
        f"Model features: {len(model_dataset.feature_columns)}"
    )

    git_commit = _get_git_commit()

    print("\n[3/3] Tracking models...")

    _track_model(
        "logistic_regression",
        model_dataset,
        train_logistic_regression,
        predict_logistic_regression,
        predict_logistic_regression_proba,
        git_commit,
    )

    _track_model(
        "random_forest",
        model_dataset,
        train_random_forest,
        predict_random_forest,
        predict_random_forest_proba,
        git_commit,
    )

    _track_model(
        "xgboost",
        model_dataset,
        train_xgboost,
        predict_xgboost,
        predict_xgboost_proba,
        git_commit,
    )

    print("\nMLflow tracking complete.")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Tracking database: {MLFLOW_DB_PATH}")


if __name__ == "__main__":
    main()
