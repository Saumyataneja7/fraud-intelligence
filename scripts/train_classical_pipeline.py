from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from fraud_intelligence.ml.dataset import prepare_model_dataset
from fraud_intelligence.ml.evaluation import (
    EvaluationResult,
    evaluate_model,
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

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "fraud-intelligence-reproducible-training"

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
    / "reproducible_training_summary.json"
)

DATASET_NAME = "fraud-intelligence-synthetic"
DATASET_VERSION = "1.0.0"

MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "xgboost",
)

CANONICAL_FEATURE_COLUMNS = (
    # Phase 4.1 — Transaction + Temporal
    "log_amount",
    "is_card_payment",
    "is_bank_transfer",
    "is_wallet",
    "is_online",
    "is_purchase",
    "is_transfer",
    "is_withdrawal",
    "is_payment",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "is_weekend",
    "is_night",

    # Phase 4.2 — Customer Historical
    "customer_txn_count_before",
    "customer_amount_sum_before",
    "customer_amount_mean_before",
    "customer_amount_median_before",
    "customer_amount_max_before",
    "customer_unique_merchants_before",
    "customer_unique_devices_before",
    "customer_unique_ips_before",

    # Phase 4.3 — Velocity
    "customer_txn_count_1m",
    "customer_txn_count_5m",
    "customer_txn_count_15m",
    "customer_amount_sum_1m",
    "customer_amount_sum_5m",
    "customer_amount_sum_15m",
    "customer_unique_merchants_5m",
    "customer_unique_devices_5m",
    "customer_unique_ips_5m",

    # Phase 4.4 — Entity Historical
    "account_txn_count_before",
    "card_txn_count_before",
    "merchant_txn_count_before",
    "device_txn_count_before",
    "ip_txn_count_before",
    "account_amount_sum_before",
    "card_amount_sum_before",
    "merchant_amount_sum_before",
    "device_amount_sum_before",
    "ip_amount_sum_before",

    # Phase 4.5 — Novelty
    "is_new_device",
    "is_new_ip",
    "is_new_merchant",
    "is_new_card",

    # Phase 4.6 — Behavioral Deviation
    "amount_vs_customer_mean",
    "amount_vs_customer_median",
    "amount_vs_customer_max",
    "amount_deviation_from_customer_mean",
    "is_new_payment_method",
    "is_new_transaction_type",
)

def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _configure_mlflow() -> None:
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

def _apply_canonical_feature_order(model_dataset):
    actual_features = tuple(model_dataset.feature_columns)
    expected_features = CANONICAL_FEATURE_COLUMNS

    if len(expected_features) != 53:
        raise ValueError(
            "Canonical feature contract must contain exactly 53 features."
        )

    missing = set(expected_features) - set(actual_features)
    unexpected = set(actual_features) - set(expected_features)

    if missing:
        raise ValueError(
            f"Missing canonical model features: {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            f"Unexpected model features: {sorted(unexpected)}"
        )

    return model_dataset.__class__(
        X_train=model_dataset.X_train.loc[:, expected_features].copy(),
        y_train=model_dataset.y_train,
        X_validation=model_dataset.X_validation.loc[:, expected_features].copy(),
        y_validation=model_dataset.y_validation,
        X_test=model_dataset.X_test.loc[:, expected_features].copy(),
        y_test=model_dataset.y_test,
        train_ids=model_dataset.train_ids,
        validation_ids=model_dataset.validation_ids,
        test_ids=model_dataset.test_ids,
        feature_columns=expected_features,
    )

def _log_common_metadata(
    model_name: str,
    model_dataset,
    split_counts: dict[str, int],
) -> None:
    mlflow.set_tags(
        {
            "phase": "12.4",
            "pipeline": "reproducible-classical-training",
            "model_name": model_name,
            "dataset_name": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "git_commit": _get_git_commit(),
        }
    )

    mlflow.log_params(
        {
            "feature_count": model_dataset.n_features,
            "feature_columns": ",".join(
                model_dataset.feature_columns
            ),
            "train_rows": split_counts["train"],
            "validation_rows": split_counts["validation"],
            "test_rows": split_counts["test"],
        }
    )


def _log_evaluation(
    result: EvaluationResult,
) -> None:
    metrics = result.metrics

    prefix = result.split_name

    mlflow.log_metrics(
        {
            f"{prefix}_precision": metrics.precision,
            f"{prefix}_recall": metrics.recall,
            f"{prefix}_f1": metrics.f1,
            f"{prefix}_pr_auc": metrics.pr_auc,
            f"{prefix}_roc_auc": metrics.roc_auc,
            f"{prefix}_true_negatives": (
                metrics.true_negatives
            ),
            f"{prefix}_false_positives": (
                metrics.false_positives
            ),
            f"{prefix}_false_negatives": (
                metrics.false_negatives
            ),
            f"{prefix}_true_positives": (
                metrics.true_positives
            ),
        }
    )


def _save_summary(
    results: list[EvaluationResult],
    model_dataset,
    split_counts: dict[str, int],
) -> None:
    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "feature_dataset": str(
            FEATURE_DATASET_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "feature_count": model_dataset.n_features,
        "models": list(MODEL_NAMES),
        "split_counts": split_counts,
        "git_commit": _get_git_commit(),
        "results": [
            {
                "model_name": result.model_name,
                "split_name": result.split_name,
                "metrics": asdict(result.metrics),
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


def _train_logistic_regression(
    model_dataset,
    split_counts: dict[str, int],
) -> list[EvaluationResult]:
    model = train_logistic_regression(
        model_dataset.X_train,
        model_dataset.y_train,
    )

    validation_pred = predict_logistic_regression(
        model,
        model_dataset.X_validation,
    )
    validation_proba = predict_logistic_regression_proba(
        model,
        model_dataset.X_validation,
    )

    test_pred = predict_logistic_regression(
        model,
        model_dataset.X_test,
    )
    test_proba = predict_logistic_regression_proba(
        model,
        model_dataset.X_test,
    )

    results = [
        evaluate_model(
            model_name="logistic_regression",
            split_name="validation",
            y_true=model_dataset.y_validation,
            y_pred=validation_pred,
            y_proba=validation_proba,
        ),
        evaluate_model(
            model_name="logistic_regression",
            split_name="test",
            y_true=model_dataset.y_test,
            y_pred=test_pred,
            y_proba=test_proba,
        ),
    ]

    with mlflow.start_run(
        run_name="reproducible-logistic-regression",
    ):
        _log_common_metadata(
            "logistic_regression",
            model_dataset,
            split_counts,
        )

        for result in results:
            _log_evaluation(result)

        mlflow.sklearn.log_model(
            model.pipeline,
            name="model",
        )

    return results


def _train_random_forest(
    model_dataset,
    split_counts: dict[str, int],
) -> list[EvaluationResult]:
    model = train_random_forest(
        model_dataset.X_train,
        model_dataset.y_train,
    )

    validation_pred = predict_random_forest(
        model,
        model_dataset.X_validation,
    )
    validation_proba = predict_random_forest_proba(
        model,
        model_dataset.X_validation,
    )

    test_pred = predict_random_forest(
        model,
        model_dataset.X_test,
    )
    test_proba = predict_random_forest_proba(
        model,
        model_dataset.X_test,
    )

    results = [
        evaluate_model(
            model_name="random_forest",
            split_name="validation",
            y_true=model_dataset.y_validation,
            y_pred=validation_pred,
            y_proba=validation_proba,
        ),
        evaluate_model(
            model_name="random_forest",
            split_name="test",
            y_true=model_dataset.y_test,
            y_pred=test_pred,
            y_proba=test_proba,
        ),
    ]

    with mlflow.start_run(
        run_name="reproducible-random-forest",
    ):
        _log_common_metadata(
            "random_forest",
            model_dataset,
            split_counts,
        )

        for result in results:
            _log_evaluation(result)

        mlflow.sklearn.log_model(
            model.classifier,
            name="model",
        )

    return results


def _train_xgboost(
    model_dataset,
    split_counts: dict[str, int],
) -> list[EvaluationResult]:
    model = train_xgboost(
        model_dataset.X_train,
        model_dataset.y_train,
    )

    validation_pred = predict_xgboost(
        model,
        model_dataset.X_validation,
    )
    validation_proba = predict_xgboost_proba(
        model,
        model_dataset.X_validation,
    )

    test_pred = predict_xgboost(
        model,
        model_dataset.X_test,
    )
    test_proba = predict_xgboost_proba(
        model,
        model_dataset.X_test,
    )

    results = [
        evaluate_model(
            model_name="xgboost",
            split_name="validation",
            y_true=model_dataset.y_validation,
            y_pred=validation_pred,
            y_proba=validation_proba,
        ),
        evaluate_model(
            model_name="xgboost",
            split_name="test",
            y_true=model_dataset.y_test,
            y_pred=test_pred,
            y_proba=test_proba,
        ),
    ]

    with mlflow.start_run(
        run_name="reproducible-xgboost",
    ):
        _log_common_metadata(
            "xgboost",
            model_dataset,
            split_counts,
        )

        for result in results:
            _log_evaluation(result)

        mlflow.xgboost.log_model(
            model.classifier,
            name="model",
        )

    return results


def main() -> None:
    print("=" * 72)
    print("Phase 12.4 — Reproducible Classical Training Pipeline")
    print("=" * 72)

    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Feature dataset not found: "
            f"{FEATURE_DATASET_PATH}"
        )

    _configure_mlflow()

    print("\n[1/5] Loading temporal splits...")

    splits = load_and_split_feature_dataset(
        FEATURE_DATASET_PATH
    )

    print(
        f"Train:      {splits.train_rows:,} rows"
    )
    print(
        f"Validation: {splits.validation_rows:,} rows"
    )
    print(
        f"Test:       {splits.test_rows:,} rows"
    )

    split_counts = {
        "train": splits.train_rows,
        "validation": splits.validation_rows,
        "test": splits.test_rows,
    }

    print("\n[2/5] Preparing model dataset...")

    model_dataset = prepare_model_dataset(
        splits
    )

    model_dataset = _apply_canonical_feature_order(
        model_dataset
    )

    print(
        f"Model features: "
        f"{model_dataset.n_features}"
    )

    print("\n[3/5] Training Logistic Regression...")

    lr_results = _train_logistic_regression(
        model_dataset,
        split_counts,
    )

    print("\n[4/5] Training Random Forest and XGBoost...")

    rf_results = _train_random_forest(
        model_dataset,
        split_counts,
    )

    xgb_results = _train_xgboost(
        model_dataset,
        split_counts,
    )

    all_results = (
        lr_results
        + rf_results
        + xgb_results
    )

    _save_summary(
        all_results,
        model_dataset,
        split_counts,
    )

    print("\n[5/5] Training pipeline complete.")

    for result in all_results:
        print(
            f"{result.model_name:20s} "
            f"{result.split_name:10s} "
            f"PR-AUC={result.metrics.pr_auc:.6f} "
            f"F1={result.metrics.f1:.6f}"
        )

    print(
        "\nSummary:",
        SUMMARY_PATH.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
