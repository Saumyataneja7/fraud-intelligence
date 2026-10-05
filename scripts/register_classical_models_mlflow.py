from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from fraud_intelligence.ml.dataset import prepare_model_dataset
from fraud_intelligence.ml.logistic_regression import (
    predict_logistic_regression_proba,
    train_logistic_regression,
)
from fraud_intelligence.ml.random_forest import (
    predict_random_forest_proba,
    train_random_forest,
)
from fraud_intelligence.ml.splits import (
    load_and_split_feature_dataset,
)
from fraud_intelligence.ml.xgboost import (
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

EXPERIMENT_NAME = "fraud-intelligence-classical-ml"
REGISTRY_EXPERIMENT_NAME = "fraud-intelligence-model-registry"

DATASET_NAME = "fraud-intelligence-synthetic"
DATASET_VERSION = "1.0.0"

MODELS = {
    "logistic_regression": (
        "fraud-intelligence-logistic-regression",
    ),
    "random_forest": (
        "fraud-intelligence-random-forest",
    ),
    "xgboost": (
        "fraud-intelligence-xgboost",
    ),
}


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _get_source_runs() -> dict[str, object]:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment not found: {EXPERIMENT_NAME}"
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="list",
    )

    source_runs: dict[str, object] = {}

    for run in runs:
        model_name = run.data.params.get("model_name")

        if model_name in MODELS:
            source_runs[model_name] = run

    missing = set(MODELS) - set(source_runs)

    if missing:
        raise RuntimeError(
            "Missing Phase 12.2 MLflow runs for: "
            + ", ".join(sorted(missing))
        )

    return source_runs


def _log_common_metadata(
    model_name: str,
    source_run,
    feature_count: int,
    feature_columns: tuple[str, ...],
    split_counts: dict[str, int],
) -> None:
    mlflow.set_tags(
        {
            "phase": "12.3",
            "model_name": model_name,
            "registry_status": "Candidate",
            "dataset_name": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "source_experiment": EXPERIMENT_NAME,
            "source_run_id": source_run.info.run_id,
            "git_commit": _get_git_commit(),
        }
    )

    mlflow.log_params(
        {
            "feature_count": feature_count,
            "feature_columns": ",".join(feature_columns),
            "train_rows": split_counts["train"],
            "validation_rows": split_counts["validation"],
            "test_rows": split_counts["test"],
        }
    )

    mlflow.log_metrics(
        {
            "source_test_pr_auc": source_run.data.metrics[
                "test_pr_auc"
            ],
            "source_test_f1": source_run.data.metrics[
                "test_f1"
            ],
        }
    )


def _register_model(
    model_name: str,
    registered_name: str,
    model,
    model_input,
    model_output,
    source_run,
    mlflow_flavor: str,
    split_counts: dict[str, int],
) -> None:
    signature = infer_signature(
        model_input,
        model_output,
    )

    with mlflow.start_run(
        run_name=f"registry-{model_name}",
        experiment_id=mlflow.get_experiment_by_name(
            REGISTRY_EXPERIMENT_NAME
        ).experiment_id,
    ):
        _log_common_metadata(
            model_name=model_name,
            source_run=source_run,
            feature_count=len(model_input.columns),
            feature_columns=tuple(model_input.columns),
            split_counts=split_counts,
        )

        if mlflow_flavor == "sklearn":
            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
            )
        elif mlflow_flavor == "xgboost":
            model_info = mlflow.xgboost.log_model(
                model,
                name="model",
                signature=signature,
            )
        else:
            raise ValueError(
                f"Unsupported MLflow flavor: {mlflow_flavor}"
            )

        model_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=registered_name,
        )

        client = MlflowClient()

        client.set_model_version_tag(
            name=registered_name,
            version=model_version.version,
            key="lifecycle_stage",
            value="Candidate",
        )

        client.set_model_version_tag(
            name=registered_name,
            version=model_version.version,
            key="source_run_id",
            value=source_run.info.run_id,
        )

        client.set_model_version_tag(
            name=registered_name,
            version=model_version.version,
            key="dataset_version",
            value=DATASET_VERSION,
        )

        client.set_model_version_tag(
            name=registered_name,
            version=model_version.version,
            key="phase",
            value="12.3",
        )

        client.set_registered_model_alias(
            registered_name,
            "candidate",
            model_version.version,
        )

        print(
            f"Registered {model_name}: "
            f"{registered_name} "
            f"version={model_version.version} "
            f"alias=candidate"
        )


def main() -> None:
    print("=" * 72)
    print("Phase 12.3 — MLflow Model Registry")
    print("=" * 72)

    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Feature dataset not found: "
            f"{FEATURE_DATASET_PATH}"
        )

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB_PATH}"
    )

    mlflow.set_registry_uri(
        f"sqlite:///{MLFLOW_DB_PATH}"
    )

    mlflow.set_experiment(
        REGISTRY_EXPERIMENT_NAME
    )

    print("\n[1/4] Loading Phase 12.2 source runs...")

    source_runs = _get_source_runs()

    for model_name, run in source_runs.items():
        print(
            f"{model_name}: "
            f"{run.info.run_id}"
        )

    print("\n[2/4] Loading temporal splits...")

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

    print("\n[3/4] Preparing model dataset...")

    model_dataset = prepare_model_dataset(
        splits
    )

    split_counts = {
        "train": len(splits.train),
        "validation": len(splits.validation),
        "test": len(splits.test),
    }

    print(
        f"Model features: "
        f"{len(model_dataset.feature_columns)}"
    )

    print("\n[4/4] Registering models...")

    logistic_model = train_logistic_regression(
        model_dataset.X_train,
        model_dataset.y_train,
    )

    logistic_output = predict_logistic_regression_proba(
        logistic_model,
        model_dataset.X_validation,
    )

    _register_model(
        model_name="logistic_regression",
        registered_name=MODELS["logistic_regression"][0],
        model=logistic_model.pipeline,
        model_input=model_dataset.X_validation,
        model_output=logistic_output,
        source_run=source_runs["logistic_regression"],
        mlflow_flavor="sklearn",
        split_counts=split_counts,
    )

    random_forest_model = train_random_forest(
        model_dataset.X_train,
        model_dataset.y_train,
    )

    random_forest_output = predict_random_forest_proba(
        random_forest_model,
        model_dataset.X_validation,
    )

    _register_model(
        model_name="random_forest",
        registered_name=MODELS["random_forest"][0],
        model=random_forest_model.classifier,
        model_input=model_dataset.X_validation,
        model_output=random_forest_output,
        source_run=source_runs["random_forest"],
        mlflow_flavor="sklearn",
        split_counts=split_counts,
    )

    xgboost_model = train_xgboost(
        model_dataset.X_train,
        model_dataset.y_train,
    )

    xgboost_output = predict_xgboost_proba(
        xgboost_model,
        model_dataset.X_validation,
    )

    _register_model(
        model_name="xgboost",
        registered_name=MODELS["xgboost"][0],
        model=xgboost_model.classifier,
        model_input=model_dataset.X_validation,
        model_output=xgboost_output,
        source_run=source_runs["xgboost"],
        mlflow_flavor="xgboost",
        split_counts=split_counts,
    )

    print("\nMLflow model registration complete.")


if __name__ == "__main__":
    main()
