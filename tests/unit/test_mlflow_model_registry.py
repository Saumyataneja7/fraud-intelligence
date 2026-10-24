from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression


def test_registered_model_can_be_loaded_and_predict() -> None:
    original_tracking_uri = mlflow.get_tracking_uri()
    original_registry_uri = mlflow.get_registry_uri()

    try:
        with tempfile.TemporaryDirectory(
            prefix="fraud-mlflow-test-"
        ) as tmp:
            db_path = Path(tmp) / "tracking.db"
            uri = f"sqlite:///{db_path}"

            mlflow.set_tracking_uri(uri)
            mlflow.set_registry_uri(uri)
            mlflow.set_experiment("registry-regression-test")

            X = np.array(
                [[0.0], [1.0], [2.0], [3.0]],
                dtype=np.float64,
            )
            y = np.array([0, 0, 1, 1])

            classifier = LogisticRegression().fit(X, y)

            with mlflow.start_run():
                model_info = mlflow.sklearn.log_model(
                    classifier,
                    name="test_classifier",
                    input_example=X[:1],
                )

            model_name = "fraud-intelligence-test-classifier"
            registered = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=model_name,
            )

            client = MlflowClient()
            version = client.get_model_version(
                name=model_name,
                version=registered.version,
            )

            assert version.status == "READY"

            loaded = mlflow.pyfunc.load_model(
                f"models:/{model_name}/{registered.version}"
            )
            predictions = loaded.predict(X[:2])

            assert len(predictions) == 2
            assert set(np.asarray(predictions).tolist()).issubset(
                {0, 1}
            )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)
        mlflow.set_registry_uri(original_registry_uri)
