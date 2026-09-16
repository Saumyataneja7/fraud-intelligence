from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from fraud_intelligence.features.contracts import FeatureContract
from fraud_intelligence.features.leakage_validation import (
    ENGINEERED_FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMNS,
    assert_feature_dataset_is_leakage_safe,
)
from fraud_intelligence.features.pipeline import build_feature_dataset


@dataclass(frozen=True)
class FeatureDatasetArtifacts:
    dataset_path: Path
    metadata_path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_model_feature_columns(
    columns: Iterable[str],
) -> list[str]:
    """
    Return engineered columns intended for model training.

    Raw transaction columns, identifiers, and targets are deliberately
    excluded. The materialized feature dataset itself still retains those
    columns for investigation, auditability, and downstream graph/API use.
    """

    columns = list(columns)

    excluded = IDENTIFIER_COLUMNS | TARGET_COLUMNS

    return [
        column
        for column in columns
        if (
            column in ENGINEERED_FEATURE_COLUMNS
            and column not in excluded
        )
    ]


def validate_model_feature_columns(
    columns: Iterable[str],
) -> None:
    """
    Validate the final model feature column contract.
    """

    columns = list(columns)

    FeatureContract().validate_feature_columns(columns)

    forbidden = (
        set(columns)
        & (IDENTIFIER_COLUMNS | TARGET_COLUMNS)
    )

    if forbidden:
        raise ValueError(
            "Model feature set contains forbidden columns: "
            f"{sorted(forbidden)}"
        )

    unknown = set(columns) - ENGINEERED_FEATURE_COLUMNS

    if unknown:
        raise ValueError(
            "Model feature set contains unexpected columns: "
            f"{sorted(unknown)}"
        )


def build_validated_feature_dataset(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the complete Phase 4 feature dataset and validate it
    before returning it.

    Returns
    -------
    features:
        Complete dataset containing identifiers, engineered features,
        and targets.

    model_features:
        Columns approved for downstream model training.
    """

    features = build_feature_dataset(transactions)

    model_features = get_model_feature_columns(
        features.columns
    )

    validate_model_feature_columns(model_features)

    assert_feature_dataset_is_leakage_safe(
        original=transactions,
        features=features,
        model_features=model_features,
    )

    return features, model_features


def build_feature_metadata(
    features: pd.DataFrame,
    model_features: list[str],
    source_path: Path | None = None,
) -> dict:
    """
    Create metadata describing the materialized feature dataset.
    """

    fraud_count = None

    if "is_fraud" in features.columns:
        fraud_count = int(
            features["is_fraud"].sum()
        )

    metadata = {
        "dataset_name": "fraud-intelligence-feature-dataset",
        "dataset_version": "1.0.0",
        "phase": "4.9",
        "row_count": int(len(features)),
        "column_count": int(len(features.columns)),
        "identifier_columns": [
            column
            for column in features.columns
            if column in IDENTIFIER_COLUMNS
        ],
        "target_columns": [
            column
            for column in features.columns
            if column in TARGET_COLUMNS
        ],
        "model_feature_columns": model_features,
        "model_feature_count": len(model_features),
        "engineered_feature_columns": [
            column
            for column in features.columns
            if column in ENGINEERED_FEATURE_COLUMNS
        ],
        "engineered_feature_count": len(
            [
                column
                for column in features.columns
                if column in ENGINEERED_FEATURE_COLUMNS
            ]
        ),
        "fraud_count": fraud_count,
        "fraud_rate": (
            fraud_count / len(features)
            if fraud_count is not None and len(features) > 0
            else None
        ),
        "source_path": (
            str(source_path)
            if source_path is not None
            else None
        ),
        "dtypes": {
            column: str(dtype)
            for column, dtype in features.dtypes.items()
        },
    }

    return metadata


def materialize_feature_dataset(
    transactions: pd.DataFrame,
    output_dir: str | Path,
    source_path: Path | None = None,
) -> FeatureDatasetArtifacts:
    """
    Build, validate, and persist the Phase 4 feature dataset.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    features, model_features = (
        build_validated_feature_dataset(transactions)
    )

    dataset_path = (
        output_dir / "feature_dataset.parquet"
    )

    metadata_path = (
        output_dir / "feature_dataset_metadata.json"
    )

    features.to_parquet(
        dataset_path,
        index=False,
    )

    metadata = build_feature_metadata(
        features=features,
        model_features=model_features,
        source_path=source_path,
    )

    metadata["sha256"] = _sha256_file(
        dataset_path
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return FeatureDatasetArtifacts(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
    )


__all__ = [
    "FeatureDatasetArtifacts",
    "get_model_feature_columns",
    "validate_model_feature_columns",
    "build_validated_feature_dataset",
    "build_feature_metadata",
    "materialize_feature_dataset",
]