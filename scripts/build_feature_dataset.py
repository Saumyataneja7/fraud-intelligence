from pathlib import Path

import pandas as pd

from fraud_intelligence.features.dataset import (
    materialize_feature_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "synthetic"
    / "transactions.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
)


def main() -> None:
    print("Loading transactions...")
    print(f"Source: {RAW_PATH}")

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw transaction dataset not found: {RAW_PATH}"
        )

    transactions = pd.read_parquet(RAW_PATH)

    print(f"Input shape: {transactions.shape}")

    artifacts = materialize_feature_dataset(
        transactions=transactions,
        output_dir=OUTPUT_DIR,
        source_path=RAW_PATH,
    )

    print("\nFeature dataset created successfully.")
    print(f"Dataset:  {artifacts.dataset_path}")
    print(f"Metadata: {artifacts.metadata_path}")


if __name__ == "__main__":
    main()