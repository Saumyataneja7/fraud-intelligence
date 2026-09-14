from __future__ import annotations

import hashlib
from pathlib import Path


DATASET_DIR = Path("data/raw/synthetic")
OUTPUT_FILE = Path("reports/data_quality/DATASET_MANIFEST.txt")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    files = sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file()
    )

    if not files:
        raise RuntimeError(
            f"No dataset files found in {DATASET_DIR}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as manifest:
        manifest.write(
            "Fraud Intelligence Synthetic Dataset Manifest\n"
        )
        manifest.write("Dataset Version: 1.0.0\n")
        manifest.write("Generation Seed: 42\n")
        manifest.write("\n")

        for path in files:
            digest = sha256_file(path)

            relative_path = path

            manifest.write(
                f"{digest}  {relative_path}\n"
            )

    print(f"Manifest created: {OUTPUT_FILE}")
    print(f"Files hashed: {len(files)}")


if __name__ == "__main__":
    main()