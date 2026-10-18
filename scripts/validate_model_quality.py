
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
    / "reproducible_training_summary.json"
)

EXPECTED_MODELS = {
    "logistic_regression",
    "random_forest",
    "xgboost",
}

# Initial engineering thresholds, based on existing validation baselines.
MINIMUM_METRICS = {
    "pr_auc": 0.30,
    "roc_auc": 0.85,
    "recall": 0.20,
}


def evaluate_model_quality(summary_path: Path) -> tuple[list[dict], list[str]]:
    """Evaluate models using validation metrics only."""
    errors: list[str] = []

    if not summary_path.is_file():
        return [], [f"Training summary not found: {summary_path}"]

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"Unable to read training summary: {exc}"]

    if not isinstance(summary, dict):
        return [], ["Training summary must contain a JSON object."]

    if summary.get("feature_count") != 53:
        errors.append("Training summary must declare exactly 53 features.")

    results = summary.get("results")
    if not isinstance(results, list):
        return [], errors + ["Training summary results must be a list."]

    validation_results: dict[str, dict] = {}

    for result in results:
        if not isinstance(result, dict):
            errors.append("Training summary contains an invalid result entry.")
            continue

        if result.get("split_name") != "validation":
            # Test metrics are deliberately excluded from gate decisions.
            continue

        model_name = result.get("model_name")
        if model_name not in EXPECTED_MODELS:
            errors.append(
                f"Unexpected validation model: {model_name!r}."
            )
            continue

        if model_name in validation_results:
            errors.append(
                f"Duplicate validation result for {model_name}."
            )
            continue

        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            errors.append(
                f"Validation metrics missing for {model_name}."
            )
            continue

        validation_results[model_name] = metrics

    missing_models = EXPECTED_MODELS - set(validation_results)
    if missing_models:
        errors.append(
            "Missing validation results for: "
            + ", ".join(sorted(missing_models))
        )

    evaluations: list[dict] = []

    for model_name in sorted(EXPECTED_MODELS & set(validation_results)):
        metrics = validation_results[model_name]
        failures: list[str] = []
        checked_metrics: dict[str, float] = {}

        for metric_name, minimum in MINIMUM_METRICS.items():
            value = metrics.get(metric_name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                failures.append(
                    f"{metric_name} must be a finite number from 0 to 1."
                )
                continue

            checked_metrics[metric_name] = float(value)

            if value < minimum:
                failures.append(
                    f"{metric_name}={value:.4f} is below {minimum:.2f}."
                )

        evaluations.append(
            {
                "model_name": model_name,
                "status": "FAILED" if failures else "PASSED",
                "metrics": checked_metrics,
                "failures": failures,
            }
        )

    return evaluations, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate classical ML models against quality thresholds."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to the reproducible training summary JSON.",
    )
    args = parser.parse_args()

    evaluations, errors = evaluate_model_quality(args.summary)

    print("MODEL QUALITY GATE")
    print(f"Summary: {args.summary}")

    for evaluation in evaluations:
        metrics = evaluation["metrics"]
        metric_text = ", ".join(
            f"{name}={value:.4f}"
            for name, value in metrics.items()
        )
        print(
            f'{evaluation["model_name"]}: {evaluation["status"]}'
            + (f" ({metric_text})" if metric_text else "")
        )
        for failure in evaluation["failures"]:
            print(f"  - {failure}")

    for error in errors:
        print(f"ERROR: {error}")

    failed_models = [
        evaluation
        for evaluation in evaluations
        if evaluation["status"] != "PASSED"
    ]

    if errors or failed_models or len(evaluations) != len(EXPECTED_MODELS):
        print("MODEL QUALITY GATE: FAILED")
        return 1

    print("MODEL QUALITY GATE: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
