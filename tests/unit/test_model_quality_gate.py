
import json

from scripts.validate_model_quality import evaluate_model_quality


def make_summary():
    models = {
        "logistic_regression": {
            "pr_auc": 0.33,
            "roc_auc": 0.90,
            "recall": 0.72,
        },
        "random_forest": {
            "pr_auc": 0.39,
            "roc_auc": 0.88,
            "recall": 0.24,
        },
        "xgboost": {
            "pr_auc": 0.37,
            "roc_auc": 0.87,
            "recall": 0.42,
        },
    }

    results = []
    for name, metrics in models.items():
        results.append({
            "model_name": name,
            "split_name": "validation",
            "metrics": metrics,
        })

    return {
        "feature_count": 53,
        "results": results,
    }


def write_summary(tmp_path, summary):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_all_models_pass(tmp_path):
    path = write_summary(tmp_path, make_summary())

    evaluations, errors = evaluate_model_quality(path)

    assert errors == []
    assert len(evaluations) == 3
    assert all(item["status"] == "PASSED" for item in evaluations)


def test_model_below_threshold_fails(tmp_path):
    summary = make_summary()
    result = next(
        item for item in summary["results"]
        if item["model_name"] == "xgboost"
    )
    result["metrics"]["pr_auc"] = 0.10

    evaluations, errors = evaluate_model_quality(
        write_summary(tmp_path, summary)
    )

    xgboost = next(
        item for item in evaluations
        if item["model_name"] == "xgboost"
    )
    assert errors == []
    assert xgboost["status"] == "FAILED"
    assert any("pr_auc" in failure for failure in xgboost["failures"])


def test_missing_metric_fails_closed(tmp_path):
    summary = make_summary()
    result = next(
        item for item in summary["results"]
        if item["model_name"] == "random_forest"
    )
    del result["metrics"]["roc_auc"]

    evaluations, errors = evaluate_model_quality(
        write_summary(tmp_path, summary)
    )

    random_forest = next(
        item for item in evaluations
        if item["model_name"] == "random_forest"
    )
    assert errors == []
    assert random_forest["status"] == "FAILED"
    assert any("roc_auc" in failure for failure in random_forest["failures"])


def test_test_split_metrics_cannot_make_model_pass(tmp_path):
    summary = make_summary()
    summary["results"] = [
        item for item in summary["results"]
        if item["model_name"] != "xgboost"
    ]
    summary["results"].append({
        "model_name": "xgboost",
        "split_name": "test",
        "metrics": {
            "pr_auc": 1.0,
            "roc_auc": 1.0,
            "recall": 1.0,
        },
    })

    evaluations, errors = evaluate_model_quality(
        write_summary(tmp_path, summary)
    )

    assert any("xgboost" in error for error in errors)
    assert not any(
        item["model_name"] == "xgboost" and item["status"] == "PASSED"
        for item in evaluations
    )


def test_non_finite_metric_fails(tmp_path):
    summary = make_summary()
    result = next(
        item for item in summary["results"]
        if item["model_name"] == "logistic_regression"
    )
    result["metrics"]["pr_auc"] = float("nan")

    evaluations, errors = evaluate_model_quality(
        write_summary(tmp_path, summary)
    )

    logistic = next(
        item for item in evaluations
        if item["model_name"] == "logistic_regression"
    )
    assert errors == []
    assert logistic["status"] == "FAILED"
    assert any("finite number" in failure for failure in logistic["failures"])


def test_missing_summary_fails(tmp_path):
    evaluations, errors = evaluate_model_quality(
        tmp_path / "does-not-exist.json"
    )

    assert evaluations == []
    assert errors
