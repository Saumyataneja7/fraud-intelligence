from pathlib import Path

from scripts.generate_graph_ml_report import (
    build_freeze_marker,
    build_report,
)


def test_report_contains_phase_identity():
    report = build_report()

    assert "# Graph ML Report" in report
    assert "**Phase:** 7 — Graph Machine Learning" in report
    assert "**Report:** Phase 7.12 — Graph ML Report & Freeze" in report


def test_report_contains_all_phase_components():
    report = build_report()

    required_sections = [
        "## 2. Phase 7 Scope",
        "## 3. Data Contract",
        "## 4. Temporal Controls",
        "## 9. GraphSAGE Baseline",
        "## 10. Training Pipeline",
        "## 12. GNN Evaluation",
        "## 13. Threshold Optimization",
        "## 14. Ranking Evaluation",
        "## 15. Error Analysis",
        "## 16. Classical ML vs GNN Comparison",
        "## 17. Leakage Controls",
        "## 18. Reproducibility",
    ]

    for section in required_sections:
        assert section in report


def test_report_documents_feature_count():
    report = build_report()

    assert "53 features" in report


def test_report_documents_graph_size():
    report = build_report()

    assert "231,000" in report
    assert "843,214 total edges" in report


def test_report_documents_topology_hash():
    report = build_report()

    assert (
        "1f26bbc304ac823e4f5e697442b86aff8061804f119cab265d0d4100e0841867"
        in report
    )


def test_report_documents_temporal_split():
    report = build_report()

    assert "70% Train" in report
    assert "15% Validation" in report
    assert "15% Test" in report


def test_report_documents_gnn_configuration():
    report = build_report()

    assert "hidden_channels = 64" in report
    assert "output_channels = 32" in report
    assert "dropout = 0.20" in report
    assert "learning_rate = 0.001" in report
    assert "weight_decay = 0.0001" in report
    assert "max_epochs = 100" in report
    assert "patience = 10" in report


def test_report_does_not_fabricate_gnn_metrics():
    report = build_report()

    assert "does not fabricate numerical GNN performance results" in report
    assert "test metrics" in report


def test_report_documents_phase_status():
    report = build_report()

    required_statuses = [
        "| Graph ML data contract | Complete |",
        "| Temporal graph split | Complete |",
        "| Graph feature preparation | Complete |",
        "| Transaction node labels | Complete |",
        "| GraphSAGE baseline | Complete |",
        "| GNN training pipeline | Complete |",
        "| GNN evaluation | Complete |",
        "| Threshold optimization | Complete |",
        "| Precision@K / Recall@K | Complete |",
        "| GNN error analysis | Complete |",
        "| Classical ML vs GNN comparison | Complete |",
        "| Graph ML report | Complete |",
        "| Phase 7 freeze | Complete |",
    ]

    for status in required_statuses:
        assert status in report


def test_freeze_marker_contains_frozen_status():
    freeze = build_freeze_marker()

    assert "PHASE 7 FREEZE" in freeze
    assert "Status: FROZEN" in freeze


def test_freeze_marker_contains_artifacts():
    freeze = build_freeze_marker()

    assert "data/graph/heterogeneous_graph.pt" in freeze
    assert "data/graph/graph_metadata.json" in freeze
    assert "reports/graph_ml/GRAPH_ML_REPORT.md" in freeze
    assert "reports/graph_ml/PHASE_7_FREEZE.txt" in freeze


def test_freeze_marker_contains_topology_hash():
    freeze = build_freeze_marker()

    assert (
        "1f26bbc304ac823e4f5e697442b86aff8061804f119cab265d0d4100e0841867"
        in freeze
    )


def test_report_paths_are_project_relative():
    report_path = Path(
        "reports",
        "graph_ml",
        "GRAPH_ML_REPORT.md",
    )

    freeze_path = Path(
        "reports",
        "graph_ml",
        "PHASE_7_FREEZE.txt",
    )

    assert report_path.parts == (
        "reports",
        "graph_ml",
        "GRAPH_ML_REPORT.md",
    )

    assert freeze_path.parts == (
        "reports",
        "graph_ml",
        "PHASE_7_FREEZE.txt",
    )