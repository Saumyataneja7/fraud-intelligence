from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraud_intelligence.graph.contracts import (
    EDGE_TYPES,
    NODE_TYPES,
)
from fraud_intelligence.graph.materialization import (
    GRAPH_ARTIFACT_VERSION,
)
from scripts.generate_graph_report import (
    build_freeze,
    build_report,
)


def _metadata() -> dict:
    return {
        "artifact": {
            "name": (
                "fraud-intelligence-heterogeneous-graph"
            ),
            "version": GRAPH_ARTIFACT_VERSION,
            "format": (
                "torch-geometric-heterodata"
            ),
        },
        "schema": {
            "node_types": list(NODE_TYPES),
            "edge_types": list(EDGE_TYPES),
        },
        "graph": {
            "total_nodes": 14,
            "total_edges": 11,
            "node_counts": {
                node_type: 2
                for node_type in NODE_TYPES
            },
            "edge_counts": {
                edge_type: 1
                for edge_type in EDGE_TYPES
            },
        },
        "topology": {
            "sha256": "a" * 64,
        },
        "statistics": {
            "degree_statistics": [
                {
                    "node_type": node_type,
                    "node_count": 2,
                    "min_degree": 0.0,
                    "max_degree": 1.0,
                    "mean_degree": 0.5,
                    "median_degree": 0.5,
                    "p95_degree": 1.0,
                }
                for node_type in NODE_TYPES
            ],
            "isolated_node_counts": {
                node_type: 1
                for node_type in NODE_TYPES
            },
            "connected_components": {
                "component_count": 8,
                "largest_component_size": 7,
                "largest_component_fraction": 0.5,
                "size_distribution": [
                    7,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
            },
        },
    }


def test_build_report_contains_phase_information() -> None:
    report = build_report(_metadata())

    assert "# Graph Construction Report" in report
    assert "Phase 6.10" in report
    assert "Phase 6 — Graph Construction" in report


def test_report_contains_all_node_types() -> None:
    report = build_report(_metadata())

    for node_type in NODE_TYPES:
        assert f"`{node_type}`" in report


def test_report_contains_all_edge_types() -> None:
    report = build_report(_metadata())

    for edge_type in EDGE_TYPES:
        assert f"`{edge_type}`" in report


def test_report_contains_artifact_information() -> None:
    report = build_report(_metadata())

    assert "heterogeneous_graph.pt" in report
    assert "graph_metadata.json" in report
    assert "a" * 64 in report


def test_report_documents_temporal_controls() -> None:
    report = build_report(_metadata())

    assert "event_timestamp < T" in report
    assert "same-timestamp" in report
    assert "Future events are rejected" in report


def test_report_documents_label_exclusion() -> None:
    report = build_report(_metadata())

    assert "`is_fraud`" in report
    assert "`fraud_scenario`" in report
    assert "Not used for topology" in report


def test_report_documents_limitations() -> None:
    report = build_report(_metadata())

    assert "synthetic dataset" in report
    assert "structural connectivity" in report


def test_freeze_contains_frozen_components() -> None:
    freeze = build_freeze(_metadata())

    assert "PHASE 6 FREEZE" in freeze
    assert "Status: FROZEN" in freeze
    assert "Graph validation" in freeze
    assert "Temporal graph controls" in freeze
    assert "Graph artifact materialization" in freeze


def test_freeze_contains_topology_hash() -> None:
    freeze = build_freeze(_metadata())

    assert "a" * 64 in freeze


def test_freeze_points_to_graph_ml() -> None:
    freeze = build_freeze(_metadata())

    assert "Next phase:" in freeze
    assert "Graph ML" in freeze


def test_metadata_fixture_is_json_serializable() -> None:
    metadata = _metadata()

    serialized = json.dumps(metadata)

    assert isinstance(serialized, str)


def test_report_is_deterministic_except_timestamp() -> None:
    first = build_report(_metadata())
    second = build_report(_metadata())

    # Remove generated timestamp before comparison.
    first_body = first.split(
        "Generated at: `",
        maxsplit=1,
    )[1].split(
        "`",
        maxsplit=1,
    )[1]

    second_body = second.split(
        "Generated at: `",
        maxsplit=1,
    )[1].split(
        "`",
        maxsplit=1,
    )[1]

    assert first_body == second_body