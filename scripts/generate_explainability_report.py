from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fraud_intelligence.explainability.comparison import (
    build_explainability_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports" / "explainability"

REPORT_PATH = (
    REPORT_DIR / "EXPLAINABILITY_REPORT.md"
)

FREEZE_PATH = (
    REPORT_DIR / "PHASE_8_FREEZE.txt"
)


def _format_methods() -> str:
    comparisons = build_explainability_comparison()

    lines: list[str] = []

    for item in comparisons:
        lines.append(f"### {item.method}")
        lines.append("")
        lines.append(
            f"- **Model family:** {item.model_family}"
        )
        lines.append(
            f"- **Evidence type:** {item.evidence_type}"
        )
        lines.append(
            f"- **Scope:** {item.scope}"
        )
        lines.append(
            f"- **Feature-based:** {item.feature_based}"
        )
        lines.append(
            f"- **Graph-based:** {item.graph_based}"
        )
        lines.append(
            f"- **Directional:** {item.directional}"
        )
        lines.append(
            f"- **Transaction-level:** "
            f"{item.transaction_level}"
        )
        lines.append("")
        lines.append("**Strengths:**")

        for strength in item.strengths:
            lines.append(f"- {strength}")

        lines.append("")
        lines.append("**Limitations:**")

        for limitation in item.limitations:
            lines.append(f"- {limitation}")

        lines.append("")

    return "\n".join(lines)


def build_report() -> str:
    generated_at = datetime.now(
        timezone.utc
    ).isoformat()

    return f"""# Explainability Report

## Phase 8 — Explainability

**Project:** Fraud Intelligence  
**Phase:** 8 — Explainability  
**Status:** Complete  
**Generated:** {generated_at}

---

## 1. Objective

Phase 8 provides structured, validated, and human-readable
explanations for transaction fraud predictions.

The explainability layer answers:

- Why was a transaction flagged?
- Which transaction features contributed to the model output?
- What graph context is connected to the transaction?
- Which neighboring nodes or features are influential?
- What evidence is available to a fraud analyst?

The explainability layer does not establish that fraud actually
occurred. It describes model evidence and graph context.

---

## 2. Explainability Architecture

```text
Classical ML Models
        |
        +--> Native Feature Attribution
        |
        +--> SHAP
        |
        v
Transaction Explanation
        ^
        |
Graph ML
        |
        +--> Historical Neighborhood
        |
        +--> GNN Feature Importance
        |
        +--> GNN Node Importance
        |
        v
Human-Readable Explanation
        |
        v
Leakage / Contract Validation

---

## 3. Phase Components

### 8.1 Explainability Data Contract

Defines the base explainability contract, including:

-transaction fraud explanation task
-transaction target node
-target label separation
-forbidden explanation columns
-required transaction identifiers
-explanation component definitions

### 8.2 Classical ML Feature Attribution

Provides model-specific feature attribution for:

-Logistic Regression
-Random Forest
-XGBoost

Attributions are ranked deterministically by absolute
contribution.

### 8.3 SHAP Integration

Provides local SHAP explanations for supported tree-based models.

The implementation uses model feature columns as the authoritative
feature schema and rejects forbidden fields.

### 8.4 GNN / Graph Explanation Contract

Defines the graph explanation contract, including:

-target transaction
-neighbor nodes
-supporting edges
-node importance
-edge importance
-allowed graph node types
-allowed graph relations
-temporal restrictions

### 8.5 GNN Neighborhood Explanation

Extracts deterministic graph neighborhoods around a transaction.

Historical transaction context must satisfy:

neighbor_timestamp < target_timestamp

Same-timestamp and future transaction context are excluded.

### 8.6 GNN Feature / Node Importance

Provides perturbation-based importance for:

-transaction features
-neighboring graph nodes

The target transaction itself is not ablated.

### 8.7 Transaction Explanation Assembly

Combines prediction evidence with available:

-classical attribution
-SHAP
-graph neighborhood
-GNN importance

No model prediction or attribution is recalculated.

### 8.8 Human-Readable Explanation

Converts structured explanation evidence into:

-summary
-risk status
-feature reasons
-graph findings
-supporting evidence
-limitations

The human-readable layer does not introduce a new prediction threshold.

### 8.9 Explanation Validation & Leakage Controls

Validates:

-forbidden features
-target labels
-identifiers used as model features
-timestamp restrictions
-future graph context
-same-timestamp graph context
-target transaction consistency
-human-readable leakage

### 8.10 Explainability Comparison

Provides a descriptive comparison of:

-Classical Attribution
-SHAP
-GNN Neighborhood
-GNN Importance

The comparison documents differences in evidence type, scope,
directionality, strengths, and limitations without ranking methods.

### 8.11 Explainability Tests & Integration

Validates that the explainability components operate together:

prediction
    +
classical / SHAP evidence
    +
graph evidence
    |
    v
TransactionExplanation
    |
    v
HumanReadableExplanation
    |
    v
Leakage Validation

---

## 4. Explainability Methods

{_format_methods()}

---

## 5. Temporal and Leakage Controls

Explainability follows the project's strict temporal design.

Historical graph context must be strictly earlier than the target
transaction timestamp.

The following are prohibited as explanatory model features:

-is_fraud
-fraud_scenario
-transaction_id
-customer_id
-account_id
-card_id
-merchant_id
-device_id
-ip_id
-node_id
-timestamp

Fraud scenario information remains diagnostic/post-hoc information
and is not used as predictive explanation evidence.

---

## 6. Human-Readable Interpretation

Human-readable explanations describe model evidence rather than
claiming ground truth.

For example:

A transaction may be flagged because specific transaction
features increased the model score and because historical graph
connections provided additional model evidence.

This distinction is important:

Model evidence
      !=
Proof of fraud

---

## 7. Validation Principles

The explainability layer follows these principles:

-Existing model outputs are reused rather than recalculated.
-Existing attribution results are reused rather than recalculated.
-Explanation features must come from the approved feature schema.
-Target labels cannot enter explanation features.
-Future information cannot enter historical graph context.
-Same-timestamp transaction context is excluded.
-Transaction and graph targets must remain consistent.
-Human-readable output must not expose forbidden model fields.
-Explanation methods are descriptive and are not ranked.
-Explainability does not replace fraud investigation.

---

## 8. Test Coverage

Phase 8 contains dedicated tests for:

-explainability contracts
-classical attribution
-SHAP
-GNN explanation contracts
-graph neighborhoods
-GNN importance
-transaction assembly
-human-readable explanations
-leakage validation
-explainability comparison
-end-to-end explainability integration

Phase 8.11 provides the integration-level validation across the
explainability pipeline.

---

## 9. Known Dependency Warnings

The test environment currently produces external dependency warnings
from PyTorch and SHAP.

These include:

-PyTorch torch.jit.script deprecation warning
-SHAP colormap deprecation warnings

These warnings do not represent failures in the Phase 8
implementation.

---

## 10. Phase 8 Deliverables

src/fraud_intelligence/explainability/
├── __init__.py
├── contracts.py
├── classical.py
├── shap.py
├── graph_neighborhood.py
├── gnn_importance.py
├── transaction.py
├── human_readable.py
├── validation.py
└── comparison.py

Tests:

tests/unit/
├── test_explainability_contracts.py
├── test_explainability_classical.py
├── test_explainability_shap.py
├── test_explainability_gnn_contract.py
├── test_explainability_graph_neighborhood.py
├── test_explainability_gnn_importance.py
├── test_explainability_transaction.py
├── test_explainability_human_readable.py
├── test_explainability_validation.py
└── test_explainability_comparison.py

tests/integration/
└── test_explainability_integration.py

---

## 11. Phase 8 Completion

Phase 8 establishes a complete explainability layer covering
classical ML, SHAP, graph structure, GNN importance, transaction-level
assembly, human-readable interpretation, validation, and integration.

The layer is designed to preserve the project's temporal leakage
controls and separation between predictive evidence and post-hoc
diagnostic information.

Phase 8 status: COMPLETE.
"""

def build_freeze_marker() -> str:
    return """PHASE 8 FREEZE

Project: Fraud Intelligence
Phase: 8 — Explainability
Status: COMPLETE / FROZEN

Completed:
8.1 Explainability Data Contract
8.2 Classical ML Feature Attribution
8.3 SHAP Integration
8.4 GNN / Graph Explanation Contract
8.5 GNN Neighborhood Explanation
8.6 GNN Feature / Node Importance
8.7 Transaction Explanation Assembly
8.8 Human-Readable Explanation
8.9 Explanation Validation & Leakage Controls
8.10 Explainability Comparison
8.11 Explainability Tests & Integration
8.12 Explainability Report & Freeze

Freeze rules:

Do not modify Phase 8 explainability contracts without
explicitly reopening Phase 8.
Do not modify frozen Phase 5, Phase 6, or Phase 7 artifacts
as part of Phase 8.
Do not introduce target leakage into explanation features.
Do not use future or same-timestamp graph context.
Do not treat explanation evidence as proof of fraud.
"""

def main() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        build_report(),
        encoding="utf-8",
    )

    FREEZE_PATH.write_text(
        build_freeze_marker(),
        encoding="utf-8",
    )

    print(f"Generated: {REPORT_PATH}")
    print(f"Generated: {FREEZE_PATH}")

if __name__=="__main__":
    main()