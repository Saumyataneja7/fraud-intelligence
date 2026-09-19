# Explainability Report

## Phase 8 — Explainability

**Project:** Fraud Intelligence  
**Phase:** 8 — Explainability  
**Status:** Complete  
**Generated:** 2026-09-19T08:03:57.098657+00:00

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

### Classical Attribution

- **Model family:** classical_ml
- **Evidence type:** model feature attribution
- **Scope:** individual transaction
- **Feature-based:** True
- **Graph-based:** False
- **Directional:** True
- **Transaction-level:** True

**Strengths:**
- Simple to interpret.
- Directly tied to classical model parameters or feature importance.
- Efficient to compute.

**Limitations:**
- Does not explain graph structure.
- Feature importance semantics depend on the model.

### SHAP

- **Model family:** classical_ml
- **Evidence type:** local feature attribution
- **Scope:** individual transaction
- **Feature-based:** True
- **Graph-based:** False
- **Directional:** True
- **Transaction-level:** True

**Strengths:**
- Provides local feature contributions.
- Shows positive and negative feature effects.
- Provides a model-independent explanation interface for supported models.

**Limitations:**
- Computational cost can be higher than native feature importance.
- Does not explain graph topology.

### GNN Neighborhood

- **Model family:** graph_ml
- **Evidence type:** historical graph context
- **Scope:** transaction neighborhood
- **Feature-based:** False
- **Graph-based:** True
- **Directional:** False
- **Transaction-level:** True

**Strengths:**
- Shows connected entities and transactions.
- Provides structural context around a transaction.
- Supports fraud-ring investigation context.

**Limitations:**
- Shows structural context rather than causal evidence.
- Interpretation depends on graph construction.

### GNN Importance

- **Model family:** graph_ml
- **Evidence type:** feature and node perturbation importance
- **Scope:** individual transaction
- **Feature-based:** True
- **Graph-based:** True
- **Directional:** False
- **Transaction-level:** True

**Strengths:**
- Connects transaction features with graph context.
- Can identify influential neighboring nodes.
- Provides model-specific importance estimates.

**Limitations:**
- Perturbation importance is not causal evidence.
- Results depend on the perturbation strategy.
- Can be computationally expensive.


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
