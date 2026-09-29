# Fraud Intelligence Report

## Phase 9 — Fraud Intelligence & Investigation

**Status:** COMPLETE / FROZEN

**Phase:** 9  
**Final verification:** 252 tests passed

---

## 1. Objective

Phase 9 transforms the outputs of the predictive and graph-ML layers
into structured fraud investigation intelligence.

The purpose is to answer:

1. Is this transaction suspicious according to the existing model?
2. What contextual signals surround the transaction?
3. Which entities are connected to the transaction?
4. Which candidate fraud networks contain the transaction?
5. What structural evidence supports investigation of those networks?
6. What does the surrounding network look like to an investigator?

Phase 9 does not replace the predictive models from Phases 5 and 7.

---

## 2. Architectural Position

The completed architecture is:

```text
Data
  |
  v
Validation
  |
  v
Feature Engineering
  |
  v
Classical ML
  |
  v
Graph Construction
  |
  v
Graph ML
  |
  v
Explainability
  |
  v
Fraud Intelligence
  |
  +--> Suspicious Transaction Intelligence
  |
  +--> Entity Relationship Intelligence
  |
  +--> Fraud Ring Candidate Detection
  |
  +--> Ring-Level Network Scoring
  |
  +--> Fraud Ring Evidence
  |
  +--> Entity Investigation View
  |
  +--> Transaction Investigation View
  |
  +--> Fraud Ring Investigation View
  |
  +--> Leakage / Consistency Validation
  |
  v
Integration Validation

---

## 3. Phase 9 Components

### 9.1 Fraud Intelligence Data Contract

Defines the common Phase 9 investigation contract.

Key controls:

-allowed entity types
-allowed relationship types
-intelligence components
-forbidden predictive fields
-temporal investigation rule

Forbidden predictive fields:

-is_fraud
-fraud_scenario

### 9.2 Suspicious Transaction Intelligence

Provides contextual signals around an existing model prediction.

Examples include:

-historical amount behaviour
-transaction velocity
-historical amount aggregates
-entity diversity
-novelty indicators

The component does not create a new fraud score.

The model probability and prediction label are preserved from the
existing predictive layer.

### 9.3 Entity Relationship Intelligence

Provides structured relationships between:

-Customer
-Account
-Card
-Transaction
-Merchant
-Device
-IP

Relationship topology follows the frozen Phase 6 graph contract.

Temporal investigation context is restricted to information strictly
before the target transaction timestamp when an event timestamp exists.

### 9.4 Fraud Ring Candidate Detection

Detects structurally connected candidate networks.

Candidate detection is:

-graph-based
-deterministic
-label-independent
-model-independent

A candidate is not automatically declared to be a fraud ring.

### 9.5 Ring-Level Network Scoring

Provides a structural network score based on:

-entity count
-relationship count
-relationship density
-entity-type diversity
-relationship-type diversity
-non-transaction connectivity

The resulting score is a structural investigation score.

It is NOT:

-a fraud probability
-a model prediction
-a calibrated risk probability

### 9.6 Fraud Ring Evidence Assembly

Converts structural properties into investigation evidence.

Evidence categories include:

-entity connectivity
-relationship richness
-entity-type diversity
-relationship-type diversity
-network density
-non-transaction connectivity

Evidence strength is descriptive and based on the existing structural
network score.

### 9.7 Customer / Entity Investigation View

Provides an investigation-oriented view for:

-Customer
-Account
-Card
-Transaction
-Merchant
-Device
-IP

The view exposes:

-related entities
-related transactions
-candidate ring IDs
-existing network scores
-investigation summary
-temporal rule

### 9.8 Transaction Investigation View

Combines:

-existing prediction
-suspicion signals
-related entities
-relationships
-candidate ring IDs
-network scores
-ring evidence

The transaction view does not recompute the model prediction.

### 9.9 Fraud Ring Investigation View

Aggregates a complete candidate network into:

-entities
-relationships
-transactions
-entity-type counts
-relationship-type counts
-structural network score
-ring evidence
-investigation summary

It intentionally remains a candidate-network investigation view.

### 9.10 Intelligence Validation & Leakage Controls

The validation layer checks:

-forbidden predictive fields
-future context
-same-timestamp context
-invalid prediction values
-relationship consistency
-transaction identity consistency
-prediction consistency
-ring/evidence consistency
-cross-component consistency

Temporal rule:

historical event timestamp < target transaction timestamp

Same-timestamp event context is rejected because the project does not
define an event-ordering mechanism within the same timestamp.

### 9.11 Fraud Intelligence Integration Tests

The complete Phase 9 workflow was tested across the frozen components.

Final Phase 9 verification:

252 passed

This includes:

-Phase 9 unit tests
-Phase 9 integration tests

---

## 4. Leakage Controls

Phase 9 preserves the project's temporal integrity requirements.

### Forbidden predictive fields

The investigation layer must not consume:

is_fraud
fraud_scenario

as predictive investigation context.

### Future context

Information after the target transaction timestamp is rejected.

### Same-timestamp context

Information at exactly the target transaction timestamp is rejected
where event timestamps are available.

This avoids assuming an event ordering that does not exist in the
dataset.

### Static relationships

Untimestamped structural relationships may be retained because they
represent graph structure rather than timestamped events.

---

## 5. Separation of Responsibilities

Phase 8 and Phase 9 have intentionally different responsibilities.

### Phase 8

Answers:

Why did the model make this prediction?

Includes:

-feature attribution
-SHAP
-graph neighborhood explanation
-node / edge importance
-human-readable explanation

### Phase 9

Answers:

What does the surrounding fraud network tell an investigator?

Includes:

-suspicious transaction context
-entity relationships
-candidate networks
-structural network scoring
-ring evidence
-investigation views

Phase 9 does not override Phase 8.

---

## 6. Determinism

Phase 9 investigation components use deterministic ordering wherever
collections are exposed to investigators.

Examples:

-entity ordering
-relationship ordering
-candidate IDs
-network-score ordering
-evidence ordering
-batch investigation-view ordering

This supports reproducibility and stable downstream API/frontend
integration.

---

## 7. Phase 9 Test Summary
Component	Tests
9.1 Data Contract	45
9.2 Suspicious Transaction Intelligence	22
9.3 Entity Relationship Intelligence	19
9.4 Fraud Ring Candidate Detection	18
9.5 Ring-Level Network Scoring	19
9.6 Fraud Ring Evidence Assembly	22
9.7 Entity Investigation View	23
9.8 Transaction Investigation View	24
9.9 Fraud Ring Investigation View	25
9.10 Validation & Leakage Controls	27
9.11 Integration Tests	8
Total	252

---

## 8. Frozen Dependencies

Phase 9 depends on the following frozen project layers:

-Phase 4 feature engineering
-Phase 5 classical ML
-Phase 6 heterogeneous graph
-Phase 7 graph ML
-Phase 8 explainability

Phase 9 does not modify their contracts or topology.

---

## 9. Important Interpretation Constraints

The following distinctions are intentional.

### Prediction probability

Comes from the existing predictive model.

### Suspicion signals

Describe contextual characteristics surrounding a transaction.

### Structural network score

Describes the structure of a candidate network.

### Evidence

Describes observable structural properties.

### Candidate fraud ring

Represents a structurally connected investigation candidate.

None of these components should be interpreted as an independent
ground-truth fraud declaration.

### 9.12 Fraud Intelligence Report & Freeze

This report documents the completed Phase 9 fraud intelligence and
investigation layer.

Phase 9.12 does not introduce new intelligence logic. It records the
final contracts, investigation components, validation controls, test
coverage, and freeze status for Phase 9.

Final Phase 9 verification:

```text
252 passed

---

## 10. Phase 9 Freeze

Phase 9 is frozen after:

252 tests passed

The frozen scope includes:

src/fraud_intelligence/intelligence/
├── contracts.py
├── suspicious_transaction.py
├── entity_relationships.py
├── fraud_ring_candidates.py
├── ring_network_scoring.py
├── fraud_ring_evidence.py
├── entity_investigation.py
├── transaction_investigation.py
├── ring_investigation.py
└── validation.py

No additional Phase 9 intelligence logic should be added without
explicitly opening a new phase or change request.

---

## 11. Next Project Layer

The next project layer can consume the frozen Phase 9 investigation
contracts.

Potential downstream consumers include:

-FastAPI investigation endpoints
-React investigation dashboards
-fraud analyst workflows
-model monitoring
-MLflow/artifact tracking
-Dockerized deployment
-production-style observability

These are downstream integrations and are not part of the Phase 9
freeze itself.
