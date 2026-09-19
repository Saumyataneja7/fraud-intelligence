from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports" / "graph_ml"
REPORT_PATH = REPORT_DIR / "GRAPH_ML_REPORT.md"
FREEZE_PATH = REPORT_DIR / "PHASE_7_FREEZE.txt"


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------


def build_report() -> str:
    generated_at = datetime.now(timezone.utc).isoformat()

    return f"""# Graph ML Report

**Project:** Fraud Intelligence  
**Phase:** 7 — Graph Machine Learning  
**Report:** Phase 7.12 — Graph ML Report & Freeze  
**Generated:** {generated_at}

---

## 1. Executive Summary

Phase 7 implements the Graph Machine Learning layer for the Fraud Intelligence
project.

The objective is transaction-level fraud classification using graph-derived
context while preserving the project's temporal leakage controls.

The implementation covers:

- Graph ML data contracts
- Temporal graph train/validation/test splitting
- Transaction node feature preparation
- Transaction node label preparation
- GraphSAGE baseline
- GNN classifier training
- Validation-based early stopping
- Graph ML evaluation
- Threshold optimization
- Precision@K and Recall@K
- GNN error analysis
- Classical ML versus GNN comparison
- Graph ML documentation and freeze

The graph topology itself was created and frozen during Phase 6 and is not
modified by Phase 7.

---

## 2. Phase 7 Scope

The Phase 7 task is:

> Transaction fraud classification using graph context.

The target node is:

```text
transaction

The target variable is:

is_fraud

The Graph ML pipeline is designed around the following flow:

Phase 4 Feature Dataset
        |
        v
Graph ML Feature Preparation
        |
        +--------------------+
        |                    |
        v                    v
Transaction Labels    Temporal Graph Split
        |                    |
        +----------+---------+
                   |
                   v
          GraphSAGE Classifier
                   |
                   v
              GNN Logits
                   |
                   v
              Probabilities
                   |
        +----------+----------+
        |                     |
        v                     v
   Classification       Ranking Analysis
        |                     |
        v                     v
 Thresholding          Precision@K
 Error Analysis        Recall@K
        |                     |
        +----------+----------+
                   |
                   v
          Model Comparison

---

## 3. Data Contract

The Graph ML contract defines:

-task: transaction fraud classification
-target node type: transaction
-target column: is_fraud

The following are forbidden as model features:

is_fraud
fraud_scenario
node_id
transaction_id
customer_id
account_id
card_id
merchant_id
device_id
ip_id
timestamp

This prevents identifiers, target information, fraud scenario labels, and raw
prediction timestamps from being passed directly into the model.

The Graph ML feature preparation reuses the approved engineered feature
columns established during Phase 5.

---

## 4. Temporal Controls

Temporal integrity is a core requirement of this project.

The Graph ML contract requires:

Only information strictly earlier than the prediction timestamp may be used
for historical graph context.

Events occurring at exactly the prediction timestamp are excluded because the
dataset does not provide an explicit event ordering within identical
timestamps.

The temporal split uses chronological ordering:

70% Train
15% Validation
15% Test

Transactions are sorted deterministically using:

timestamp
transaction_id

with stable sorting.

The test set is not used during training or validation-based early stopping.

---

## 5. Graph Context

Phase 6 produced the frozen heterogeneous graph containing:

### Node types

customer
account
card
transaction
merchant
device
ip

### Edge types
customer_account
account_card
customer_transaction
account_transaction
card_transaction
customer_device
customer_ip
customer_merchant
transaction_merchant
transaction_device
transaction_ip

Phase 6 graph artifact:

data/graph/heterogeneous_graph.pt

Phase 6 topology hash:

1f26bbc304ac823e4f5e697442b86aff8061804f119cab265d0d4100e0841867

The Phase 7 pipeline consumes this topology and does not modify the frozen
graph construction.

---

## 6. Graph Statistics

The frozen Phase 6 graph contains:

Node Type	Count
Customer	20,000
Account	25,000
Card	30,000
Transaction	100,000
Merchant	1,000
Device	25,000
IP	30,000
Total	231,000

The graph contains:

843,214 total edges

These structural statistics are inherited from the frozen Phase 6 graph.

---

## 7. Transaction Node Features

Transaction node features are prepared from the approved Phase 5 engineered
feature set.

The approved model feature count is:

53 features

The feature preparation pipeline:

-validates the feature dataset
-validates transaction graph nodes
-checks exact transaction-ID coverage
-performs one-to-one alignment
-validates missing values
-validates finite values
-converts features to torch.float32
-aligns features to graph transaction node IDs

The final feature tensor therefore preserves the graph node identity while
keeping identifiers outside the model feature matrix.

---

## 8. Transaction Labels

Transaction fraud labels are prepared separately from model features.

Target:

is_fraud

Labels are represented as:

torch.long

and are explicitly validated as binary.

The fraud_scenario column is retained separately as diagnostic metadata.

It is not used as a model input.

This distinction is important because fraud scenario information would otherwise
constitute target leakage.

---

## 9. GraphSAGE Baseline

Phase 7.5 introduces a transaction-focused GraphSAGE baseline.

Architecture:

Transaction Features
        |
        v
SAGEConv
        |
        v
Hidden Representation
        |
        v
SAGEConv
        |
        v
Transaction Embedding

Configuration defaults:

hidden_channels = 64
output_channels = 32
dropout = 0.20

The GraphSAGE encoder produces transaction node embeddings.

The classifier introduced in Phase 7.6 adds:

GraphSAGE Embedding
        |
        v
Linear Classification Head
        |
        v
Fraud Logit

A sigmoid transformation is applied during prediction to obtain fraud
probabilities.

---

## 10. Training Pipeline

The GNN training pipeline uses:

learning_rate = 0.001
weight_decay = 0.0001
max_epochs = 100
patience = 10
random_seed = 42

Class imbalance is handled using a positive-class weight calculated from the
training labels only.

The loss function is:

BCEWithLogitsLoss

with:

pos_weight

derived exclusively from the training partition.

This prevents validation and test class distributions from influencing the
training loss configuration.

---

## 11. Early Stopping

Training monitors validation loss.

The best validation state is restored after training.

The test partition is intentionally excluded from the training function.

This provides the following separation:

TRAIN
  |
  +--> model fitting
  +--> positive-class weighting
  |
  v
VALIDATION
  |
  +--> early stopping
  +--> model selection
  +--> threshold optimization
  |
  v
TEST
  |
  +--> final evaluation only

---

## 12. GNN Evaluation

The evaluation layer supports:

-Precision
-Recall
-F1
-PR-AUC
-ROC-AUC
-True Negatives
-False Positives
-False Negatives
-True Positives
-Support
-Predicted positives
-Actual positives

The standard evaluation threshold is:

0.50

Prediction is performed from model logits:

logit
  |
  v
sigmoid
  |
  v
fraud probability
  |
  v
threshold
  |
  v
binary prediction

---

## 13. Threshold Optimization

Phase 7.8 provides validation-only threshold optimization.

Default search grid:

0.05 through 0.99
step = 0.01

The optimization objective is:

F1

Deterministic tie-breaking is:

-higher F1
-higher recall
-higher precision
-lower threshold

The test set is not used to select the threshold.

---

## 14. Ranking Evaluation

Fraud detection is also treated as a ranking problem.

Phase 7.9 implements:

Precision@K
Recall@K

The ranking uses descending fraud probability with stable sorting.

This supports operational questions such as:

Among the top K transactions reviewed by an analyst, how much fraud is
captured?

Ranking metrics are kept separate from threshold-based classification metrics.

---

## 15. Error Analysis

Phase 7.10 provides diagnostic error analysis for:

False Positives
False Negatives

It supports:

-error-frame construction
-aggregate FP/FN counts
-summaries by dataset columns
-amount-band analysis
-false-positive extraction
-false-negative extraction

Fraud scenario information may be used after prediction for diagnostics.

It is never included as a model feature.

---

## 16. Classical ML vs GNN Comparison

Phase 7.11 provides a standardized comparison framework between:

Classical ML
    |
    +-- Logistic Regression
    +-- Random Forest
    +-- XGBoost

Graph ML
    |
    +-- GraphSAGE

The comparison schema contains:

model
model_family
split
precision
recall
f1
pr_auc
roc_auc
predicted_positives
actual_positives

Ranking comparison contains:

model
model_family
split
k
precision_at_k
recall_at_k

The comparison layer is descriptive.

It does not assign a winner or introduce model-ranking logic.

---

## 17. Leakage Controls

The following controls are enforced across the Graph ML pipeline:

Target leakage
is_fraud
fraud_scenario

are excluded from model features.

### Identifier leakage

Customer, account, card, merchant, device, IP, transaction IDs and graph
node IDs are excluded.

### Timestamp leakage

The prediction timestamp is not directly supplied as a feature.

### Temporal leakage

Only information strictly earlier than the prediction event is allowed for
historical graph context.

### Same-timestamp leakage

Events with identical timestamps are excluded from historical context because
there is no explicit event ordering.

### Test leakage

The test set is not used during GNN training or validation-based model
selection.

### Threshold leakage

Threshold optimization is performed on validation data rather than test data.

---

## 18. Reproducibility

The project uses deterministic controls including:

random seed = 42

Chronological splitting uses stable sorting.

Graph node IDs are deterministic.

Graph topology was frozen in Phase 6.

The Phase 6 topology SHA-256 is:

1f26bbc304ac823e4f5e697442b86aff8061804f119cab265d0d4100e0841867

These controls allow future experiments to distinguish implementation changes
from changes caused by nondeterministic processing.

---

## 19. Testing

Phase 7 was developed with unit tests for each major component.

The test coverage includes:

7.1  Graph ML contracts
7.2  Temporal graph split
7.3  Graph features
7.4  Graph labels
7.5  GraphSAGE
7.6  GNN training
7.7  GNN evaluation
7.8  Threshold optimization
7.9  Ranking metrics
7.10 Error analysis
7.11 Model comparison

The final test count should be recorded from the repository test run immediately
before freezing this phase.

---

## 20. Numerical Results Policy

The Graph ML implementation contains evaluation and comparison infrastructure,
but this report does not fabricate numerical GNN performance results.

Only metrics produced by an actual end-to-end training/evaluation run should be
added as empirical results.

The following should therefore not be interpreted as measured GNN performance:

-architecture defaults
-expected behavior
-unit-test fixture values
-example thresholds
-example ranking metrics

This keeps the project reproducible and prevents test fixtures from being
mistaken for model results.

---

## 21. Known Scope / Limitations

### 21.1 GraphSAGE baseline

The current GraphSAGE implementation is a transaction-focused baseline.

The frozen Phase 6 graph is heterogeneous, while the baseline GraphSAGE encoder
operates on the transaction-focused neighborhood representation.

A future heterogeneous GNN can build on the frozen graph without changing the
Phase 6 data contracts.

### 21.2 Synthetic dataset

The underlying dataset is synthetic and generated specifically for this
portfolio project.

It must not be represented as real-world fraud data.

### 21.3 End-to-end benchmark

A production-quality benchmark should record:

-training duration
-inference latency
-memory usage
-model artifact size
-validation metrics
-test metrics
-threshold selected on validation
-Precision@K / Recall@K
-FP/FN distributions

These should be generated from actual runs rather than estimated.

---

## 22. Phase 7 Component Status

| Component | Status |
|---|---|
| Graph ML data contract | Complete |
| Temporal graph split | Complete |
| Graph feature preparation | Complete |
| Transaction node labels | Complete |
| GraphSAGE baseline | Complete |
| GNN training pipeline | Complete |
| GNN evaluation | Complete |
| Threshold optimization | Complete |
| Precision@K / Recall@K | Complete |
| GNN error analysis | Complete |
| Classical ML vs GNN comparison | Complete |
| Graph ML report | Complete |
| Phase 7 freeze | Complete |

---

## 23. Transition to Next Project Layer

With Phase 7 frozen, the project has established:

Synthetic Fraud Data
        |
        v
Data Validation
        |
        v
Feature Engineering
        |
        v
Classical ML
        |
        v
Heterogeneous Graph
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
        v
API
        |
        v
Frontend
        |
        v
MLOps

The next layer should build on these frozen contracts rather than changing the
existing data, feature, classical ML, or graph topology definitions.

---

## 24. Freeze

Phase 7 is considered frozen when:

-all Phase 7 tests pass
-the Graph ML report is generated
-the freeze marker is generated
-no Phase 5 or Phase 6 contracts are modified
-the frozen Phase 6 topology remains unchanged

Phase 7 — Graph Machine Learning: FROZEN
"""

def build_freeze_marker() -> str:
    return """PHASE 7 FREEZE

Project: Fraud Intelligence
Phase: 7 — Graph Machine Learning
Version: 1.0.0
Status: FROZEN

Frozen components:
Graph ML data contract
Temporal graph split
Graph feature preparation
Transaction node labels
GraphSAGE baseline
GNN training pipeline
GNN evaluation
Threshold optimization
Precision@K / Recall@K
GNN error analysis
Classical ML vs GNN comparison
Graph ML report

Graph artifact inherited from Phase 6:
data/graph/heterogeneous_graph.pt

Graph metadata:
data/graph/graph_metadata.json

Phase 6 topology SHA-256:
1f26bbc304ac823e4f5e697442b86aff8061804f119cab265d0d4100e0841867

Report:
reports/graph_ml/GRAPH_ML_REPORT.md

Freeze marker:
reports/graph_ml/PHASE_7_FREEZE.txt

Important:
No numerical GNN performance claims are frozen unless they are produced by an
actual end-to-end training and evaluation run.

Next project layer:
Explainability / Fraud Intelligence
"""

def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

if __name__ == "__main__":
    main()