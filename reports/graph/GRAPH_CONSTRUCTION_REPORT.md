# Graph Construction Report

## Phase

**Phase 6 — Graph Construction**

**Phase 6.10 — Graph Construction Report & Freeze**

Generated at: `2026-09-16T14:25:08.175256+00:00`

---

## 1. Executive Summary

The Fraud Intelligence project now contains a validated,
heterogeneous fraud graph constructed from the synthetic
transaction dataset.

The graph represents:

- customers
- accounts
- cards
- transactions
- merchants
- devices
- IP addresses

The graph topology is constructed from relational entity
relationships and transaction relationships.

Fraud labels are not used to create graph topology.

---

## 2. Artifact

| Property | Value |
|---|---|
| Artifact name | `fraud-intelligence-heterogeneous-graph` |
| Artifact version | `1.0.0` |
| Format | `torch-geometric-heterodata` |
| Graph file | `data/graph/heterogeneous_graph.pt` |
| Metadata file | `data/graph/graph_metadata.json` |
| Topology SHA-256 | `1f26bbc304ac823e4f5e697442b86aff8061804f119cab265d0d4100e0841867` |

---

## 3. Node Schema

The canonical node types are:

`customer`, `account`, `card`, `transaction`, `merchant`, `device`, `ip`

### Source ID mapping

| Node type | Source identifier |
|---|---|
| `customer` | `customer_id` |
| `account` | `account_id` |
| `card` | `card_id` |
| `transaction` | `transaction_id` |
| `merchant` | `merchant_id` |
| `device` | `device_id` |
| `ip` | `ip_id` |

### Node counts

| Node type | Count |
|---|---:|
| `account` | 25,000 |
| `card` | 30,000 |
| `customer` | 20,000 |
| `device` | 25,000 |
| `ip` | 30,000 |
| `merchant` | 1,000 |
| `transaction` | 100,000 |

**Total nodes:** 231,000

---

## 4. Edge Schema

The graph contains the following canonical relationships:

`customer_account`, `account_card`, `customer_transaction`, `account_transaction`, `card_transaction`, `customer_device`, `customer_ip`, `customer_merchant`, `transaction_merchant`, `transaction_device`, `transaction_ip`

### Edge counts

| Edge type | Count |
|---|---:|
| `account_card` | 27,328 |
| `account_transaction` | 100,000 |
| `card_transaction` | 100,000 |
| `customer_account` | 23,925 |
| `customer_device` | 27,973 |
| `customer_ip` | 33,923 |
| `customer_merchant` | 130,065 |
| `customer_transaction` | 100,000 |
| `transaction_device` | 100,000 |
| `transaction_ip` | 100,000 |
| `transaction_merchant` | 100,000 |

**Total edges:** 843,214

---

## 5. Graph Construction

Graph construction follows the established pipeline:

```text
Relational source data
        ↓
Node tables
        ↓
Deterministic graph ID mapping
        ↓
Typed edge tables
        ↓
Heterogeneous PyG graph
        ↓
Structural validation
        ↓
Temporal controls
        ↓
Graph statistics
        ↓
Materialized graph artifact
'''

Node IDs are deterministic contiguous integer identifiers.

Edge endpoints reference graph node IDs rather than raw
business identifiers.

---

## 6. Structural Validation

The graph validation layer verifies:

-canonical node types
-canonical edge types
-node ID presence
-node ID dtype
-contiguous node IDs
-edge index shape
-edge index dtype
-endpoint ranges
-duplicate edges
-graph metadata consistency
-node and edge counts

The graph must pass structural validation before it is
materialized.

---

## 7. Temporal Controls

Temporal controls use strict point-in-time semantics.

For a cutoff timestamp T, only events satisfying:

event_timestamp < T

are considered historically available.

Events occurring:

event_timestamp == T

are excluded because the dataset does not provide an
explicit ordering mechanism for events sharing the same
timestamp.

Future events are rejected by the temporal validation layer.

This prevents future transaction information from being
introduced into historical graph context.

---

## 8. Graph Statistics
Degree statistics

| Node type | Count | Min | Max | Mean | Median | P95 |
|---|---:|---:|---:|---:|---:|---:|
| `customer` | 20,000 | 5.00 | 29.00 | 15.79 | 16.00 | 21.00 |
| `account` | 25,000 | 0.00 | 19.00 | 6.05 | 6.00 | 11.00 |
| `card` | 30,000 | 0.00 | 18.00 | 4.24 | 4.00 | 9.00 |
| `transaction` | 100,000 | 6.00 | 6.00 | 6.00 | 6.00 | 6.00 |
| `merchant` | 1,000 | 157.00 | 314.00 | 230.06 | 229.00 | 269.00 |
| `device` | 25,000 | 0.00 | 51.00 | 5.12 | 4.00 | 16.00 |
| `ip` | 30,000 | 0.00 | 36.00 | 4.46 | 3.00 | 14.00 |

Isolated nodes

| Node type | Isolated nodes |
|---|---:|
| `account` | 1,075 |
| `card` | 2,672 |
| `customer` | 0 |
| `device` | 8,046 |
| `ip` | 9,402 |
| `merchant` | 0 |
| `transaction` | 0 |

Connected components
Statistic	Value
Component count	21,196
Largest component size	209,805
Largest component fraction	0.908247

Connected-component statistics are structural diagnostics.
They are not fraud-ring classifications.

---

## 9. Reproducibility

The graph artifact includes:

-explicit artifact version
-canonical node schema
-canonical edge schema
-node counts
-edge counts
-graph statistics
-deterministic topology hash

The graph can be loaded and structurally validated through
the graph materialization layer.

The topology hash provides an integrity check against
unexpected graph modification.

---

## 10. Leakage Controls

The graph construction design explicitly separates:

### Allowed

-entity relationships
-transaction relationships
-graph identifiers
-transaction timestamps
-historical relationships subject to point-in-time filtering

### Not used for topology

-'is_fraud'
-'fraud_scenario'
-future transactions
-same-timestamp transactions without event ordering

The `is_fraud` and `fraud_scenario` columns are retained only
as label metadata where required by the source transaction
representation. They are not used to create graph topology.

Fraud labels remain targets or post-hoc analysis attributes,
rather than graph-construction inputs.

---

## 11. Limitations

This graph is generated from a synthetic dataset designed for
Fraud Intelligence experimentation.

It is not a representation of real financial activity.

The current graph does not establish causal relationships
between entities.

Shared devices, IP addresses, merchants, or other entities
can have legitimate explanations.

Connected components therefore represent structural connectivity
rather than confirmed fraud rings.

Temporal filtering is available as an explicit control layer;
future Graph ML pipelines must use it whenever a
point-in-time training or inference graph is required.

---

## 12. Phase 6 Completion

Phases 6.1 through 6.9 establish:

-graph contracts
-node construction
-edge construction
-deterministic ID mapping
-heterogeneous graph construction
-structural validation
-temporal controls
-graph statistics
-graph artifact materialization

The graph-construction layer is therefore ready for the
next stage of the project.

---

## 13. Transition to Graph ML

The next stage can consume the validated graph artifact for
Graph ML experimentation.

Potential next-stage work includes:

-graph feature preparation
-node/edge feature design
-temporal graph dataset construction
-GraphSAGE
-graph-based fraud prediction
-graph embeddings
-graph-level investigation
-explainability

No Graph ML model is included in Phase 6.

Graph artifact version: 1.0.0

Status: Phase 6 graph construction complete.
