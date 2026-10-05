# MLOps Architecture

## 1. Purpose

This document defines the MLOps architecture for the Fraud Intelligence platform.

The goal is to establish a production-oriented lifecycle for:

- data versioning
- feature generation
- model training
- model evaluation
- model quality gates
- model registration
- model deployment
- API serving
- frontend consumption
- monitoring and observability
- reproducibility

This document defines the target architecture. Components that are planned for later Phase 12 work are explicitly marked as future implementation.

---

## 2. Current Project State

The project currently has the following completed and frozen components:

- Versioned synthetic fraud dataset
- Feature engineering pipeline
- Classical ML models
- Graph construction
- Graph ML
- Explainability
- Fraud intelligence
- FastAPI backend
- React + TypeScript frontend
- API integration tests
- Frontend build and lint validation

Phase 11 is complete and frozen.

Phase 12 introduces the MLOps and productionization layer.

### Important boundary

The architecture described below contains both:

1. Components that already exist in the project.
2. Components planned for implementation in later Phase 12 stages.

A planned component must not be represented as implemented until its corresponding phase is complete.

---

## 3. Target MLOps Lifecycle

```text
Frozen Dataset
      |
      v
Feature Engineering
      |
      v
Feature Dataset
      |
      v
Training Pipeline
(LR / RF / XGBoost)
      |
      v
Model Evaluation
(PR-AUC / Recall / Precision / F1)
      |
      v
Quality Gate
      |
      v
Model Registry
Candidate
   |
Validated
   |
Approved
   |
Production
   |
Archived
      |
      v
FastAPI Prediction Service
      |
      v
React Frontend

Observability operates alongside the serving and model lifecycle:
                    +------------------------+
                    |   MLOps Observability |
                    +------------------------+
                    | Data Drift             |
                    | Prediction Drift      |
                    | Model Performance     |
                    | API Health             |
                    +------------------------+

---

## 4. Architecture Overview

                    +----------------------+
                    |    Frozen Dataset    |
                    |       v1.0.0         |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Feature Engineering  |
                    |   Feature Dataset    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Training Pipeline  |
                    | LR / RF / XGBoost    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |   Model Evaluation   |
                    | PR-AUC / Recall /    |
                    | Precision / F1       |
                    +----------+-----------+
                               |
                         Quality Gate
                               |
                    +----------v-----------+
                    |    Model Registry    |
                    | Candidate            |
                    | Validated            |
                    | Approved             |
                    | Production           |
                    | Archived             |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |       FastAPI        |
                    |  Prediction Service  |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |    React Frontend    |
                    | Fraud Investigation  |
                    +----------------------+

                               ^
                               |
                    +----------------------+
                    | MLOps Observability  |
                    | Data Drift           |
                    | Prediction Drift     |
                    | Model Performance    |
                    | API Health           |
                    +----------------------+

---

## 5. Data Versioning

The dataset is treated as a versioned input to the ML lifecycle.

The currently frozen dataset is:

Dataset: fraud-intelligence-synthetic
Version: 1.0.0

The dataset contains:
- 100,000 transactions
- 1% synthetic fraud
- deterministic generation using seed 42
- no missing values
- unique transaction identifiers
- UTC timestamps
- predefined fraud scenarios

The dataset version is an important reproducibility boundary.

A future training run should be traceable to:

Dataset Version
        +
Feature Dataset Version
        +
Code Version
        +
Training Configuration
        +
Model Artifact

---

## 6. Feature Dataset

Feature engineering transforms the frozen transactional dataset into the model-ready feature dataset.

The current approved feature dataset contains:
53 approved features

The feature dataset is generated using the existing feature engineering pipeline.

The MLOps lifecycle should treat the feature dataset as a reproducible artifact rather than an undocumented intermediate file.

Future training runs should record:
- source dataset version
- feature generation code version
- feature columns
- feature configuration
- dataset row count
- feature dataset location
- generation timestamp

---

## 7. Training Pipeline

The classical ML training stage currently contains:
- Logistic Regression
- Random Forest
- XGBoost

The training pipeline uses chronological data splitting to reduce temporal leakage.

The established split is:
Training      70%
Validation    15%
Test          15%

The test set remains isolated from model selection and threshold optimization.

Future MLOps implementation should make the training process reproducible through explicit configuration.

A training run should capture:
- dataset version
- feature dataset version
- model type
- hyperparameters
- random seed
- training configuration
- feature count
- train/validation/test boundaries
- code version
- resulting metrics
- model artifact

---

## 8. Model Evaluation

Model evaluation is based on fraud detection metrics rather than accuracy alone.

Primary metrics include:
- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC
- True Positives
- False Positives
- True Negatives
- False Negatives

Because fraud is highly imbalanced, PR-AUC and fraud-class performance are particularly important.

The current project already contains persisted classical model evaluation results under:
reports/model_evaluation/

These evaluation artifacts form the baseline for future MLOps quality gates.

---

## 9. Quality Gate

A quality gate determines whether a trained model is eligible to progress through the model lifecycle.

Conceptually:
Training
   |
   v
Evaluation
   |
   v
+----------------------+
| Quality Gate          |
|                      |
| Required metrics     |
| Regression checks    |
| Artifact validation  |
| Data validation      |
+----------+-----------+
           |
      +----+----+
      |         |
    PASS       FAIL
      |         |
      v         v
 Registry    Reject

The quality gate should eventually evaluate:
- required evaluation metrics
- metric regression against approved baselines
- artifact existence
- feature compatibility
- model metadata completeness
- training reproducibility
- data quality checks

The exact automated thresholds will be defined in the later model quality phase.

---

## 10. Model Lifecycle

The target model lifecycle is:
Candidate
    |
    v
Validated
    |
    v
Approved
    |
    v
Production
    |
    v
Archived

### Candidate
A newly trained model that has completed training but has not yet passed all quality checks.

### Validated
A model that has passed automated validation and evaluation requirements.

### Approved
A validated model that has been explicitly selected for deployment.

### Production
The currently deployed model version used by the prediction service.

### Archived
A previous model version retained for reproducibility, auditability, and rollback.

The model registry is a target component of the later Phase 12 implementation.

---

## 11. Deployment Architecture

The target serving architecture is:
                 +------------------+
                 |  Model Registry  |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 |   FastAPI API     |
                 |                  |
                 | Prediction       |
                 | Investigation    |
                 | Explainability   |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 | React Frontend   |
                 +------------------+

The API is responsible for serving model-backed functionality.

The frontend consumes the API rather than directly loading model artifacts.

This separation provides:
- independent frontend/backend deployment
- centralized model serving
- API-level validation
- easier model replacement
- independent scaling
- clearer observability boundaries

---

## 12. Reproducibility

A production training run should be reproducible from a defined combination of:
Dataset Version
+
Feature Dataset
+
Source Code Commit
+
Python Environment
+
Training Configuration
+
Random Seed
+
Model Configuration

The resulting model should have associated metadata describing its origin.

At minimum, model metadata should eventually include:
model_name
model_version
model_type
dataset_version
feature_version
code_version
training_timestamp
training_parameters
evaluation_metrics
artifact_location

---

## 13. Observability

The target MLOps platform monitors four major areas.

### 13.1 Data Drift

Monitor whether production input distributions differ significantly from the training/reference dataset.

Examples:
- transaction amount distribution
- transaction frequency
- geographic features
- device-related features
- merchant-related features

### 13.2 Prediction Drift

Monitor changes in model output distributions.

Examples:
- fraud probability distribution
- predicted fraud rate
- score distribution
- prediction volume

### 13.3 Model Performance

When ground-truth fraud labels become available, monitor:
- Precision
- Recall
- F1
- PR-AUC
- False Positive Rate
- False Negative Rate

Performance degradation can trigger model investigation or retraining.

### 13.4 API Health

Monitor the serving layer for:
- request volume
- latency
- error rate
- HTTP status codes
- service availability
- model loading failures

---

## 14. Monitoring Feedback Loop

The target lifecycle includes a feedback loop:
Production
    |
    v
Predictions
    |
    v
Monitoring
    |
    +----> Data Drift
    |
    +----> Prediction Drift
    |
    +----> Model Performance
    |
    +----> API Health
    |
    v
Investigation
    |
    v
Retraining Decision
    |
    v
Training Pipeline

This enables the system to evolve from a static ML application into a continuously maintainable fraud intelligence platform.

---

## 15. Phase Boundaries

Phase 12 is intentionally divided into separate implementation stages.

Phase	Responsibility	Status
12.1	MLOps architecture	In progress
12.2	MLflow experiment tracking	Planned
12.3	Model registry	Planned
12.4	Reproducible training pipeline	Planned
12.5	Dockerization	Planned
12.6	Frontend production build/environment	Planned
12.7	CI/CD	Planned
12.8	Model/data quality gates	Planned
12.9	API observability	Planned
12.10	ML monitoring	Planned
12.11	Production architecture documentation	Planned
12.12	Final production validation	Planned


No later-phase implementation is included in Phase 12.1.

---

## 16. Current vs Target State

### Currently implemented
- Frozen dataset
- Feature engineering
- Feature dataset
- Classical ML models
- Model evaluation artifacts
- FastAPI prediction and investigation APIs
- React frontend
- API integration tests
- Frontend build validation

### Target / future implementation
- MLflow experiment tracking
- Model registry
- Automated training pipeline
- Model quality gates
- Dockerized services
- CI/CD
- API observability
- ML monitoring
- Automated retraining workflow

The distinction is intentional to prevent documentation from claiming production capabilities that have not yet been implemented.

---

## 17. Design Principles

The MLOps architecture follows these principles:
1. Reproducibility
   Every model should be traceable to its data, code, configuration, and environment.
2. Temporal integrity
   Future information must not leak into model training or evaluation.
3. Artifact traceability
   Models and evaluation results should have identifiable versions.
4. Quality before deployment
   Models must pass validation before reaching production.
5. Separation of concerns
   Training, model management, serving, frontend, and monitoring remain distinct components.
6. Observability by design
   Data, model, and API behavior should be measurable.
7. Rollback capability
   Previous production models should remain identifiable and recoverable.
8. Explicit phase boundaries
   Frozen components should not be modified casually when implementing later MLOps stages.

---

## 18. Expected End State

The completed MLOps implementation should provide a lifecycle similar to:
Versioned Data
      |
      v
Reproducible Features
      |
      v
Tracked Training Run
      |
      v
Automated Evaluation
      |
      v
Quality Gate
      |
      v
Model Registry
      |
      v
Approved Production Model
      |
      v
FastAPI Serving
      |
      v
React Application
      |
      v
Production Monitoring
      |
      +----------------------+
      |                      |
      v                      |
Drift / Performance --------+
      |
      v
Retraining Decision

This architecture provides the foundation for the remaining Phase 12 implementation.