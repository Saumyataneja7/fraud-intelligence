# MLflow Model Registry Audit

## Purpose

Phase 12.13 audits the existing MLflow tracking and model registry state
without modifying registered models or historical runs.

## Registry validation

The following registered models were inspected using the local MLflow
tracking database:

| Registered model | Version | Status | `candidate` alias |
|---|---:|---|---:|
| `fraud-intelligence-logistic-regression` | 1 | READY | 1 |
| `fraud-intelligence-random-forest` | 1 | READY | 1 |
| `fraud-intelligence-xgboost` | 1 | READY | 1 |

Each model has a signature with 53 unique input features and an output
signature. The input feature names match the canonical
`model_feature_columns` list in
`data/processed/features/feature_dataset_metadata.json` exactly.

## Inference smoke tests

Each registered model was loaded and produced a prediction using synthetic
input. The XGBoost test used the data types declared by its MLflow signature.

These checks validate model loading, input schema compatibility, and basic
inference only. They do not establish model quality on real transactions.

## Historical failed run

A historical failed XGBoost run remains in the tracking database. Its cause
has not been established. The run was preserved; this audit did not modify
or delete it.

## Scope and limitations

This audit was read-only with respect to the MLflow tracking database and
model registry. No model versions or aliases were changed. Production
deployment, real-transaction inference, and ongoing monitoring are not
validated by these checks.
