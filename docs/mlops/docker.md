# Dockerized API

## Purpose

Phase 12.5 packages the Fraud Intelligence FastAPI application as a
self-contained Linux container for reproducible API runtime execution.

The container includes:

- FastAPI application
- API service and contract code
- Python runtime dependencies
- Frozen feature dataset
- Graph API SQLite index

The PyTorch/PyTorch Geometric dependencies are included because the current
explainability import chain loads the GNN explainability module at API startup.

## Build

From the repository root:

```bash
docker build -t fraud-intelligence-api:12.5 .

### Run

docker run -d \
  --name fraud-intelligence-api \
  -p 8000:8000 \
  fraud-intelligence-api:12.5

The API is available at:
http://127.0.0.1:8000

### Validation

Health check:
curl -sS http://127.0.0.1:8000/health

Expected:
{
  "status": "ok",
  "service": "fraud-intelligence-api",
  "version": "1.0.0"
}

Metadata:
curl -sS http://127.0.0.1:8000/metadata

Graph API:
curl -sS "http://127.0.0.1:8000/graph/0?node_type=customer"

Transaction investigation:
curl -sS -i http://127.0.0.1:8000/transaction/TXN_000000000

The current repository intentionally does not contain the XGBoost model
artifact. Therefore the transaction investigation endpoint is expected to
return HTTP 503 with:
Model artifact not found: /app/models/classical/xgboost_model.joblib

This is expected behavior and does not indicate a Docker runtime failure.

### Image contents

The Docker image includes:
/app
├── api/
├── src/
├── data/
│   ├── processed/features/feature_dataset.parquet
│   └── graph/graph_api_index.sqlite
└── requirements-api.txt

The following are intentionally excluded from the API image:
- .venv
- test files
- frontend dependencies/build output
- MLflow local tracking data
- raw/intermediate datasets
- the PyG heterogeneous_graph.pt artifact
- model artifacts
- notebooks
- local logs

The API uses the SQLite graph index rather than loading the PyG graph artifact.

This preserves the Phase 10.7 API architecture and avoids initializing the
large PyG graph inside the Uvicorn process.

### Runtime dependency decision

torch and torch-geometric are included in requirements-api.txt even though
the API does not directly load the materialized PyG graph.

The reason is the existing explainability import chain:
api.services.explanation
        ↓
fraud_intelligence.explainability
        ↓
gnn_importance
        ↓
torch

Removing these dependencies would prevent the FastAPI application from starting.

### Phase 12.5 validation result

Validated on the local Docker Desktop Linux/ARM64 environment:
- Docker image build: PASS
- Container startup: PASS
- Uvicorn startup: PASS
- /health: PASS
- /metadata: PASS
- /graph/{node_id}: PASS
- Transaction investigation error handling: PASS
- Expected missing-model HTTP 503: PASS

The Docker image is therefore validated as a working containerized API runtime.

EOF
printf '\n--- Docker files ---\n'
test -f Dockerfile && echo "Dockerfile: OK"
test -f requirements-api.txt && echo "requirements-api.txt: OK"
test -f .dockerignore && echo ".dockerignore: OK"
test -f docs/mlops/docker.md && echo "docker.md: OK"
printf '\n--- Git diff ---\n'
git status --short
git add Dockerfile requirements-api.txt .dockerignore docs/mlops/docker.md
git commit -m "feat: dockerize fraud intelligence API"
git push origin main
printf '\n--- Final status ---\n'
git status --short
git log -1 --oneline


### What this closes

This gives us a clean **Phase 12.5 Dockerization milestone**:

**Source → API runtime dependencies → Docker image → container → FastAPI → packaged data → endpoint validation**

And importantly, we're **not changing the frozen Phase 10 API**, not creating a fake XGBoost artifact, and not starting Phase 12.6.