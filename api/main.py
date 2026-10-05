from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.contracts.common import HealthResponse
from api.contracts.fraud_ring import FraudRingInvestigationResponse

from api.contracts.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from api.services.prediction import (
    PredictionService,
    PredictionServiceError,
)
from api.contracts.transaction import (
    TransactionInvestigationResponse,
)
from api.services.transaction_investigation import (
    TransactionInvestigationService,
    TransactionInvestigationServiceError,
)
from api.contracts.entity import (
    EntityInvestigationResponse,
)
from api.services.entity_investigation import (
    EntityInvestigationService,
    EntityInvestigationServiceError,
)
from api.services.fraud_ring import (
    FraudRingInvestigationService,
    FraudRingInvestigationServiceError,
)
from api.contracts.graph import GraphResponse
from api.services.graph import (
    GraphInvestigationService,
    GraphInvestigationServiceError,
)

from api.services.graph import (
    GraphInvestigationService,
    GraphInvestigationServiceError,
)
from api.contracts.explanation import ExplanationResponse
from api.services.explanation import (
    ExplanationService,
    ExplanationServiceError,
)
from api.contracts.model_metrics import ModelMetricsResponse
from api.services.model_metrics import (
    ModelMetricsService,
    ModelMetricsServiceError,
)
from api.services.transaction_fraud_ring import (
    TransactionFraudRingService,
    TransactionFraudRingServiceError,
)

API_VERSION = "1.0.0"
SERVICE_NAME = "fraud-intelligence-api"


app = FastAPI(
    title="Fraud Intelligence API",
    description=(
        "Production-style API for the Fraud Intelligence "
        "investigation platform."
    ),
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
)
def health() -> HealthResponse:
    """Return API health status."""

    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=API_VERSION,
    )


@app.get(
    "/metadata",
    tags=["system"],
)
def metadata() -> dict[str, object]:
    """Return static API and project metadata."""

    return {
        "service": SERVICE_NAME,
        "version": API_VERSION,
        "project": "Fraud Intelligence",
        "phase": "10.2",
        "status": "operational",
        "dataset": {
            "name": "fraud-intelligence-synthetic",
            "version": "1.0.0",
        },
        "architecture": {
            "feature_engineering": "Phase 4",
            "classical_ml": "Phase 5",
            "graph": "Phase 6",
            "graph_ml": "Phase 7",
            "explainability": "Phase 8",
            "fraud_intelligence": "Phase 9",
            "api": "Phase 10",
        },
        "frozen_phases": [
            "Phase 1",
            "Phase 2",
            "Phase 3",
            "Phase 4",
            "Phase 5",
            "Phase 6",
            "Phase 7",
            "Phase 8",
            "Phase 9",
        ],
    }

prediction_service = PredictionService()

transaction_investigation_service = (
    TransactionInvestigationService(
        prediction_service=prediction_service,
    )
)

entity_investigation_service = (
    EntityInvestigationService()
)

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
)
def predict(
    request: PredictionRequest = Body(...),
) -> PredictionResponse:
    """Predict fraud probability for a transaction."""

    try:
        return prediction_service.predict(
            request.transaction_id
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except PredictionServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
@app.get(
    "/transaction/{transaction_id}",
    response_model=TransactionInvestigationResponse,
    tags=["transaction"],
)

def investigate_transaction(
    transaction_id: str,
) -> TransactionInvestigationResponse:
    """Return investigation intelligence for a transaction."""

    try:
        return transaction_investigation_service.investigate(
            transaction_id
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except (
        PredictionServiceError,
        TransactionInvestigationServiceError,
    ) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

@app.get(
    "/customer/{customer_id}",
    response_model=EntityInvestigationResponse,
    tags=["entity"],
)
def investigate_customer(
    customer_id: str,
) -> EntityInvestigationResponse:
    """Return investigation intelligence for a customer."""

    try:
        return entity_investigation_service.investigate(
            customer_id
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except EntityInvestigationServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


fraud_ring_investigation_service = FraudRingInvestigationService()

@app.get(
    "/fraud-ring/{ring_id}",
    response_model=FraudRingInvestigationResponse,
    tags=["fraud-ring"],
)
def investigate_fraud_ring(
    ring_id: str,
) -> FraudRingInvestigationResponse:
    try:
        return fraud_ring_investigation_service.investigate(
            ring_id
        )

    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Fraud ring {ring_id!r} was not found.",
        )

    except FraudRingInvestigationServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

transaction_fraud_ring_service = TransactionFraudRingService()


@app.get(
    "/transaction/{transaction_id}/fraud-ring",
    response_model=FraudRingInvestigationResponse,
    tags=["fraud-ring"],
)
def investigate_transaction_fraud_ring(
    transaction_id: str,
) -> FraudRingInvestigationResponse:
    try:
        return transaction_fraud_ring_service.investigate(
            transaction_id
        )

    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No fraud-ring candidate was found for "
                f"transaction {transaction_id!r}."
            ),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except TransactionFraudRingServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

graph_investigation_service = GraphInvestigationService()

@app.get(
    "/graph/{node_id}",
    response_model=GraphResponse,
    tags=["graph"],
)
def investigate_graph(node_id: int, node_type: str) -> GraphResponse:
    try:
        return graph_investigation_service.investigate(
            node_type=node_type,
            node_id=node_id,
        )
    except GraphInvestigationServiceError as exc:
        message = str(exc)

        if message.startswith("Unknown graph node"):
            raise HTTPException(status_code=404, detail=message) from exc

        if message.startswith("Unsupported graph node type"):
            raise HTTPException(status_code=404, detail=message) from exc

        if message.startswith("Invalid graph node id"):
            raise HTTPException(status_code=422, detail=message) from exc

        raise HTTPException(
            status_code=503,
            detail="Graph investigation service is unavailable.",
        ) from exc


explanation_service = ExplanationService()

@app.get(
    "/explanation/{transaction_id}",
    response_model=ExplanationResponse,
    tags=["explanation"],
)
def explain_transaction(
    transaction_id: str,
) -> ExplanationResponse:
    """Return explainability evidence for a transaction."""

    try:
        return explanation_service.explain(
            transaction_id
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except ExplanationServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


model_metrics_service = ModelMetricsService()

@app.get(
    "/model-metrics",
    response_model=ModelMetricsResponse,
    tags=["model"],
)
def model_metrics() -> ModelMetricsResponse:
    """Return frozen Phase 5 model evaluation metrics."""

    try:
        return model_metrics_service.get_metrics()

    except ModelMetricsServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc