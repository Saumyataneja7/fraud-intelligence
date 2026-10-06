FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY src ./src

COPY data/processed/features/feature_dataset.parquet \
     ./data/processed/features/feature_dataset.parquet

COPY data/graph/graph_api_index.sqlite \
     ./data/graph/graph_api_index.sqlite

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
