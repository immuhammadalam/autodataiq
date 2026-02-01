# AutoDataIQ ETL service: extract, transform, load (metadata-driven, client-scoped)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY etl/ ./etl/
COPY ingestion/ ./ingestion/
COPY data/ ./data/

ENV PYTHONPATH=/app
ENV DATA_UPLOADS_PATH=/app/data/uploads
ENV DATA_RAW_PATH=/app/data/raw
ENV DATA_PROCESSED_PATH=/app/data/processed

# Default: run ETL for a client/dataset (pass --client-id and --dataset-id at run)
CMD ["python", "-m", "etl.run_etl", "--client-id", "default", "--dataset-id", "default"]
