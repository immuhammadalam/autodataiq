# AutoDataIQ ML service: auto-ML train/predict, client-specific models
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY etl/ ./etl/
COPY analytics/ ./analytics/
COPY ml/ ./ml/

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models

RUN mkdir -p /app/models

# Default: train for a client/dataset (override with command)
CMD ["python", "-c", "from ml.train import train; train('default', 'default')"]
