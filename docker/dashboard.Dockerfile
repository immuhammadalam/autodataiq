# AutoDataIQ Dashboard: Streamlit, client/dataset selectors, B2B UI
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY etl/ ./etl/
COPY ingestion/ ./ingestion/
COPY analytics/ ./analytics/
COPY ml/ ./ml/
COPY dashboard/ ./dashboard/

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
