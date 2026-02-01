"""
Application schema: clients, raw_datasets, processed_datasets, analytics_results, ml_models.
Generic design: no business-specific columns; client_id and dataset_id tag all data.
"""
import logging
from sqlalchemy import text

from config.db import get_engine

logger = logging.getLogger(__name__)


def create_schema(engine):
    """Create all application tables if not exist. Idempotent."""
    with engine.connect() as conn:
        # Clients: one row per tenant
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS clients (
                client_id VARCHAR(64) PRIMARY KEY,
                name VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        # Raw datasets: metadata for uploaded files; actual raw data stored as JSON/table per dataset
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_datasets (
                dataset_id VARCHAR(64) PRIMARY KEY,
                client_id VARCHAR(64) NOT NULL REFERENCES clients(client_id),
                name VARCHAR(255),
                file_name VARCHAR(255),
                row_count INT,
                column_count INT,
                inferred_schema JSONB,
                status VARCHAR(32) DEFAULT 'uploaded',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        # Processed datasets: cleaned data table name + metadata
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS processed_datasets (
                dataset_id VARCHAR(64) PRIMARY KEY,
                client_id VARCHAR(64) NOT NULL REFERENCES clients(client_id),
                raw_dataset_id VARCHAR(64) REFERENCES raw_datasets(dataset_id),
                table_name VARCHAR(128) NOT NULL,
                row_count INT,
                column_count INT,
                processed_schema JSONB,
                status VARCHAR(32) DEFAULT 'processed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        # Analytics results: store summary stats, trends, correlations per client/dataset
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS analytics_results (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(64) NOT NULL REFERENCES clients(client_id),
                dataset_id VARCHAR(64) NOT NULL,
                result_type VARCHAR(64),
                result_key VARCHAR(128),
                result_value JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        # ML models: metadata and metrics per client/dataset
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ml_models (
                model_id VARCHAR(64) PRIMARY KEY,
                client_id VARCHAR(64) NOT NULL REFERENCES clients(client_id),
                dataset_id VARCHAR(64) NOT NULL,
                problem_type VARCHAR(32),
                target_column VARCHAR(128),
                model_type VARCHAR(64),
                metrics JSONB,
                feature_importance JSONB,
                artifact_path VARCHAR(512),
                status VARCHAR(32) DEFAULT 'trained',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.commit()
    logger.info("Application schema created.")


def ensure_client(engine, client_id: str, name: str = None):
    """Upsert a client. Call before any client-scoped operation."""
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO clients (client_id, name) VALUES (:cid, :name)
                ON CONFLICT (client_id) DO UPDATE SET name = COALESCE(EXCLUDED.name, clients.name)
            """),
            {"cid": client_id, "name": name or client_id},
        )
        conn.commit()
