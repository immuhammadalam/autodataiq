"""
Database connection. All credentials from environment variables.
Designed for multi-tenant: all queries should filter by client_id where applicable.
"""
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


def get_connection_string() -> str:
    """Build connection string from env. No hardcoded credentials."""
    user = os.environ.get("DB_USER", "autodataiq")
    password = os.environ.get("DB_PASSWORD", "autodataiq_secret")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ.get("DB_NAME", "autodataiq")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def get_engine(pool_size: int = 5, max_overflow: int = 10) -> Engine:
    """Create SQLAlchemy engine with connection pooling."""
    conn_str = get_connection_string()
    engine = create_engine(
        conn_str,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        echo=False,
    )
    return engine
