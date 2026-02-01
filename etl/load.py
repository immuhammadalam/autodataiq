"""
Load: Store cleaned data into PostgreSQL; maintain raw vs processed; tag all records with client_id.
Processed data stored in a table per dataset: data_{client_id}_{dataset_id} (sanitized names).
Metadata in processed_datasets.
"""
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sqlalchemy import text

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from config.db import get_engine
from config.schema import create_schema, ensure_client

logger = logging.getLogger(__name__)


def _sanitize_table_name(s: str) -> str:
    """Sanitize for PostgreSQL table name: alphanumeric and underscore only."""
    return re.sub(r"[^a-z0-9_]", "_", s.lower())[:63]


def _table_name(client_id: str, dataset_id: str) -> str:
    """Processed table name: data_{client_id}_{dataset_id} (sanitized)."""
    return "data_" + _sanitize_table_name(client_id) + "_" + _sanitize_table_name(dataset_id)


def _sanitize_column(name: str) -> str:
    """Sanitize column name for PostgreSQL."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", str(name).strip().lower())
    return s or "col_" + str(hash(name) % 10000)


def load(
    client_id: str,
    dataset_id: str,
    df: pd.DataFrame,
    schema: Dict[str, Any],
) -> str:
    """
    Create processed table (or replace), insert cleaned data, add client_id column, update processed_datasets.
    Returns table_name.
    """
    engine = get_engine()
    create_schema(engine)
    ensure_client(engine, client_id)

    table_name = _table_name(client_id, dataset_id)
    df = df.copy()
    df.columns = [_sanitize_column(c) for c in df.columns]
    df["client_id"] = client_id
    df["dataset_id"] = dataset_id

    # Convert datetime columns to string for PostgreSQL
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
        conn.commit()

    df.to_sql(table_name, engine, index=False, method="multi", chunksize=1000)
    logger.info("Loaded %s: %d rows", table_name, len(df))

    processed_schema = {
        "columns": list(df.columns),
        "row_count": len(df),
        "column_count": len(df.columns),
    }
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO processed_datasets (dataset_id, client_id, raw_dataset_id, table_name, row_count, column_count, processed_schema, status)
                VALUES (:did, :cid, :rid, :tname, :rows, :cols, CAST(:schema_json AS jsonb), 'processed')
                ON CONFLICT (dataset_id) DO UPDATE SET
                    table_name = EXCLUDED.table_name,
                    row_count = EXCLUDED.row_count,
                    column_count = EXCLUDED.column_count,
                    processed_schema = EXCLUDED.processed_schema,
                    status = 'processed'
            """),
            {
                "did": dataset_id,
                "cid": client_id,
                "rid": dataset_id,
                "tname": table_name,
                "rows": len(df),
                "cols": len(df.columns),
                "schema_json": json.dumps(processed_schema, default=str),
            },
        )
        conn.commit()

    return table_name
