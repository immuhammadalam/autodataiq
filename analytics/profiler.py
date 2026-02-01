"""
Profiler: Automatically generate summary statistics for a dataset.
Generic: adapts to numeric/date/categorical columns. Saves to analytics_results.
"""
import json
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

# Project root
ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from config.db import get_engine
from config.schema import create_schema
from sqlalchemy import text

logger = logging.getLogger(__name__)


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary stats per column: count, nulls, dtype; for numeric: min, max, mean, std; for categorical: n_unique, top."""
    summary = {"columns": {}, "row_count": len(df), "column_count": len(df.columns)}
    for col in df.columns:
        ser = df[col]
        info = {"count": int(ser.count()), "nulls": int(ser.isna().sum()), "dtype": str(ser.dtype)}
        if pd.api.types.is_numeric_dtype(ser):
            info["min"] = float(ser.min()) if ser.notna().any() else None
            info["max"] = float(ser.max()) if ser.notna().any() else None
            info["mean"] = float(ser.mean()) if ser.notna().any() else None
            info["std"] = float(ser.std()) if ser.notna().any() else None
        elif pd.api.types.is_datetime64_any_dtype(ser):
            info["min"] = str(ser.min()) if ser.notna().any() else None
            info["max"] = str(ser.max()) if ser.notna().any() else None
        else:
            info["n_unique"] = int(ser.nunique())
            top = ser.value_counts().head(5)
            info["top_values"] = top.to_dict() if hasattr(top, "to_dict") else list(top.items())
        summary["columns"][col] = info
    return summary


def save_analytics_result(engine, client_id: str, dataset_id: str, result_type: str, result_key: str, result_value: Dict):
    """Insert one row into analytics_results."""
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO analytics_results (client_id, dataset_id, result_type, result_key, result_value)
                VALUES (:cid, :did, :rtype, :rkey, CAST(:rval AS jsonb))
            """),
            {
                "cid": client_id,
                "did": dataset_id,
                "rtype": result_type,
                "rkey": result_key,
                "rval": json.dumps(result_value, default=str),
            },
        )
        conn.commit()


def run_profiler(client_id: str, dataset_id: str) -> Dict[str, Any]:
    """Load processed table for client/dataset, profile, save to analytics_results. Returns summary."""
    engine = get_engine()
    create_schema(engine)
    table_name = _get_processed_table(engine, client_id, dataset_id)
    if not table_name:
        logger.warning("No processed table for client=%s dataset=%s", client_id, dataset_id)
        return {}
    df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
    summary = profile_dataframe(df)
    save_analytics_result(engine, client_id, dataset_id, "profiler", "summary_stats", summary)
    logger.info("Profiler saved for %s/%s", client_id, dataset_id)
    return summary


def _get_processed_table(engine, client_id: str, dataset_id: str) -> str:
    """Return processed table name for client/dataset."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT table_name FROM processed_datasets WHERE client_id = :cid AND dataset_id = :did"),
            {"cid": client_id, "did": dataset_id},
        ).fetchone()
    return row[0] if row else None
