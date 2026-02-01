"""
Extract: Read uploaded data by client_id/dataset_id, validate row/column consistency, log schema mismatches.
Metadata-driven: uses raw_datasets.inferred_schema and file path from uploads.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys_path = str(ROOT)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from config.db import get_engine
from etl.schema_inference import infer_schema

logger = logging.getLogger(__name__)

UPLOADS_PATH = os.environ.get("DATA_UPLOADS_PATH", str(ROOT / "data" / "uploads"))


def get_raw_path(client_id: str, dataset_id: str) -> Path:
    """Resolve path to raw CSV for client/dataset."""
    return Path(UPLOADS_PATH) / client_id / f"{dataset_id}.csv"


def load_raw_metadata(engine, client_id: str, dataset_id: str) -> Optional[Dict[str, Any]]:
    """Load raw_datasets row for client/dataset. Returns inferred_schema and file info."""
    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT dataset_id, client_id, name, file_name, row_count, column_count, inferred_schema
                FROM raw_datasets WHERE client_id = :cid AND dataset_id = :did
            """),
            {"cid": client_id, "did": dataset_id},
        ).fetchone()
    if not row:
        return None
    return {
        "dataset_id": row[0],
        "client_id": row[1],
        "name": row[2],
        "file_name": row[3],
        "row_count": row[4],
        "column_count": row[5],
        "inferred_schema": row[6] if isinstance(row[6], dict) else (json.loads(row[6]) if row[6] else {}),
    }


def extract(client_id: str, dataset_id: str) -> Tuple[pd.DataFrame, Dict[str, Any], list]:
    """
    Extract CSV for client/dataset. Validate row count and column consistency vs raw_datasets.
    Returns (df, inferred_schema_dict, list of schema_mismatch messages for logging).
    """
    path = get_raw_path(client_id, dataset_id)
    if not path.exists():
        logger.warning("Raw file not found: %s", path)
        raise FileNotFoundError(f"Raw file not found: {path}")

    df = pd.read_csv(path)
    engine = get_engine()
    meta = load_raw_metadata(engine, client_id, dataset_id)
    mismatches = []

    if meta:
        expected_rows = meta.get("row_count")
        expected_cols = meta.get("column_count") or (len(meta.get("inferred_schema", {}).get("columns", [])))
        if expected_rows is not None and len(df) != expected_rows:
            mismatches.append(f"row_count_mismatch: expected {expected_rows}, got {len(df)}")
        if expected_cols is not None and len(df.columns) != expected_cols:
            mismatches.append(f"column_count_mismatch: expected {expected_cols}, got {len(df.columns)}")
        expected_col_names = meta.get("inferred_schema", {}).get("columns", [])
        if expected_col_names and set(df.columns) != set(expected_col_names):
            missing = set(expected_col_names) - set(df.columns)
            extra = set(df.columns) - set(expected_col_names)
            if missing:
                mismatches.append(f"missing_columns: {missing}")
            if extra:
                mismatches.append(f"extra_columns: {extra}")
    else:
        # No metadata: infer schema now
        meta = {"client_id": client_id, "dataset_id": dataset_id, "inferred_schema": infer_schema(df)}

    for m in mismatches:
        logger.warning("Schema mismatch [%s/%s]: %s", client_id, dataset_id, m)

    return df, meta.get("inferred_schema") or infer_schema(df), mismatches
