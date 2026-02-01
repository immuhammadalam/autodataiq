"""
Upload handler: accept CSV uploads, store with client_id/dataset_id, infer schema, register in raw_datasets.
Entry point for any client upload (dashboard or API). No business-specific logic.
"""
import os
import uuid
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys_path = str(ROOT)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from config.db import get_engine
from config.schema import create_schema, ensure_client
from etl.schema_inference import infer_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

UPLOADS_PATH = os.environ.get("DATA_UPLOADS_PATH", str(ROOT / "data" / "uploads"))


def save_upload(
    client_id: str,
    file_path_or_buffer,
    dataset_id: Optional[str] = None,
    name: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Tuple[str, dict]:
    """
    Save uploaded CSV and register in raw_datasets.
    file_path_or_buffer: path to CSV file or file-like object (e.g. Streamlit UploadedFile).
    max_rows: if set, only load this many rows (for large files to avoid timeout).
    Returns (dataset_id, inferred_schema).
    """
    dataset_id = dataset_id or str(uuid.uuid4())[:8]
    ensure_client(get_engine(), client_id, name=name or client_id)

    # Rewind buffer so read works (Streamlit UploadedFile can be consumed once)
    if hasattr(file_path_or_buffer, "seek"):
        file_path_or_buffer.seek(0)
    if hasattr(file_path_or_buffer, "read"):
        df = pd.read_csv(file_path_or_buffer, nrows=max_rows)
    else:
        df = pd.read_csv(file_path_or_buffer, nrows=max_rows)

    if df.empty:
        raise ValueError("Uploaded file has no rows")

    schema = infer_schema(df)
    out_dir = Path(UPLOADS_PATH) / client_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset_id}.csv"
    df.to_csv(out_file, index=False)
    logger.info("Saved upload: client=%s dataset=%s rows=%d path=%s", client_id, dataset_id, len(df), out_file)

    engine = get_engine()
    create_schema(engine)
    with engine.connect() as conn:
        from sqlalchemy import text
        conn.execute(
            text("""
                INSERT INTO raw_datasets (dataset_id, client_id, name, file_name, row_count, column_count, inferred_schema, status)
                VALUES (:did, :cid, :name, :fname, :rows, :cols, CAST(:schema_json AS jsonb), 'uploaded')
                ON CONFLICT (dataset_id) DO UPDATE SET
                    name = COALESCE(EXCLUDED.name, raw_datasets.name),
                    file_name = EXCLUDED.file_name,
                    row_count = EXCLUDED.row_count,
                    column_count = EXCLUDED.column_count,
                    inferred_schema = EXCLUDED.inferred_schema,
                    status = 'uploaded'
            """),
            {
                "did": dataset_id,
                "cid": client_id,
                "name": name or dataset_id,
                "fname": out_file.name,
                "rows": schema["row_count"],
                "cols": schema["column_count"],
                "schema_json": json.dumps(schema, default=str),
            },
        )
        conn.commit()

    return dataset_id, schema
