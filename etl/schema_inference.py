"""
Schema inference: detect data types, date/numeric/categorical columns.
Generic - no business assumptions. Used by ingestion and ETL.
"""
import logging
from typing import Dict, List, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Type categories for downstream ETL and analytics
TYPE_DATE = "date"
TYPE_NUMERIC = "numeric"
TYPE_CATEGORICAL = "categorical"
TYPE_TEXT = "text"
TYPE_UNKNOWN = "unknown"


def _infer_column_type(series: pd.Series) -> str:
    """Infer high-level type: date, numeric, categorical, or text."""
    if series.isna().all():
        return TYPE_UNKNOWN
    # Try numeric
    if pd.api.types.is_numeric_dtype(series):
        return TYPE_NUMERIC
    # Try datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return TYPE_DATE
    # Try parsing as date
    sample = series.dropna().head(100)
    if len(sample) > 0:
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() / max(len(sample), 1) > 0.8:
                return TYPE_DATE
        except Exception:
            pass
    # Object/string
    n_unique = series.nunique()
    n_total = len(series.dropna())
    if n_total == 0:
        return TYPE_UNKNOWN
    # High cardinality -> text; low -> categorical
    if n_unique / n_total > 0.5 or n_unique > 100:
        return TYPE_TEXT
    return TYPE_CATEGORICAL


def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Infer schema: dtype per column, and lists of date/numeric/categorical columns.
    Returns dict suitable for JSONB storage and ETL config.
    """
    column_types = {}
    date_cols = []
    numeric_cols = []
    categorical_cols = []
    text_cols = []

    for col in df.columns:
        col_type = _infer_column_type(df[col])
        column_types[col] = {
            "inferred": col_type,
            "dtype": str(df[col].dtype),
            "nullable": df[col].isna().any(),
        }
        if col_type == TYPE_DATE:
            date_cols.append(col)
        elif col_type == TYPE_NUMERIC:
            numeric_cols.append(col)
        elif col_type == TYPE_CATEGORICAL:
            categorical_cols.append(col)
        elif col_type == TYPE_TEXT:
            text_cols.append(col)

    return {
        "columns": list(df.columns),
        "column_types": column_types,
        "date_columns": date_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "text_columns": text_cols,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
    }


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: lowercase, replace spaces/special with underscore."""
    df = df.copy()
    new_names = []
    for c in df.columns:
        s = str(c).strip().lower()
        s = "".join(c if c.isalnum() or c == "_" else "_" for c in s)
        s = "_".join(s.split())
        new_names.append(s or "unnamed")
    df.columns = new_names
    return df
