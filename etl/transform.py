"""
Transform: Handle missing values dynamically, normalize column names, detect outliers,
encode categoricals, standardize numeric columns. Metadata-driven using inferred_schema.
"""
import logging
from typing import Dict, Any, List

import pandas as pd
import numpy as np

# Project root
ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from etl.schema_inference import (
    normalize_column_names,
    TYPE_DATE,
    TYPE_NUMERIC,
    TYPE_CATEGORICAL,
    TYPE_TEXT,
)

logger = logging.getLogger(__name__)


def _get_schema_lists(schema: Dict) -> tuple:
    """Extract date/numeric/categorical lists from inferred_schema."""
    if not schema:
        return [], [], []
    return (
        schema.get("date_columns") or [],
        schema.get("numeric_columns") or [],
        schema.get("categorical_columns") or [],
    )


def handle_missing(df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
    """Fill missing values: numeric -> median, categorical/text -> mode or 'unknown', date -> forward fill or drop."""
    df = df.copy()
    date_cols, numeric_cols, cat_cols = _get_schema_lists(schema)
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
        elif col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].ffill().bfill()
        else:
            mode_val = df[col].mode()
            fill = mode_val.iloc[0] if len(mode_val) else "unknown"
            df[col] = df[col].fillna(fill)
    return df


def detect_outliers_iqr(df: pd.DataFrame, numeric_columns: List[str], factor: float = 1.5) -> pd.DataFrame:
    """Cap numeric columns at IQR-based bounds (optional). Returns df with outliers capped."""
    df = df.copy()
    for col in numeric_columns:
        if col not in df.columns:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = ser.quantile(0.25), ser.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - factor * iqr, q3 + factor * iqr
        df[col] = ser.clip(low, high)
    return df


def encode_categoricals(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """Label-encode categorical columns for ML; leave originals for analytics or add _encoded suffix."""
    df = df.copy()
    for col in categorical_columns:
        if col not in df.columns:
            continue
        uniques = df[col].dropna().unique()
        mapping = {v: i for i, v in enumerate(uniques)}
        df[col + "_encoded"] = df[col].map(mapping).fillna(-1).astype(int)
    return df


def standardize_numeric(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """Z-score standardize numeric columns; add _scaled suffix to preserve originals."""
    df = df.copy()
    for col in numeric_columns:
        if col not in df.columns:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        mean, std = ser.mean(), ser.std()
        if std == 0 or np.isnan(std):
            df[col + "_scaled"] = 0.0
        else:
            df[col + "_scaled"] = (ser - mean) / std
    return df


def transform(
    df: pd.DataFrame,
    schema: Dict,
    *,
    normalize_names: bool = False,
    handle_missing_values: bool = True,
    cap_outliers: bool = True,
    encode_cats: bool = True,
    scale_numeric: bool = False,
) -> pd.DataFrame:
    """
    Full transform pipeline. Configurable flags for extensibility (e.g. client overrides).
    Column name normalization is off by default so schema (date/numeric/categorical lists) matches df columns;
    load step sanitizes names for SQL.
    """
    if normalize_names:
        df = normalize_column_names(df)
    date_cols = [c for c in (schema.get("date_columns") or []) if c in df.columns]
    numeric_cols = [c for c in (schema.get("numeric_columns") or []) if c in df.columns]
    cat_cols = [c for c in (schema.get("categorical_columns") or []) if c in df.columns]

    if handle_missing_values:
        schema_for_missing = {**schema, "date_columns": date_cols, "numeric_columns": numeric_cols, "categorical_columns": cat_cols}
        df = handle_missing(df, schema_for_missing)
    if cap_outliers and numeric_cols:
        df = detect_outliers_iqr(df, numeric_cols)
    if encode_cats and cat_cols:
        df = encode_categoricals(df, cat_cols)
    if scale_numeric and numeric_cols:
        df = standardize_numeric(df, numeric_cols)

    logger.info("Transform complete: %d rows, %d columns", len(df), len(df.columns))
    return df
