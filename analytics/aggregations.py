"""
Aggregations: Trends over time, correlations. Adapts to dataset type (time-series if date cols, else numeric aggregations).
Saves to analytics_results for dashboard consumption.
"""
import json
import logging
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

# Project root
ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from config.db import get_engine
from config.schema import create_schema
from sqlalchemy import text

from analytics.profiler import save_analytics_result, _get_processed_table

logger = logging.getLogger(__name__)


from config.date_columns import DATE_COLUMN_PRIORITY


def _infer_date_column(df: pd.DataFrame) -> Optional[str]:
    """Infer date column from dataset (order_date, transaction_date, date, etc.) for trend aggregation."""
    cols_lower = {c: c.lower().strip() for c in df.columns}
    for preferred in DATE_COLUMN_PRIORITY:
        for col in df.columns:
            if preferred not in cols_lower.get(col, ""):
                continue
            try:
                ser = pd.to_datetime(df[col], errors="coerce")
                if ser.notna().sum() / max(len(df), 1) > 0.5:
                    years = ser.dt.year
                    if years.notna().any() and years.max() > 2000:
                        return col
            except Exception:
                pass
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if df[col].max().year > 2000 if hasattr(df[col].max(), "year") else True:
                return col
        try:
            ser = pd.to_datetime(df[col], errors="coerce")
            if ser.notna().sum() / max(len(df), 1) > 0.5:
                if ser.dt.year.max() > 2000:
                    return col
        except Exception:
            pass
    return None


def trends_over_time(df: pd.DataFrame, date_col: str, value_cols: List[str]) -> Dict[str, Any]:
    """Aggregate value_cols by month from date_col. Uses order_date; period output as string YYYY-MM for proper trend lines."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty or value_cols is None:
        return {}
    # Use year-month as string (e.g. 2022-01) so JSON and charts get correct x-axis, not 1970
    df["_period"] = df[date_col].dt.to_period("M").astype(str)
    result = {}
    for v in value_cols:
        if v not in df.columns:
            continue
        agg = df.groupby("_period")[v].apply(lambda x: pd.to_numeric(x, errors="coerce").sum()).reset_index()
        agg.columns = ["period", "value"]
        agg["period"] = agg["period"].astype(str)
        result[v] = agg.to_dict(orient="records")
    return result


def correlations(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
    """Compute correlation matrix for numeric columns. Returns upper triangle as list of (col1, col2, corr)."""
    if not numeric_columns:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    sub = df[[c for c in numeric_columns if c in df.columns]]
    if sub.shape[1] < 2:
        return {"pairs": [], "matrix": {}}
    corr = sub.corr()
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append({
                "col1": corr.columns[i],
                "col2": corr.columns[j],
                "correlation": float(corr.iloc[i, j]),
            })
    return {"pairs": pairs, "matrix": corr.fillna(0).to_dict()}


def run_aggregations(client_id: str, dataset_id: str) -> Dict[str, Any]:
    """
    Load processed table, compute trends (if date col) and correlations (if numeric cols), save to analytics_results.
    """
    engine = get_engine()
    create_schema(engine)
    table_name = _get_processed_table(engine, client_id, dataset_id)
    if not table_name:
        return {}
    df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
    results = {}

    date_col = _infer_date_column(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude client_id/dataset_id if present
    numeric_cols = [c for c in numeric_cols if c not in ("client_id", "dataset_id")]

    if date_col and numeric_cols:
        trends = trends_over_time(df, date_col, numeric_cols[:5])
        if trends:
            save_analytics_result(engine, client_id, dataset_id, "aggregations", "trends", trends)
            results["trends"] = trends

    if len(numeric_cols) >= 2:
        corr = correlations(df, numeric_cols)
        save_analytics_result(engine, client_id, dataset_id, "aggregations", "correlations", corr)
        results["correlations"] = corr

    logger.info("Aggregations saved for %s/%s", client_id, dataset_id)
    return results
