"""
Predict: Load trained model for client/dataset, run predictions on processed data, return results.
Client-specific: model and data scoped by client_id and dataset_id.
Includes next-year forecast when model has a year feature.
"""
import os
import pickle
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from sqlalchemy import text

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from config.db import get_engine
from ml.auto_ml import prepare_xy

logger = logging.getLogger(__name__)


def _get_latest_model(engine, client_id: str, dataset_id: str, target_column: Optional[str] = None) -> Optional[dict]:
    """Return latest ml_models row. If target_column given, return that model; else latest overall."""
    with engine.connect() as conn:
        if target_column:
            row = conn.execute(
                text("""
                    SELECT model_id, artifact_path, feature_importance
                    FROM ml_models WHERE client_id = :cid AND dataset_id = :did AND status = 'trained' AND target_column = :target
                    ORDER BY created_at DESC LIMIT 1
                """),
                {"cid": client_id, "did": dataset_id, "target": target_column},
            ).fetchone()
        else:
            row = conn.execute(
                text("""
                    SELECT model_id, artifact_path, feature_importance
                    FROM ml_models WHERE client_id = :cid AND dataset_id = :did AND status = 'trained'
                    ORDER BY created_at DESC LIMIT 1
                """),
                {"cid": client_id, "did": dataset_id},
            ).fetchone()
    return {"model_id": row[0], "artifact_path": row[1], "feature_importance": row[2]} if row else None


def _get_trained_targets(engine, client_id: str, dataset_id: str) -> List[str]:
    """Return list of target_column values we have trained models for (e.g. ['revenue', 'quantity'])."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT target_column FROM ml_models
                WHERE client_id = :cid AND dataset_id = :did AND status = 'trained'
            """),
            {"cid": client_id, "did": dataset_id},
        ).fetchall()
    return [r[0] for r in rows if r[0]]


from config.date_columns import DATE_COLUMN_PRIORITY


def _add_year_from_date(df: pd.DataFrame) -> pd.DataFrame:
    """Infer date column from dataset (order_date, transaction_date, date, etc.) and add numeric year + month for predictions."""
    if "year" in df.columns:
        if "month" not in df.columns:
            # Try to add month from same date column used for year
            for col in df.columns:
                try:
                    ser = pd.to_datetime(df[col], errors="coerce")
                    if ser.notna().sum() / max(len(df), 1) > 0.5 and ser.dt.year.max() > 2000:
                        df["month"] = ser.dt.month
                        break
                except Exception:
                    pass
        return df
    df = df.copy()
    cols_lower = {c: c.lower().strip() for c in df.columns}
    for preferred in DATE_COLUMN_PRIORITY:
        for col in df.columns:
            if preferred not in cols_lower.get(col, ""):
                continue
            ser = pd.to_datetime(df[col], errors="coerce")
            if ser.notna().sum() / max(len(df), 1) > 0.5:
                years = ser.dt.year
                if years.notna().any() and years.max() > 2000:
                    df["year"] = years
                    df["month"] = ser.dt.month
                    return df
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df["year"] = pd.to_datetime(df[col], errors="coerce").dt.year
            df["month"] = pd.to_datetime(df[col], errors="coerce").dt.month
            return df
    for col in df.columns:
        try:
            ser = pd.to_datetime(df[col], errors="coerce")
            if ser.notna().sum() / max(len(df), 1) > 0.5:
                years = ser.dt.year
                if years.notna().any() and years.max() > 2000:
                    df["year"] = years
                    df["month"] = ser.dt.month
                    return df
        except Exception:
            pass
    return df


def _get_max_year(df: pd.DataFrame) -> Optional[int]:
    """Return max year from dataframe (year column or extracted from date column)."""
    df = _add_year_from_date(df)
    if "year" not in df.columns:
        return None
    return int(pd.to_numeric(df["year"], errors="coerce").dropna().max())


def _get_processed_table(engine, client_id: str, dataset_id: str) -> Optional[str]:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT table_name FROM processed_datasets WHERE client_id = :cid AND dataset_id = :did"),
            {"cid": client_id, "did": dataset_id},
        ).fetchone()
    return row[0] if row else None


def load_model(artifact_path: str) -> dict:
    """Load pickle artifact. Returns dict with model, feature_names, target_column, problem_type."""
    # Resolve path: if relative or container path, try MODEL_PATH
    if not os.path.isabs(artifact_path) or not os.path.exists(artifact_path):
        base = os.environ.get("MODEL_PATH", str(ROOT / "models"))
        candidate = os.path.join(base, os.path.basename(artifact_path))
        if os.path.exists(candidate):
            artifact_path = candidate
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def predict(
    client_id: str,
    dataset_id: str,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load latest model and processed table for client/dataset, run predictions, return DataFrame with predictions.
    """
    engine = get_engine()
    meta = _get_latest_model(engine, client_id, dataset_id)
    if not meta:
        raise FileNotFoundError(f"No trained model for client={client_id} dataset={dataset_id}")
    artifact = load_model(meta["artifact_path"])
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    target_column = artifact["target_column"]

    table_name = _get_processed_table(engine, client_id, dataset_id)
    if not table_name:
        raise FileNotFoundError(f"No processed table for client={client_id} dataset={dataset_id}")

    sql = f'SELECT * FROM "{table_name}"'
    if limit:
        sql += f" LIMIT {int(limit)}"
    df = pd.read_sql(sql, engine)
    if df.empty:
        return pd.DataFrame(columns=["prediction"])

    X, _, _ = prepare_xy(df, target_column, feature_columns=feature_names)
    X = X[feature_names] if all(c in X.columns for c in feature_names) else X
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_names]
    preds = model.predict(X)
    df = df.copy()
    df["prediction"] = preds
    if hasattr(model, "predict_proba"):
        df["prediction_probability"] = model.predict_proba(X)[:, 1]
    return df


def _predict_next_year_for_target(
    engine,
    client_id: str,
    dataset_id: str,
    target_column: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Predict total for one target (e.g. revenue or quantity) for next year. If target_column is None, use latest model."""
    meta = _get_latest_model(engine, client_id, dataset_id, target_column=target_column)
    if not meta:
        return None
    artifact = load_model(meta["artifact_path"])
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    target_column = artifact["target_column"]
    problem_type = artifact.get("problem_type", "regression")

    table_name = _get_processed_table(engine, client_id, dataset_id)
    if not table_name:
        return None
    df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
    if df.empty:
        return None

    df = _add_year_from_date(df)
    if "year" not in df.columns:
        logger.warning("No year/date column for next-year forecast")
        return None
    max_year = int(pd.to_numeric(df["year"], errors="coerce").dropna().max())
    next_year = max_year + 1

    if "year" not in feature_names:
        logger.warning("Model has no 'year' feature; cannot forecast next year")
        return None

    # Use last year's rows with year set to next year
    yr = pd.to_numeric(df["year"], errors="coerce")
    df_last = df[yr == max_year].copy()
    if df_last.empty:
        df_last = df.copy()
    df_last["year"] = next_year

    X_next = df_last.reindex(columns=feature_names)
    for c in feature_names:
        if c not in X_next.columns:
            X_next[c] = 0
    X_next = X_next[feature_names].fillna(0)

    preds = model.predict(X_next)
    predicted_total = float(np.sum(preds))

    return {
        "next_year": next_year,
        "predicted_total": predicted_total,
        "target_column": target_column,
        "problem_type": problem_type,
        "last_year_in_data": max_year,
    }


def predict_next_year(
    client_id: str,
    dataset_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Predict total target (e.g. sales) for the year after the last year in the dataset.
    Uses the latest trained model. Returns dict with next_year, predicted_total, target_column, or None.
    """
    engine = get_engine()
    return _predict_next_year_for_target(engine, client_id, dataset_id, target_column=None)


def get_all_next_year_forecasts(client_id: str, dataset_id: str) -> List[Dict[str, Any]]:
    """
    Return next-year forecasts for all trained targets (e.g. revenue and quantity).
    Each item has next_year, predicted_total, target_column, last_year_in_data.
    """
    engine = get_engine()
    targets = _get_trained_targets(engine, client_id, dataset_id)
    results = []
    for target in targets:
        meta = _get_latest_model(engine, client_id, dataset_id, target_column=target)
        if not meta:
            continue
        artifact = load_model(meta["artifact_path"])
        model = artifact["model"]
        feature_names = artifact["feature_names"]
        table_name = _get_processed_table(engine, client_id, dataset_id)
        if not table_name:
            continue
        df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
        if df.empty:
            continue
        df = _add_year_from_date(df)
        if "year" not in df.columns or "year" not in feature_names:
            continue
        max_year = int(pd.to_numeric(df["year"], errors="coerce").dropna().max())
        next_year = max_year + 1
        yr = pd.to_numeric(df["year"], errors="coerce")
        df_last = df[yr == max_year].copy()
        if df_last.empty:
            df_last = df.copy()
        df_last["year"] = next_year
        X_next = df_last.reindex(columns=feature_names).fillna(0)
        for c in feature_names:
            if c not in X_next.columns:
                X_next[c] = 0
        X_next = X_next[feature_names]
        preds = model.predict(X_next)
        results.append({
            "next_year": next_year,
            "predicted_total": float(np.sum(preds)),
            "target_column": target,
            "last_year_in_data": max_year,
        })
    return results


def get_next_year_predictions_table(client_id: str, dataset_id: str) -> pd.DataFrame:
    """
    Return a DataFrame of predictions for each month of the year after the last year in the dataset's date column.
    Columns: prediction_date (first day of each month, e.g. 2026-01-01 .. 2026-12-01), prediction_sales (units), prediction_revenue.
    One row per month (12 rows). Date column is inferred from dataset (order_date, transaction_date, date, etc.).
    """
    engine = get_engine()
    table_name = _get_processed_table(engine, client_id, dataset_id)
    if not table_name:
        return pd.DataFrame()
    df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
    if df.empty:
        return pd.DataFrame()
    df = _add_year_from_date(df)
    if "year" not in df.columns:
        return pd.DataFrame()
    max_year = int(pd.to_numeric(df["year"], errors="coerce").dropna().max())
    next_year = max_year + 1
    yr = pd.to_numeric(df["year"], errors="coerce")
    df_last = df[yr == max_year].copy()
    if df_last.empty:
        df_last = df.copy()
    if "month" not in df_last.columns:
        df_last["month"] = 1

    targets = _get_trained_targets(engine, client_id, dataset_id)
    # Dataset-agnostic: one prediction column per target, named prediction_{target}
    rows = []
    for month in range(1, 13):
        prediction_date = f"{next_year}-{month:02d}-01"
        row = {"prediction_date": prediction_date}

        df_month = df_last.copy()
        df_month["year"] = next_year
        df_month["month"] = month

        for target in targets:
            meta = _get_latest_model(engine, client_id, dataset_id, target_column=target)
            if not meta:
                continue
            artifact = load_model(meta["artifact_path"])
            model = artifact["model"]
            feature_names = artifact["feature_names"]
            if "year" not in feature_names:
                continue
            X_next = df_month.reindex(columns=feature_names).fillna(0)
            for c in feature_names:
                if c not in X_next.columns:
                    X_next[c] = 0
            X_next = X_next[feature_names]
            preds = model.predict(X_next)
            total = float(np.sum(preds))
            row[f"prediction_{target}"] = total

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame()
    # Dynamic columns: prediction_date + all prediction_* for each trained target (dataset-agnostic)
    pred_cols = [c for c in out.columns if c.startswith("prediction_") and c != "prediction_date"]
    out = out[["prediction_date"] + pred_cols].copy()
    return out
