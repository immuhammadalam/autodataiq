"""
Train: Load processed data for client/dataset, run ML readiness gate, model selection, train, save artifact and metadata to ml_models.
Client-specific: each model tagged with client_id and dataset_id.
Uses time-aware train/validation split when date column exists; stores user-facing metrics and explainability.
"""
import os
import json
import uuid
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np
from sqlalchemy import text

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from config.db import get_engine
from config.schema import create_schema
from ml.auto_ml import (
    infer_target_column,
    assess_ml_readiness,
    select_model,
    prepare_xy,
    compute_metrics,
    get_feature_importance,
    build_explainability,
    PROBLEM_CLASSIFICATION,
    PROBLEM_REGRESSION,
    PROBLEM_FORECASTING,
    PROBLEM_NOT_RECOMMENDED,
    RANDOM_STATE,
    TEST_SIZE,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "models"))


def _get_processed_table(engine, client_id: str, dataset_id: str) -> Optional[str]:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT table_name FROM processed_datasets WHERE client_id = :cid AND dataset_id = :did"),
            {"cid": client_id, "did": dataset_id},
        ).fetchone()
    return row[0] if row else None


def _time_aware_split(
    df: pd.DataFrame,
    target_column: str,
    date_column: Optional[str],
    feature_columns: Optional[list],
    test_size: float = 0.25,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """Prepare X,y and split train/validation. If date_column exists, sort by date and take last test_size as validation."""
    df_work = df.copy()
    if date_column and date_column in df_work.columns:
        try:
            df_work["_sort_ts"] = pd.to_datetime(df_work[date_column], errors="coerce")
            df_work = df_work.sort_values("_sort_ts").drop(columns=["_sort_ts"], errors="ignore")
        except Exception:
            pass
    X, y, feature_names = prepare_xy(df_work, target_column, feature_columns)
    if len(X) < 10:
        return X, y, pd.DataFrame(), pd.Series(dtype=float), feature_names

    n_val = max(1, int(len(X) * test_size))
    if date_column and date_column in df.columns:
        X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
        y_train, y_val = y.iloc[:-n_val], y.iloc[-n_val:]
        return X_train, y_train, X_val, y_val, feature_names

    use_stratify = False
    if y.nunique() <= 20:
        from collections import Counter
        counts = Counter(y)
        use_stratify = all(c >= 2 for c in counts.values()) and len(counts) >= 2
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE,
        stratify=y if use_stratify else None,
    )
    return X_train, y_train, X_val, y_val, feature_names


def _metrics_to_user_facing(metrics: dict, problem_type: str) -> str:
    """Convert metrics into user-facing language (e.g. '±8% accuracy')."""
    if problem_type == PROBLEM_CLASSIFICATION:
        acc = metrics.get("accuracy")
        f1 = metrics.get("f1")
        if acc is not None:
            pct = round(acc * 100)
            return f"Accuracy: {pct}% correct predictions."
        if f1 is not None:
            return f"F1 score: {round(f1, 2)} (balance of precision and recall)."
        return "Classification metrics computed."
    rmse = metrics.get("rmse")
    mape = metrics.get("mape")
    r2 = metrics.get("r2")
    parts = []
    if rmse is not None and np.isfinite(rmse):
        parts.append(f"Typical error: ±{rmse:,.2f} (RMSE)")
    if mape is not None and np.isfinite(mape):
        parts.append(f"±{mape:.1f}% average error (MAPE)")
    if r2 is not None and np.isfinite(r2):
        parts.append(f"{round(r2 * 100)}% of variation explained")
    return " ".join(parts) if parts else "Regression metrics computed."


def train(
    client_id: str,
    dataset_id: str,
    target_column: Optional[str] = None,
    feature_columns: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Load processed table, run ML readiness gate; if allowed, select model, train with time-aware split, store metrics and explainability.
    If ML not recommended, insert ml_models row with status='not_recommended' and skip training.
    Returns model_id (or None), metrics, problem_type, target_column.
    """
    engine = get_engine()
    create_schema(engine)
    table_name = _get_processed_table(engine, client_id, dataset_id)
    if not table_name:
        raise FileNotFoundError(f"No processed table for client={client_id} dataset={dataset_id}")

    df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
    if df.empty or len(df) < 20:
        raise ValueError("Insufficient data for training (need at least 20 rows)")

    target_column = target_column or infer_target_column(df)
    if not target_column:
        raise ValueError("Could not infer target column; specify target_column")

    from ml.auto_ml import _detect_date_column
    date_column = _detect_date_column(df)
    inferred_schema = {"target_column": target_column, "date_column": date_column}
    readiness = assess_ml_readiness(df, inferred_schema)

    if not readiness.get("ml_allowed", False):
        model_id = str(uuid.uuid4())[:12]
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO ml_models (model_id, client_id, dataset_id, problem_type, target_column, status, reason, confidence_level)
                    VALUES (:mid, :cid, :did, :ptype, :target, 'not_recommended', :reason, :conf)
                """),
                {
                    "mid": model_id,
                    "cid": client_id,
                    "did": dataset_id,
                    "ptype": readiness.get("problem_type", PROBLEM_NOT_RECOMMENDED),
                    "target": target_column,
                    "reason": readiness.get("reason", "ML not recommended"),
                    "conf": readiness.get("confidence_level", "low"),
                },
            )
            conn.commit()
        logger.info("ML not recommended: %s", readiness.get("reason"))
        return {
            "model_id": None,
            "metrics": {},
            "problem_type": readiness.get("problem_type"),
            "target_column": target_column,
            "status": "not_recommended",
            "reason": readiness.get("reason"),
        }

    problem_type = readiness["problem_type"]
    # For training/metrics we treat forecasting as regression
    metrics_problem_type = PROBLEM_REGRESSION if problem_type == PROBLEM_FORECASTING else problem_type

    X_train, y_train, X_val, y_val, feature_names = _time_aware_split(
        df, target_column, date_column, feature_columns, test_size=TEST_SIZE,
    )
    if X_val.empty or len(X_val) < 2:
        X, y, feature_names = prepare_xy(df, target_column, feature_columns)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )

    data_size = len(X_train) + len(X_val)
    model, model_name, reason = select_model(problem_type, data_size)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    metrics = compute_metrics(y_val, y_pred, metrics_problem_type)
    metrics["user_facing"] = _metrics_to_user_facing(metrics, metrics_problem_type)
    feature_importance = get_feature_importance(model, feature_names)
    explainability = build_explainability(feature_importance)

    prediction_horizon = "next_12_months" if (problem_type == PROBLEM_FORECASTING and date_column) else "single_period"
    confidence_level = readiness.get("confidence_level", "medium")

    model_id = str(uuid.uuid4())[:12]
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
    artifact_path = os.path.join(MODEL_PATH, f"{client_id}_{dataset_id}_{model_id}.pkl")
    with open(artifact_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": feature_names,
            "target_column": target_column,
            "problem_type": problem_type,
            "date_column": date_column,
        }, f)
    logger.info("Saved model artifact: %s", artifact_path)

    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO ml_models (
                    model_id, client_id, dataset_id, problem_type, target_column, model_type,
                    metrics, feature_importance, artifact_path, status,
                    reason, prediction_horizon, confidence_level, explainability
                )
                VALUES (
                    :mid, :cid, :did, :ptype, :target, :mtype,
                    CAST(:metrics_json AS jsonb), CAST(:fi_json AS jsonb), :path, 'trained',
                    :reason, :horizon, :conf, CAST(:explain_json AS jsonb)
                )
            """),
            {
                "mid": model_id,
                "cid": client_id,
                "did": dataset_id,
                "ptype": problem_type,
                "target": target_column,
                "mtype": model_name,
                "metrics_json": json.dumps(metrics),
                "fi_json": json.dumps(feature_importance),
                "path": artifact_path,
                "reason": reason,
                "horizon": prediction_horizon,
                "conf": confidence_level,
                "explain_json": json.dumps(explainability),
            },
        )
        conn.commit()

    return {
        "model_id": model_id,
        "metrics": metrics,
        "problem_type": problem_type,
        "target_column": target_column,
        "user_facing": metrics.get("user_facing", ""),
        "prediction_horizon": prediction_horizon,
        "confidence_level": confidence_level,
    }
