"""
Train: Load processed data for client/dataset, run auto_ml (infer target, problem type), train, save artifact and metadata to ml_models.
Client-specific: each model tagged with client_id and dataset_id.
"""
import os
import json
import uuid
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from sqlalchemy import text

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from config.db import get_engine
from config.schema import create_schema
from ml.auto_ml import (
    infer_problem_type,
    infer_target_column,
    get_baseline_model,
    prepare_xy,
    compute_metrics,
    get_feature_importance,
    PROBLEM_CLASSIFICATION,
    PROBLEM_REGRESSION,
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


def train(
    client_id: str,
    dataset_id: str,
    target_column: Optional[str] = None,
    feature_columns: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Load processed table, infer or use target, train baseline model, save artifact and insert ml_models row.
    Returns model_id and metrics.
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

    X, y, feature_names = prepare_xy(df, target_column, feature_columns)
    if len(X) < 10:
        raise ValueError("Too few rows after dropping null target")

    problem_type = infer_problem_type(y)
    model = get_baseline_model(problem_type)
    # Only stratify when every class has at least 2 samples (sklearn requirement)
    use_stratify = False
    if problem_type == PROBLEM_CLASSIFICATION:
        from collections import Counter
        counts = Counter(y)
        use_stratify = all(c >= 2 for c in counts.values()) and len(counts) >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y if use_stratify else None,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, problem_type)
    feature_importance = get_feature_importance(model, feature_names)

    model_id = str(uuid.uuid4())[:12]
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
    artifact_path = os.path.join(MODEL_PATH, f"{client_id}_{dataset_id}_{model_id}.pkl")
    with open(artifact_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": feature_names,
            "target_column": target_column,
            "problem_type": problem_type,
        }, f)
    logger.info("Saved model artifact: %s", artifact_path)

    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO ml_models (model_id, client_id, dataset_id, problem_type, target_column, model_type, metrics, feature_importance, artifact_path, status)
                VALUES (:mid, :cid, :did, :ptype, :target, :mtype, CAST(:metrics_json AS jsonb), CAST(:fi_json AS jsonb), :path, 'trained')
            """),
            {
                "mid": model_id,
                "cid": client_id,
                "did": dataset_id,
                "ptype": problem_type,
                "target": target_column,
                "mtype": type(model).__name__,
                "metrics_json": json.dumps(metrics),
                "fi_json": json.dumps(feature_importance),
                "path": artifact_path,
            },
        )
        conn.commit()

    return {"model_id": model_id, "metrics": metrics, "problem_type": problem_type, "target_column": target_column}
