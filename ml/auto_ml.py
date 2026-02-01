"""
Auto-ML: Auto-detect problem type (regression vs classification), choose baseline model.
Generic: no business assumptions. Target column can be configured or inferred (last numeric/categorical).
"""
import logging
from typing import Tuple, Optional, List

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder

# Project root
ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

PROBLEM_CLASSIFICATION = "classification"
PROBLEM_REGRESSION = "regression"
RANDOM_STATE = 42
TEST_SIZE = 0.25


def infer_problem_type(y: pd.Series) -> str:
    """Infer classification vs regression: if target has few unique int/object -> classification, else regression."""
    n_unique = y.nunique()
    if n_unique <= 2 or (n_unique <= 20 and (y.dtype == object or pd.api.types.is_integer_dtype(y))):
        return PROBLEM_CLASSIFICATION
    return PROBLEM_REGRESSION


# Preferred regression targets (sales / units) so we predict meaningful business metrics
PREFERRED_TARGETS = ["revenue", "sales", "quantity", "units_sold", "unit_sales", "total_sales", "amount"]


def infer_target_column(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> Optional[str]:
    """Suggest target: prefer revenue/sales/quantity for regression; else last numeric or categorical. Exclude client_id, dataset_id."""
    exclude = exclude or []
    exclude = set(exclude) | {"client_id", "dataset_id"}
    for preferred in PREFERRED_TARGETS:
        for col in df.columns:
            if col in exclude:
                continue
            if preferred in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
                return col
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if numeric:
        return numeric[-1]
    cat = [c for c in df.columns if df[c].dtype == object or df[c].dtype.name == "category"]
    cat = [c for c in cat if c not in exclude]
    if cat:
        return cat[-1]
    return None


def get_baseline_model(problem_type: str):
    """Return sklearn estimator for problem type. Extensible: can add XGBoost per config."""
    if problem_type == PROBLEM_CLASSIFICATION:
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
    return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)


def prepare_xy(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    drop_na_target: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare X, y; encode categorical target for classification. Returns (X, y, feature_names)."""
    df = df.copy()
    # Infer date column from dataset (see config.date_columns) and add year + month for per-month predictions
    from config.date_columns import DATE_COLUMN_PRIORITY as date_priority
    cols_lower = {c: c.lower().strip() for c in df.columns}
    date_col_found = None
    if "year" not in df.columns:
        for preferred in date_priority:
            for col in df.columns:
                if preferred not in cols_lower.get(col, ""):
                    continue
                try:
                    ser = pd.to_datetime(df[col], errors="coerce")
                    if ser.notna().sum() / max(len(df), 1) > 0.5:
                        years = ser.dt.year
                        if years.notna().any() and years.max() > 2000:
                            df["year"] = years
                            df["month"] = ser.dt.month
                            date_col_found = col
                            break
                except Exception:
                    pass
            if date_col_found is not None:
                break
        if date_col_found is None:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    ser = pd.to_datetime(df[col], errors="coerce")
                    df["year"] = ser.dt.year
                    df["month"] = ser.dt.month
                    break
            else:
                try:
                    for col in df.columns:
                        if df[col].dtype == object or "date" in col.lower():
                            ser = pd.to_datetime(df[col], errors="coerce")
                            if ser.notna().sum() / max(len(df), 1) > 0.5 and ser.dt.year.max() > 2000:
                                df["year"] = ser.dt.year
                                df["month"] = ser.dt.month
                                break
                except Exception:
                    pass
    if "year" in df.columns:
        yr = pd.to_numeric(df["year"], errors="coerce")
        med = yr.median()
        df["year"] = yr.fillna(med if pd.notna(med) else yr.max())
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce").fillna(1).clip(1, 12).astype(int)
    exclude = {target_column, "client_id", "dataset_id"}
    if feature_columns:
        feature_cols = [c for c in feature_columns if c in df.columns and c not in exclude]
    else:
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in exclude][:20]
    X = df[feature_cols].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    y = df[target_column]
    if drop_na_target:
        mask = y.notna()
        X, y = X[mask], y[mask]
    problem = infer_problem_type(y)
    if problem == PROBLEM_CLASSIFICATION:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
    return X, y, feature_cols


def compute_metrics(y_true, y_pred, problem_type: str) -> dict:
    """Compute metrics for classification or regression."""
    if problem_type == PROBLEM_CLASSIFICATION:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_feature_importance(model, feature_names: List[str]) -> dict:
    """Extract feature importance from tree-based model."""
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, [float(x) for x in model.feature_importances_]))
    return {}
