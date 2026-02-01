"""
Auto-ML: Auto-detect problem type (regression vs classification), choose baseline model.
Generic: no business assumptions. Target column can be configured or inferred (last numeric/categorical).
Includes ML readiness gate and model selection ladder.
"""
import logging
from typing import Tuple, Optional, List, Dict, Any

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
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
PROBLEM_FORECASTING = "forecasting"
PROBLEM_NOT_RECOMMENDED = "not_recommended"
RANDOM_STATE = 42
TEST_SIZE = 0.25
MIN_ROWS_READINESS = 300
MIN_ROWS_TRAIN = 20
LEAKAGE_PATTERNS = ("id", "uuid", "_key", "key_")


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


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Return first column that looks like a date/time with enough valid values."""
    from config.date_columns import DATE_COLUMN_PRIORITY
    cols_lower = {c: c.lower().strip() for c in df.columns}
    for preferred in DATE_COLUMN_PRIORITY:
        for col in df.columns:
            if preferred not in cols_lower.get(col, ""):
                continue
            try:
                ser = pd.to_datetime(df[col], errors="coerce")
                if ser.notna().sum() / max(len(df), 1) > 0.5 and ser.dt.year.max() > 2000:
                    return col
            except Exception:
                pass
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        try:
            ser = pd.to_datetime(df[col], errors="coerce")
            if ser.notna().sum() / max(len(df), 1) > 0.5 and ser.dt.year.max() > 2000:
                return col
        except Exception:
            pass
    return None


def _is_likely_leakage(col_name: str, df: pd.DataFrame, target_column: str) -> bool:
    """True if column looks like ID/uuid or is a duplicate of target."""
    col_lower = col_name.lower()
    if any(p in col_lower for p in LEAKAGE_PATTERNS):
        return True
    if col_name == target_column:
        return False
    ser = df[col_name]
    if ser.nunique() == len(ser.dropna()) and len(ser.dropna()) > len(df) * 0.9:
        return True  # near-unique identifier
    return False


def assess_ml_readiness(df: pd.DataFrame, inferred_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    ML readiness / decision gate. Checks minimum rows, target variance, date column, leakage.
    Returns: ml_allowed, problem_type (forecasting|regression|classification|not_recommended),
             reason, confidence_level (low|medium|high). If not recommended, store in ml_models and skip training.
    """
    inferred_schema = inferred_schema or {}
    target_column = inferred_schema.get("target_column") or infer_target_column(df)
    n = len(df)
    result = {
        "ml_allowed": False,
        "problem_type": PROBLEM_NOT_RECOMMENDED,
        "reason": "",
        "confidence_level": "low",
    }

    if n < MIN_ROWS_READINESS:
        result["reason"] = f"Not enough data for reliable ML (need at least {MIN_ROWS_READINESS} rows; you have {n})."
        return result

    if not target_column or target_column not in df.columns:
        result["reason"] = "No suitable target column found to predict."
        return result

    y = df[target_column].dropna()
    if len(y) < MIN_ROWS_READINESS * 0.8:
        result["reason"] = "Too many missing values in the target column."
        return result

    # Target variance: for numeric, require non-constant; for categorical, require at least 2 classes
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() < 2 or y.std() == 0:
            result["reason"] = "Target column has no variation (constant or near-constant)."
            return result
    else:
        if y.nunique() < 2:
            result["reason"] = "Target column has only one category."
            return result

    # Leakage: exclude ID-like columns from features; if target is itself leakage-like, warn
    leakage_cols = [c for c in df.columns if _is_likely_leakage(c, df, target_column)]
    if target_column in leakage_cols:
        result["reason"] = "Selected target looks like an identifier (no predictable pattern)."
        return result

    date_column = inferred_schema.get("date_column") or _detect_date_column(df)
    is_numeric_target = pd.api.types.is_numeric_dtype(df[target_column])
    n_unique_target = df[target_column].nunique()

    # Decide problem type
    if date_column and is_numeric_target:
        result["problem_type"] = PROBLEM_FORECASTING
        result["ml_allowed"] = True
        result["reason"] = "Date column and numeric target detected; using forecasting."
        result["confidence_level"] = "high" if n >= 500 else "medium"
    elif is_numeric_target:
        result["problem_type"] = PROBLEM_REGRESSION
        result["ml_allowed"] = True
        result["reason"] = "Numeric target; using regression."
        result["confidence_level"] = "high" if n >= 500 else "medium"
    elif n_unique_target <= 2 or (n_unique_target <= 20 and n_unique_target >= 2):
        result["problem_type"] = PROBLEM_CLASSIFICATION
        result["ml_allowed"] = True
        result["reason"] = "Categorical target with few classes; using classification."
        result["confidence_level"] = "high" if n >= 500 and n_unique_target >= 2 else "medium"
    else:
        result["problem_type"] = PROBLEM_CLASSIFICATION
        result["ml_allowed"] = True
        result["reason"] = "Categorical target; using classification."
        result["confidence_level"] = "low"

    result["date_column"] = date_column
    result["target_column"] = target_column
    result["row_count"] = n
    return result


def select_model(problem_type: str, data_size: int) -> Tuple[Any, str, str]:
    """
    Model selection ladder: prefer forecasting/linear for small data, XGBoost/LightGBM for tabular, fallback RandomForest.
    Returns (model, model_name, reason).
    """
    # Try optional dependencies (must run in Docker with optional deps)
    xgb_reg = xgb_clf = lgb_reg = lgb_clf = None
    try:
        import xgboost as xgb
        xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_STATE)
        xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE)
    except ImportError:
        pass
    try:
        import lightgbm as lgb
        lgb_reg = lgb.LGBMRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, verbosity=-1)
        lgb_clf = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, verbosity=-1)
    except ImportError:
        pass

    small_data = data_size < 500
    if problem_type == PROBLEM_FORECASTING:
        if small_data:
            model = LinearRegression()
            return model, "LinearRegression", "Small dataset with time; using simple trend model for stability."
        if xgb_reg is not None:
            return xgb_reg, "XGBRegressor", "Time-based numeric target; using XGBoost for robust forecasting."
        if lgb_reg is not None:
            return lgb_reg, "LGBMRegressor", "Time-based numeric target; using LightGBM for forecasting."
        return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE), "RandomForestRegressor", "Time-based target; XGBoost/LightGBM not available; using Random Forest."

    if problem_type == PROBLEM_REGRESSION:
        if small_data:
            model = LinearRegression()
            return model, "LinearRegression", "Small dataset; using linear model to avoid overfitting."
        if xgb_reg is not None:
            return xgb_reg, "XGBRegressor", "Tabular regression; using XGBoost for accuracy."
        if lgb_reg is not None:
            return lgb_reg, "LGBMRegressor", "Tabular regression; using LightGBM."
        return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE), "RandomForestRegressor", "XGBoost/LightGBM not available; using Random Forest."

    # Classification
    if small_data:
        model = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
        return model, "LogisticRegression", "Small dataset; using logistic regression for interpretability."
    if xgb_clf is not None:
        return xgb_clf, "XGBClassifier", "Tabular classification; using XGBoost."
    if lgb_clf is not None:
        return lgb_clf, "LGBMClassifier", "Tabular classification; using LightGBM."
    return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE), "RandomForestClassifier", "XGBoost/LightGBM not available; using Random Forest."


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


def _safe_mape(y_true, y_pred: np.ndarray) -> float:
    """MAPE avoiding division by zero; clip pred to avoid zeros."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_t) & np.isfinite(y_p) & (np.abs(y_t) > 1e-10)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100.0)


def compute_metrics(y_true, y_pred, problem_type: str) -> dict:
    """Compute metrics for classification or regression. Regression includes RMSE and MAPE."""
    if problem_type == PROBLEM_CLASSIFICATION:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = _safe_mape(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }


def get_feature_importance(model, feature_names: List[str]) -> dict:
    """Extract feature importance from tree-based model."""
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, [float(x) for x in model.feature_importances_]))
    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        if len(coef) == len(feature_names):
            return dict(zip(feature_names, [float(x) for x in coef]))
    return {}


def build_explainability(feature_importance: dict, top_n: int = 10) -> dict:
    """
    Summarize for non-technical users: top positive drivers, top negative drivers.
    Stores as JSON for ml_models.explainability. Avoids raw coefficients where possible.
    """
    if not feature_importance:
        return {"top_positive_drivers": [], "top_negative_drivers": [], "summary": "No driver breakdown available for this model."}
    items = [(k, float(v)) for k, v in feature_importance.items()]
    items.sort(key=lambda x: -x[1])
    top_positive = [{"feature": k, "effect": "positive", "importance": round(v, 4)} for k, v in items[:top_n] if v > 0]
    items.sort(key=lambda x: x[1])
    top_negative = [{"feature": k, "effect": "negative", "importance": round(abs(v), 4)} for k, v in items[:top_n] if v < 0]
    summary = f"Top factors: {len(top_positive)} positive, {len(top_negative)} negative."
    return {
        "top_positive_drivers": top_positive[:5],
        "top_negative_drivers": top_negative[:5],
        "summary": summary,
    }
