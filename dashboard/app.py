"""
AutoDataIQ Dashboard: Upload → ETL → Overview → Analytics → ML.
Single flow: no client/dataset selectors; uses one default client and the latest uploaded dataset.
"""
import os
import sys
import json
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.db import get_engine
from config.schema import create_schema, ensure_client
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Single default client for simplified flow
DEFAULT_CLIENT_ID = "default"


def get_engine_safe():
    try:
        return get_engine()
    except Exception as e:
        st.error("Database connection failed. Check configuration.")
        logger.exception(str(e))
        return None


def get_latest_dataset(engine, client_id: str):
    """Return the most recently uploaded dataset for the client, or None."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT dataset_id, name, row_count FROM raw_datasets WHERE client_id = :cid ORDER BY created_at DESC LIMIT 1"),
            {"cid": client_id},
        ).fetchone()
    return {"dataset_id": row[0], "name": row[1] or row[0], "row_count": row[2]} if row else None


def get_analytics_results(engine, client_id: str, dataset_id: str):
    """Load analytics_results for client/dataset (profiler, aggregations)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT result_type, result_key, result_value FROM analytics_results WHERE client_id = :cid AND dataset_id = :did"),
            {"cid": client_id, "did": dataset_id},
        ).fetchall()
    return {f"{r[0]}_{r[1]}": r[2] for r in rows}


def get_ml_models(engine, client_id: str, dataset_id: str):
    """Load ml_models rows for client/dataset (including feature_importance, explainability, reason, prediction_horizon)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT model_id, problem_type, target_column, model_type, metrics, feature_importance,
                       created_at, reason, prediction_horizon, confidence_level, explainability
                FROM ml_models WHERE client_id = :cid AND dataset_id = :did AND status = 'trained'
                ORDER BY created_at DESC
            """),
            {"cid": client_id, "did": dataset_id},
        ).fetchall()
    return [
        {
            "model_id": r[0], "problem_type": r[1], "target_column": r[2], "model_type": r[3],
            "metrics": r[4], "feature_importance": r[5], "created_at": str(r[6]),
            "reason": r[7], "prediction_horizon": r[8], "confidence_level": r[9], "explainability": r[10],
        }
        for r in rows
    ]


# ----- ML tab: dataset-agnostic, schema-aware -----
def _friendly_label(col: str) -> str:
    """Turn technical column names into human-readable labels. No business assumptions."""
    if not col:
        return col
    col_lower = col.lower().replace(" ", "_")
    if "encoded" in col_lower:
        return col.replace("_", " ").title() + " (encoded)"
    if col_lower.startswith("prediction_"):
        return "Predicted " + col.replace("prediction_", "").replace("_", " ").title()
    return col.replace("_", " ").title()


def _get_processed_table_name(engine, client_id: str, dataset_id: str):
    """Return processed table name for client/dataset, or None."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT table_name FROM processed_datasets WHERE client_id = :cid AND dataset_id = :did"),
            {"cid": client_id, "did": dataset_id},
        ).fetchone()
    return row[0] if row else None


def get_dataset_schema_analysis(engine, client_id: str, dataset_id: str) -> dict:
    """
    Schema-aware scan of the dataset. No assumptions about column names.
    Returns: columns (with type, missing %, cardinality, flags), row_count,
    regression_candidates, classification_candidates, time_columns, valid_target_columns.
    """
    table_name = _get_processed_table_name(engine, client_id, dataset_id)
    if not table_name:
        return {}
    try:
        df = pd.read_sql(f'SELECT * FROM "{table_name}" LIMIT 50000', engine)
    except Exception as e:
        logger.warning("Schema analysis failed: %s", e)
        return {}
    if df.empty:
        return {}
    n = len(df)
    columns = []
    regression_candidates = []
    classification_candidates = []
    time_columns = []
    valid_target_columns = []
    id_like_patterns = ("id", "uuid", "_key", "key_")
    for col in df.columns:
        ser = df[col]
        missing = ser.isna().sum()
        missing_pct = round(100 * missing / n, 1) if n else 0
        cardinality = int(ser.nunique())
        is_fully_null = missing == n
        is_constant = cardinality <= 1 and not is_fully_null
        col_lower = col.lower()
        is_id_like = any(p in col_lower for p in id_like_patterns) and (cardinality == n or ser.dtype in ("int64", "object"))
        is_numeric = pd.api.types.is_numeric_dtype(ser)
        is_datetime = pd.api.types.is_datetime64_any_dtype(ser)
        is_time_candidate = is_datetime
        if not is_time_candidate and ser.dtype == object:
            try:
                parsed = pd.to_datetime(ser, errors="coerce")
                if parsed.notna().sum() / max(n, 1) > 0.5:
                    is_time_candidate = True
            except Exception:
                pass
        if is_time_candidate:
            time_columns.append(col)
        is_regression_candidate = is_numeric and not is_id_like and not is_constant and not is_fully_null
        is_classification_candidate = (not is_numeric or cardinality <= 50) and not is_id_like and not is_constant and not is_fully_null
        if is_numeric:
            type_label = "numeric"
        elif is_datetime:
            type_label = "datetime"
        else:
            type_label = "categorical" if cardinality < n else "text"
        if is_regression_candidate:
            regression_candidates.append(col)
        if is_classification_candidate and not (is_numeric and cardinality > 50):
            classification_candidates.append(col)
        can_be_target = not is_id_like and not is_constant and not is_fully_null
        if can_be_target:
            valid_target_columns.append(col)
        columns.append({
            "name": col,
            "data_type": type_label,
            "missing_pct": missing_pct,
            "cardinality": cardinality,
            "is_id_like": is_id_like,
            "is_constant": is_constant,
            "is_fully_null": is_fully_null,
            "is_regression_candidate": is_regression_candidate,
            "is_classification_candidate": is_classification_candidate,
            "is_time_candidate": is_time_candidate,
        })
    return {
        "columns": columns,
        "row_count": n,
        "regression_candidates": regression_candidates,
        "classification_candidates": classification_candidates,
        "time_columns": time_columns,
        "valid_target_columns": valid_target_columns,
        "table_name": table_name,
    }


def _get_ml_data_understanding(engine, client_id: str, dataset_id: str) -> dict:
    """What the system understood about the data: target, problem type, time range, rows, missing, feature types."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT table_name FROM processed_datasets WHERE client_id = :cid AND dataset_id = :did"),
                {"cid": client_id, "did": dataset_id},
            ).fetchone()
        if not row:
            return {}
        df = pd.read_sql(f'SELECT * FROM "{row[0]}" LIMIT 50000', engine)
        if df.empty:
            return {}
        from ml.auto_ml import infer_target_column, infer_problem_type
        from ml.auto_ml import prepare_xy
        target = infer_target_column(df)
        if not target:
            return {"row_count": len(df), "target_variable": None, "problem_type": None}
        X, y, feature_names = prepare_xy(df, target, drop_na_target=True)
        problem_type = infer_problem_type(y)
        numeric = list(df.select_dtypes(include=["number"]).columns)
        cat = list(df.select_dtypes(include=["object", "category"]).columns)
        time_col = None
        time_range = None
        for c in ["order_date", "date", "transaction_date", "created_at"]:
            if c in df.columns:
                try:
                    ser = pd.to_datetime(df[c], errors="coerce")
                    if ser.notna().sum() / max(len(df), 1) > 0.5:
                        time_col = c
                        valid = ser.dropna()
                        if len(valid):
                            time_range = f"{valid.min().strftime('%b %Y')} – {valid.max().strftime('%b %Y')}"
                        break
                except Exception:
                    pass
        missing_total = int(df.isna().sum().sum())
        return {
            "target_variable": target,
            "target_friendly": _friendly_label(target),
            "problem_type": problem_type,
            "row_count": len(df),
            "rows_analyzed": len(X),
            "time_range": time_range or "Not detected",
            "numeric_features": len([c for c in feature_names if c in numeric]),
            "categorical_encoded": len([c for c in feature_names if c not in numeric]),
            "missing_values_handled": missing_total,
        }
    except Exception as e:
        logger.warning("Data understanding failed: %s", e)
        return {}


def _prediction_variance_check(pred_series) -> tuple:
    """Return (is_flat_or_unstable, message). Used for failure safety."""
    if pred_series is None or pred_series.empty:
        return False, ""
    try:
        pred_numeric = pd.to_numeric(pred_series, errors="coerce").dropna()
        if len(pred_numeric) < 2:
            return False, ""
        std = pred_numeric.std()
        if std == 0 or (pred_numeric.max() - pred_numeric.min()) < 1e-6:
            return True, "Your data does not show strong variation for this prediction."
        if std < pred_numeric.mean() * 0.01:
            return True, "Predictions show very little variation; the model may not have found strong patterns."
    except Exception:
        return False, ""
    return False, ""


def _metrics_plain_language(metrics: dict, problem_type: str) -> list:
    """Translate ML metrics into plain-English bullet points."""
    if not metrics:
        return []
    lines = []
    if problem_type == "classification":
        acc = metrics.get("accuracy")
        if acc is not None:
            pct = round(acc * 100)
            lines.append(f"**Correct predictions:** {pct}% of outcomes were predicted correctly.")
        f1 = metrics.get("f1")
        if f1 is not None:
            lines.append(f"**Balance between precision and recall:** {round(f1, 2)} (1.0 = best).")
        prec = metrics.get("precision")
        rec = metrics.get("recall")
        if prec is not None:
            lines.append(f"**Precision:** When the model says “yes,” it’s right {round(prec * 100)}% of the time.")
        if rec is not None:
            lines.append(f"**Recall:** The model finds {round(rec * 100)}% of actual “yes” cases.")
    else:
        mse = metrics.get("mse")
        if mse is not None:
            import math
            rmse = math.sqrt(mse)
            lines.append(f"**Average prediction error (RMSE):** {rmse:,.2f} — lower is better.")
        mae = metrics.get("mae")
        if mae is not None:
            lines.append(f"**Average absolute error:** {mae:,.2f}.")
        r2 = metrics.get("r2")
        if r2 is not None:
            lines.append(f"**How much variation is explained:** {round(r2 * 100)}% of the variation in the target is explained by the model.")
    return lines


def _performance_verdict(metrics: dict, problem_type: str) -> tuple:
    """Return (verdict_text, severity: good/acceptable/weak)."""
    if not metrics:
        return ("No metrics available.", "acceptable")
    if problem_type == "classification":
        acc = metrics.get("accuracy")
        if acc is None:
            return ("Performance not yet measured.", "acceptable")
        if acc >= 0.75:
            return ("Model performance is **good** for decision support.", "good")
        if acc >= 0.5:
            return ("Model performance is **acceptable** for early forecasting or exploration.", "acceptable")
        return ("Model performance is **weak**; consider more data or different features.", "weak")
    else:
        r2 = metrics.get("r2")
        if r2 is None:
            return ("Performance not yet measured.", "acceptable")
        if r2 >= 0.6:
            return ("Model performance is **good** for planning and estimates.", "good")
        if r2 >= 0.3:
            return ("Model performance is **acceptable** for rough estimates.", "acceptable")
        return ("Model performance is **weak**; predictions are rough. Consider more or better data.", "weak")


def _model_reason(problem_type: str, model_type: str) -> str:
    """Plain-English explanation of why this model was chosen."""
    if "RandomForest" in str(model_type) or "Random Forest" in str(model_type):
        if problem_type == "classification":
            return "We selected **Random Forest** because your data has mixed feature types and can have non-linear patterns; this model handles both well and is robust to noise."
        return "We selected **Random Forest** because your target is numeric and your data can have non-linear relationships; this model captures those without assuming a simple formula."
    if "Logistic" in str(model_type):
        return "We selected **Logistic Regression** for a clear, interpretable classification model that works well when the relationship is roughly linear."
    if "Linear" in str(model_type):
        return "We selected **Linear Regression** for a simple, interpretable model that fits numeric targets when relationships are roughly linear."
    return f"We selected **{model_type}** as a suitable model for your data type and target."


# Human-readable definitions for common column names (Overview tab)
COLUMN_DEFINITIONS = {
    "revenue": "Total revenue (e.g. price × quantity).",
    "sales": "Sales amount or total sales value.",
    "quantity": "Number of units sold or ordered.",
    "units_sold": "Number of units sold.",
    "order_date": "Date of the order or transaction.",
    "order_id": "Unique identifier for the order.",
    "transaction_date": "Date of the transaction.",
    "date": "Date or timestamp.",
    "unit_price": "Price per unit.",
    "price": "Price or cost.",
    "product": "Product name or identifier.",
    "product_id": "Unique identifier for the product.",
    "customer": "Customer name or identifier.",
    "customer_id": "Unique identifier for the customer.",
    "category": "Category or classification.",
    "sku": "Stock Keeping Unit (product code).",
    "client_id": "Client or tenant identifier.",
    "dataset_id": "Dataset identifier.",
    "year": "Year (derived from date).",
    "month": "Month (1–12, derived from date).",
}


def get_column_definitions(df: pd.DataFrame) -> pd.DataFrame:
    """Build a table of column name, data type, and definition for the Overview tab."""
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "int" in dtype or "float" in dtype:
            type_label = "Numeric"
        elif "datetime" in dtype or "date" in dtype:
            type_label = "Date/time"
        else:
            type_label = "Text or category"
        col_lower = col.lower().replace(" ", "_")
        definition = None
        for key, desc in COLUMN_DEFINITIONS.items():
            if key in col_lower or key.replace("_", " ") in col_lower:
                definition = desc
                break
        if definition is None and "encoded" in col_lower:
            definition = "Numeric encoding of a category (for modeling)."
        if definition is None:
            definition = f"{type_label} column."
        rows.append({"Column": col, "Data type": type_label, "Definition": definition})
    return pd.DataFrame(rows)


def _ensure_dict(val):
    """Result_value from DB may be dict or JSON string."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val) if isinstance(val, str) else {}
    except Exception:
        return {}


def _render_analytics_insights(results: dict):
    """Render analytics as visual insights: tables, line charts, heatmap, pie charts."""
    # ----- 1. Data overview (profiler summary) -----
    profiler = _ensure_dict(results.get("profiler_summary_stats"))
    if profiler:
        st.markdown("### Data overview")
        row_count = profiler.get("row_count", 0)
        col_count = profiler.get("column_count", 0)
        c1, c2 = st.columns(2)
        c1.metric("Total rows", f"{row_count:,}")
        c2.metric("Total columns", col_count)
        columns_info = profiler.get("columns", {})
        if columns_info:
            rows = []
            for col_name, info in columns_info.items():
                if not isinstance(info, dict):
                    continue
                r = {"Column": col_name, "Non-null": info.get("count", ""), "Missing": info.get("nulls", ""), "Type": str(info.get("dtype", ""))}
                if "min" in info and info["min"] is not None:
                    r["Min"] = round(info["min"], 2) if isinstance(info["min"], (int, float)) else info["min"]
                if "max" in info and info["max"] is not None:
                    r["Max"] = round(info["max"], 2) if isinstance(info["max"], (int, float)) else info["max"]
                if "mean" in info and info["mean"] is not None:
                    r["Average"] = round(info["mean"], 2)
                if "n_unique" in info:
                    r["Unique values"] = info["n_unique"]
                rows.append(r)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("---")

    # ----- 2. Top values (pie charts from profiler categoricals) -----
    if profiler:
        columns_info = profiler.get("columns", {})
        pie_cols = [
            (col_name, info) for col_name, info in (columns_info or {}).items()
            if isinstance(info, dict) and "top_values" in info and info.get("top_values")
        ]
        if pie_cols:
            st.markdown("### Top values by category")
            for i in range(0, min(len(pie_cols), 6), 2):
                pair = pie_cols[i : i + 2]
                cols = st.columns(2)
                for j, (col_name, info) in enumerate(pair):
                    top = info.get("top_values")
                    if isinstance(top, dict):
                        names = list(top.keys())[:8]
                        values = list(top.values())[:8]
                    elif isinstance(top, list):
                        names = [str(x[0]) for x in top[:8]]
                        values = [x[1] for x in top[:8]]
                    else:
                        continue
                    with cols[j]:
                        fig = px.pie(values=values, names=names, title=f"Top values: {col_name}", hole=0.4)
                        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), showlegend=True, legend=dict(orientation="h"))
                        st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")

    # ----- 3. Trends over time -----
    trends = _ensure_dict(results.get("aggregations_trends"))
    if trends:
        st.markdown("### Trends over time")
        # Short explanations by metric name for non-technical users
        trend_explanations = {
            "revenue": "Total revenue by month. The line shows how sales income changes over time; use it to spot seasonal patterns or growth.",
            "quantity": "Total quantity (units) by period. Shows how volume sold changes over time.",
            "unit_price": "Average or total unit price by period. Helps see pricing trends.",
        }
        for col_name, series in trends.items():
            if not isinstance(series, list) or not series:
                continue
            try:
                df_t = pd.DataFrame(series)
                if "period" in df_t.columns and "value" in df_t.columns:
                    df_t["value"] = pd.to_numeric(df_t["value"], errors="coerce").fillna(0)
                    df_t = df_t.sort_values("period").reset_index(drop=True)
                    periods = df_t["period"].astype(str).tolist()
                    values = df_t["value"].tolist()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=periods,
                        y=values,
                        mode="lines+markers",
                        line=dict(width=3, color="#2563eb"),
                        marker=dict(size=8),
                        name=col_name,
                    ))
                    fig.update_layout(
                        title=col_name,
                        xaxis_title="Period",
                        yaxis_title="Value",
                        height=360,
                        showlegend=False,
                    )
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    explanation = trend_explanations.get(col_name.lower(), f"**{col_name}** over time. The line connects each period so you can see the trend; dots are the actual values.")
                    st.caption(explanation)
            except Exception:
                pass
        st.markdown("---")

    # ----- 4. Correlations (heatmap + table) -----
    corr_data = _ensure_dict(results.get("aggregations_correlations"))
    if corr_data:
        st.markdown("### Correlations between numeric columns")
        matrix = corr_data.get("matrix")
        if isinstance(matrix, dict) and matrix:
            try:
                corr_df = pd.DataFrame(matrix).round(2)
                fig = go.Figure(data=go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    colorscale="RdBu",
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=corr_df.values,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                ))
                fig.update_layout(title="Correlation heatmap", xaxis_title="", yaxis_title="", height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Values from -1 (negative) to +1 (positive). Stronger color = stronger relationship.")
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
            except Exception:
                pass
        pairs = corr_data.get("pairs", [])
        if pairs and not matrix:
            rows = [{"Column A": p.get("col1"), "Column B": p.get("col2"), "Correlation": round(p.get("correlation", 0), 2)} for p in pairs[:20]]
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="AutoDataIQ",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # B2B header
    st.markdown("""
        <style>
        .main-header { font-size: 1.8rem; font-weight: 600; color: #1e3a5f; margin-bottom: 0.5rem; }
        .sub-header { color: #5a6c7d; font-size: 0.95rem; margin-bottom: 1.5rem; }
        .metric-card { background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; }
        </style>
        <div class="main-header">AutoDataIQ</div>
        <div class="sub-header">Upload your data. Get insights and predictions in minutes.</div>
    """, unsafe_allow_html=True)

    engine = get_engine_safe()
    if not engine:
        st.stop()

    create_schema(engine)
    ensure_client(engine, DEFAULT_CLIENT_ID)

    # Single flow: one client, latest dataset
    client_id = DEFAULT_CLIENT_ID
    latest = get_latest_dataset(engine, client_id)
    dataset_id = latest["dataset_id"] if latest else None

    # Tabs: Upload | Overview | Analytics | ML
    tab_upload, tab_overview, tab_analytics, tab_ml = st.tabs(["Upload", "Overview", "Analytics", "ML"])

    with tab_upload:
        st.subheader("Upload data")
        use_sample = st.checkbox(
            "Limit to first 50,000 rows (recommended for large files to avoid timeout)",
            value=False,
            key="upload_sample",
        )
        max_rows = 50_000 if use_sample else None
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="uploader")
        if uploaded:
            upload_key = (uploaded.name, getattr(uploaded, "size", 0) or 0)
            if st.session_state.get("processed_upload_key") == upload_key:
                st.success("This file was already processed. Check **Overview**, **Analytics**, and **ML** tabs.")
                st.json(st.session_state.get("last_upload_schema", {}))
            else:
                file_size_mb = (getattr(uploaded, "size", 0) or 0) / (1024 * 1024)
                if file_size_mb > 10 and not use_sample:
                    st.warning(f"File is {file_size_mb:.1f} MB. Consider enabling \"Limit to first 50,000 rows\" for faster loading.")
                status = st.empty()
                try:
                    with st.spinner("Reading CSV..."):
                        status.info("Reading CSV...")
                        from ingestion.upload_handler import save_upload
                        if hasattr(uploaded, "seek"):
                            uploaded.seek(0)
                        status.info("Inferring schema and saving...")
                        did, schema = save_upload(DEFAULT_CLIENT_ID, uploaded, name=uploaded.name, max_rows=max_rows)
                    status.info("Running ETL...")
                    from etl.run_etl import run_etl
                    run_etl(DEFAULT_CLIENT_ID, did)
                    status.info("Running analytics (profiler & aggregations)...")
                    from analytics.profiler import run_profiler
                    from analytics.aggregations import run_aggregations
                    run_profiler(DEFAULT_CLIENT_ID, did)
                    run_aggregations(DEFAULT_CLIENT_ID, did)
                    status.empty()
                    st.success("Done. Upload → ETL → Analytics complete. Go to the **ML** tab, choose what to predict, then train a model.")
                    st.json(schema.get("column_types", {}))
                    st.session_state["processed_upload_key"] = upload_key
                    st.session_state["last_upload_schema"] = schema.get("column_types", {})
                    st.rerun()
                except Exception as e:
                    status.empty()
                    st.error("Upload failed. See details below.")
                    st.exception(e)

    with tab_overview:
        st.subheader("Overview")
        if not dataset_id:
            st.info("Upload a CSV in the **Upload** tab to run ETL and see data here.")
        else:
            try:
                row = None
                with engine.connect() as conn:
                    row = conn.execute(
                        text("SELECT table_name FROM processed_datasets WHERE client_id = :cid AND dataset_id = :did"),
                        {"cid": client_id, "did": dataset_id},
                    ).fetchone()
                if row:
                    df = pd.read_sql(f'SELECT * FROM "{row[0]}" LIMIT 1000', engine)
                    st.metric("Rows (sample)", len(df))
                    st.markdown("#### Column definitions")
                    defs_df = get_column_definitions(df)
                    st.dataframe(defs_df, use_container_width=True, hide_index=True)
                    st.markdown("#### Data preview")
                    st.dataframe(df.head(100), use_container_width=True)
                else:
                    st.warning("No processed data for this dataset. Upload a CSV to run ETL automatically.")
            except Exception as e:
                st.error(str(e))

    with tab_analytics:
        st.subheader("Analytics")
        if not dataset_id:
            st.info("Upload a CSV in the **Upload** tab to run ETL and see analytics here.")
        else:
            results = get_analytics_results(engine, client_id, dataset_id)
            if not results:
                st.info("No analytics yet for this dataset. Upload a CSV to run ETL and analytics automatically, or run analytics now.")
                if st.button("Run analytics now", key="run_analytics_btn"):
                    with st.spinner("Running profiler and aggregations..."):
                        try:
                            from analytics.profiler import run_profiler
                            from analytics.aggregations import run_aggregations
                            run_profiler(client_id, dataset_id)
                            run_aggregations(client_id, dataset_id)
                            st.success("Analytics complete. Refreshing...")
                            st.rerun()
                        except Exception as ex:
                            st.error(str(ex))
                            st.exception(ex)
            else:
                _render_analytics_insights(results)

    with tab_ml:
        st.subheader("Predictions")
        if not dataset_id:
            st.info("Upload a CSV in the **Upload** tab first, then choose what to predict here.")
        else:
            schema = get_dataset_schema_analysis(engine, client_id, dataset_id)
            valid_targets = schema.get("valid_target_columns") or []
            if not valid_targets:
                st.info("No columns available to predict. Add data with at least one non-empty, non-constant column.")
            else:
                if "ml_target_column" not in st.session_state or st.session_state["ml_target_column"] not in valid_targets:
                    st.session_state["ml_target_column"] = valid_targets[0]
                idx = valid_targets.index(st.session_state["ml_target_column"]) if st.session_state["ml_target_column"] in valid_targets else 0
                selected_target = st.selectbox(
                    "What do you want to predict?",
                    options=valid_targets,
                    index=idx,
                    format_func=lambda x: _friendly_label(x),
                    key="ml_target_select",
                )
                st.session_state["ml_target_column"] = selected_target

                models = get_ml_models(engine, client_id, dataset_id)
                model_for_target = next((m for m in models if m.get("target_column") == selected_target), None)
                if not model_for_target:
                    if st.button("Get predictions", key="ml_get_predictions_btn"):
                        try:
                            from ml.train import train
                            with st.spinner("Preparing predictions…"):
                                train(client_id, dataset_id, target_column=selected_target)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                else:
                    target_label = _friendly_label(selected_target)
                    try:
                        from ml.predict import get_next_year_predictions_table
                        pred_table = get_next_year_predictions_table(client_id, dataset_id)
                        pred_col = f"prediction_{selected_target}"
                        if not pred_table.empty and pred_col in pred_table.columns:
                            display_df = pred_table[["prediction_date", pred_col]].copy()
                            display_df.columns = ["Date", f"Predicted {target_label}"]
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                        elif not pred_table.empty:
                            display_df = pred_table.copy()
                            display_df.columns = [_friendly_label(c) for c in display_df.columns]
                            st.dataframe(display_df, use_container_width=True, hide_index=True)
                        else:
                            from ml.predict import predict
                            pred_df = predict(client_id, dataset_id, limit=1000)
                            if "prediction" in pred_df.columns:
                                out = pred_df[["prediction"]].copy()
                                out.columns = [f"Predicted {target_label}"]
                                st.dataframe(out, use_container_width=True, hide_index=True)
                            else:
                                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        logger.warning("Predictions failed: %s", e)
                        st.error("Could not load predictions. Try **Get predictions** again.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**AutoDataIQ**")

if __name__ == "__main__":
    main()
