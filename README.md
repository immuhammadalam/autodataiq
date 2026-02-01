# AutoDataIQ

**Upload your data. Get insights and predictions in minutes.**

AutoDataIQ is a generic, client-ready SaaS MVP data platform for small and medium businesses, freelance clients, and non-technical users. It is architected as a **sellable product** that can be customized per client.

---

## Product vision

- **Value proposition:** Upload CSV data → automatic cleaning, validation, analytics, and ML → view results in a dashboard.
- **Target customers:** SMBs, freelance clients, operations/sales/finance teams, non-technical users.
- **Design:** Multi-tenant (client_id, dataset_id), metadata-driven ETL, auto-ML, no business-specific assumptions.

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Upload      │────▶│  ETL         │────▶│  PostgreSQL     │────▶│  Analytics   │
│  (CSV)       │     │  extract     │     │  raw_datasets   │     │  profiler    │
│  client_id   │     │  transform   │     │  processed_*    │     │  aggregations│
│  dataset_id  │     │  load        │     │  analytics_*    │     └──────┬───────┘
└──────────────┘     └──────────────┘     │  ml_models     │            │
                                            └────────┬───────┘            ▼
                                                     │             ┌──────────────┐
                                                     └────────────▶│  ML (auto)   │
                                                                    │  train/predict
                                                                    └──────┬───────┘
                                                                           │
                                                                           ▼
                                                                    ┌──────────────┐
                                                                    │  Dashboard   │
                                                                    │  Streamlit   │
                                                                    └──────────────┘
```

- **Ingestion:** Upload CSV → stored under `data/uploads/{client_id}/{dataset_id}.csv`; schema inferred; metadata in `raw_datasets`.
- **ETL:** Metadata-driven extract (validate row/column consistency), transform (missing values, normalize names, outliers, encode categoricals, optional scaling), load (one table per dataset: `data_{client_id}_{dataset_id}`, metadata in `processed_datasets`). All records tagged with `client_id`.
- **Analytics:** Profiler (summary stats per column), aggregations (trends over time if date column, correlations for numeric). Results in `analytics_results`.
- **ML:** Auto-detect problem type (classification vs regression), infer or configure target, baseline model (RandomForest), metrics and feature importance stored in `ml_models`; artifact per client/dataset.
- **Dashboard:** Client selection, dataset selection, upload, overview, analytics, ML results. B2B-style UI.

---

## Project structure

```
autodataiq/
├── data/
│   ├── uploads/          # Uploaded CSVs per client/dataset
│   ├── raw/
│   └── processed/
├── ingestion/
│   └── upload_handler.py  # Save upload, infer schema, register raw_datasets
├── etl/
│   ├── extract.py        # Read by client/dataset, validate schema
│   ├── transform.py      # Missing values, normalize, outliers, encode, scale
│   ├── load.py           # Store in table per dataset, processed_datasets
│   ├── schema_inference.py
│   └── run_etl.py
├── analytics/
│   ├── profiler.py       # Summary stats → analytics_results
│   └── aggregations.py   # Trends, correlations → analytics_results
├── ml/
│   ├── auto_ml.py        # Infer problem/target, baseline model
│   ├── train.py          # Train, save artifact, ml_models
│   └── predict.py        # Load model, predict on processed data
├── dashboard/
│   └── app.py            # Client/dataset selectors, upload, KPIs, ML
├── config/
│   ├── db.py             # Engine from env
│   └── schema.py         # clients, raw_datasets, processed_datasets, analytics_results, ml_models
├── docker/
│   ├── etl.Dockerfile
│   ├── ml.Dockerfile
│   └── dashboard.Dockerfile
├── docker-compose.yml
├── run_pipeline.py       # ETL → analytics → optional ML train
├── requirements.txt
└── README.md
```

---

## Configuration and extensibility

- **Environment:** All credentials and paths via env (see `.env.example`). No hardcoded secrets.
- **Pipelines:** ETL and ML are configurable by design: feature selection, target column, model overrides, thresholds can be added via config (YAML/DB) or CLI args.
- **Extensibility:** A freelancer can plug in client-specific logic (e.g. custom transform step, custom model) without changing core flow. Tables are generic; client_id/dataset_id tag everything.

---

## How to run

### Prerequisites

- Docker and Docker Compose
- Optional: copy `.env.example` to `.env` and set credentials

### 1. Start PostgreSQL

```bash
cd autodataiq
docker-compose up -d postgres
```

Wait for healthy (or a few seconds).

### 2. Run dashboard (upload and run pipeline from UI)

```bash
docker-compose --profile dashboard up -d dashboard
```

Open **http://localhost:8501**. Create/select client, upload a CSV, then run ETL (and optionally Train model) from the tabs.

### 3. Run pipeline from CLI (e.g. after upload)

```bash
# Ensure data/uploads/<client_id>/<dataset_id>.csv exists (e.g. from dashboard upload)
docker-compose run --rm etl python -m etl.run_etl --client-id CLIENT --dataset-id DATASET
# Then analytics (run from host if analytics is not in ETL container)
pip install -r requirements.txt
python run_pipeline.py --client-id CLIENT --dataset-id DATASET --train-ml
```

### 4. Full stack with Docker

```bash
docker-compose --profile full up --build
```

Starts Postgres, ETL (default client/dataset), ML (default), and dashboard. For real use, upload via dashboard and run ETL/ML for that client/dataset.

---

## Design choices (brief)

- **One table per processed dataset:** Enables direct SQL and analytics on cleaned data; table name `data_{client_id}_{dataset_id}` (sanitized). Alternative (single table + JSONB row) is possible for different scaling.
- **Schema inference at upload:** Enables generic pipelines; no fixed schema. ETL uses inferred date/numeric/categorical for transform and analytics.
- **Auto-ML target inference:** Last numeric or last categorical column as default target; overridable per client/dataset for paid/custom tiers.
- **analytics_results as key-value:** result_type, result_key, result_value (JSONB) allows adding new analytics without schema changes.
- **Client/dataset in dashboard first:** Ensures all flows are multi-tenant from the start.

---

## Future improvements

- **Config file:** YAML/DB for feature lists, model type overrides, thresholds per client or tier.
- **API:** REST or FastAPI for upload and pipeline trigger (headless use).
- **Orchestration:** Airflow/Prefect to schedule ETL and ML per client.
- **Tiers:** Feature flags or plans (e.g. advanced analytics, custom models) for monetization.
- **Audit:** Log all pipeline runs and model versions for compliance.
