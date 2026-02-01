# How to Run AutoDataIQ

Use either **Option A (Docker)** or **Option B (local Python)**. Paths assume you are in the project root `autodataiq/`.

---

## Option A: Run with Docker

### Step 1: Go to the project folder

```bash
cd /Users/muhammadfakhr/Downloads/ETL/autodataiq
```

### Step 2: Create environment file

```bash
cp .env.example .env
```

(Optional: edit `.env` to change DB password or ports.)

### Step 3: Start PostgreSQL

```bash
docker-compose up -d postgres
```

Wait about 5–10 seconds for Postgres to be ready.

### Step 4: Start the dashboard

```bash
docker-compose --profile dashboard up -d dashboard
```

### Step 5: Open the app

In your browser go to:

**http://localhost:8501**

### Step 6: Use the app

1. **Create a client** (if the list is empty): in the sidebar, type a Client ID (e.g. `client_1`) and click **Create client**. Refresh if needed.
2. **Upload a CSV**: open the **Upload** tab, set Client ID (e.g. `client_1`), choose a CSV file, click upload.
3. **Run ETL**: after upload, click **Run ETL for this dataset** on the Upload tab.
4. **View data**: go to **Overview** and select the client and dataset in the sidebar.
5. **Analytics**: run the pipeline (see Step 7) so profiler/aggregations run, then check the **Analytics** tab.
6. **ML**: in the **ML** tab click **Train model (auto-ML)**. Then use **Run predictions (latest model)**.

### Step 7: Run full pipeline (ETL + analytics + ML) from your machine

After at least one upload, run (replace `client_1` and the dataset id with yours; the dataset id is shown after upload, e.g. `abc12345`):

```bash
cd /Users/muhammadfakhr/Downloads/ETL/autodataiq
pip install -r requirements.txt
export DB_HOST=localhost DB_PORT=5432 DB_USER=autodataiq DB_PASSWORD=autodataiq_secret DB_NAME=autodataiq
python run_pipeline.py --client-id client_1 --dataset-id YOUR_DATASET_ID --train-ml
```

Or run ETL only via Docker:

```bash
docker-compose run --rm etl python -m etl.run_etl --client-id client_1 --dataset-id YOUR_DATASET_ID
```

---

## Option B: Run without Docker (local Python)

### Step 1: Go to the project folder

```bash
cd /Users/muhammadfakhr/Downloads/ETL/autodataiq
```

### Step 2: Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set up PostgreSQL

- Install PostgreSQL 15 locally and create a database named `autodataiq` (or use the name in `.env`).
- Create env file: `cp .env.example .env`
- Edit `.env` and set:
  - `DB_HOST=localhost`
  - `DB_PORT=5432`
  - `DB_USER=autodataiq`
  - `DB_PASSWORD=autodataiq_secret`
  - `DB_NAME=autodataiq`

Or export the same variables in the shell before running commands.

### Step 5: Load env and start the dashboard

```bash
export $(grep -v '^#' .env | xargs)
streamlit run dashboard/app.py
```

### Step 6: Open the app

In your browser go to:

**http://localhost:8501**

### Step 7: Upload and run pipeline

1. In the dashboard: create a client (sidebar), then in **Upload** tab upload a CSV.
2. Note the **dataset_id** shown after upload.
3. In a **new terminal** (same folder, venv active):

```bash
cd /Users/muhammadfakhr/Downloads/ETL/autodataiq
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
python run_pipeline.py --client-id client_1 --dataset-id YOUR_DATASET_ID --train-ml
```

Replace `client_1` and `YOUR_DATASET_ID` with the values from the dashboard.

---

## Quick reference

| Goal                    | Command |
|-------------------------|--------|
| Start Postgres only     | `docker-compose up -d postgres` |
| Start dashboard only    | `docker-compose --profile dashboard up -d dashboard` |
| Run full stack          | `docker-compose --profile full up --build` |
| Run ETL for one dataset | `docker-compose run --rm etl python -m etl.run_etl --client-id CLIENT --dataset-id DATASET` |
| Run pipeline (ETL+analytics+ML) | `python run_pipeline.py --client-id CLIENT --dataset-id DATASET --train-ml` |
| Stop containers         | `docker-compose down` |

---

## Troubleshooting

- **Dashboard says "Database connection failed"**  
  Start Postgres first and wait a few seconds. Check `.env` (or env vars) and that `DB_HOST`/`DB_PORT` match your Postgres (e.g. `localhost:5432` for local).

- **"No processed table" or "No clients yet"**  
  Upload a CSV in the Upload tab and run ETL (button on Upload tab or `run_pipeline.py`).

- **"No trained model"**  
  Run pipeline with `--train-ml` or click **Train model (auto-ML)** in the ML tab.

- **Port 8501 in use**  
  Set `STREAMLIT_PORT=8502` in `.env` (or export it), then open http://localhost:8502.
