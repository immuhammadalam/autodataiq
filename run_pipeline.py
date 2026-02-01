"""
Pipeline runner: for a given client_id and dataset_id, run ETL -> analytics (profiler, aggregations) -> optional ML train.
Invoked from CLI or scheduler. Designed for extensibility (add steps, config overrides).
"""
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.db import get_engine
from config.schema import create_schema
from etl.run_etl import run_etl
from analytics.profiler import run_profiler
from analytics.aggregations import run_aggregations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(client_id: str, dataset_id: str, train_ml: bool = False):
    """Run ETL, then profiler, then aggregations; optionally train ML."""
    create_schema(get_engine())
    run_etl(client_id, dataset_id)
    run_profiler(client_id, dataset_id)
    run_aggregations(client_id, dataset_id)
    if train_ml:
        from ml.train import train
        train(client_id, dataset_id)
    logger.info("Pipeline complete: client=%s dataset=%s", client_id, dataset_id)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--client-id", required=True)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--train-ml", action="store_true", help="Run ML training after analytics")
    args = p.parse_args()
    run_pipeline(args.client_id, args.dataset_id, train_ml=args.train_ml)
