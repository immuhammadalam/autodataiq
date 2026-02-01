"""
Run ETL for a given client_id and dataset_id: extract -> transform -> load.
Invoked from dashboard or CLI. All logic is metadata-driven and client-scoped.
"""
import os
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.db import get_engine
from etl.extract import extract
from etl.transform import transform
from etl.load import load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_etl(client_id: str, dataset_id: str) -> str:
    """Run full ETL for client/dataset. Returns processed table_name."""
    logger.info("Starting ETL: client=%s dataset=%s", client_id, dataset_id)
    df, schema, mismatches = extract(client_id, dataset_id)
    df = transform(df, schema)
    table_name = load(client_id, dataset_id, df, schema)
    logger.info("ETL complete: table=%s", table_name)
    return table_name


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--client-id", required=True)
    p.add_argument("--dataset-id", required=True)
    args = p.parse_args()
    run_etl(args.client_id, args.dataset_id)
