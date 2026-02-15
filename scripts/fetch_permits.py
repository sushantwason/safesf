"""
Fetch Building Permits from DataSF (i98e-djp9) for 311 correlation.

Output: data/events/permits.parquet (date, event_type='construction', name).
Merge into events_by_date by running fetch_events.py after this (fetch_events merges permits if present).

Run periodically.
"""
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from sodapy import Socrata
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DOMAIN = "data.sfgov.org"
PERMITS_DATASET = "i98e-djp9"


def get_client():
    token = os.getenv("SF_OPEN_DATA_APP_TOKEN")
    return Socrata(
        DOMAIN,
        app_token=token if token and token != "YOUR_TOKEN_HERE" else None,
        timeout=60,
    )


def fetch_permits(client, start_date: str, end_date: str, limit: int = 50000) -> pd.DataFrame:
    """Fetch Building Permits. Use permit_creation_date or filing_date."""
    logger.info("Fetching Building Permits...")
    date_col = None
    for col in ["permit_creation_date", "filing_date", "issued_date", "Permit Creation Date"]:
        try:
            results = client.get(
                PERMITS_DATASET,
                where=f"{col} >= '{start_date}' AND {col} <= '{end_date}'",
                limit=limit,
            )
            date_col = col
            break
        except Exception:
            continue

    if not date_col:
        try:
            results = client.get(PERMITS_DATASET, limit=min(limit, 10000))
            date_col = "permit_creation_date" if "permit_creation_date" in (results[0] or {}).keys() else list((results[0] or {}).keys())[0]
        except Exception as e:
            logger.error("Could not fetch permits: %s", e)
            return pd.DataFrame()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(results)
    if date_col not in df.columns and len(df) > 0:
        for c in ["permit_creation_date", "filing_date", "issued_date", "date"]:
            if c in df.columns:
                date_col = c
                break
    if date_col in df.columns:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    else:
        df["date"] = pd.NaT
    df["event_type"] = "construction"
    name_col = "address" if "address" in df.columns else "street_number"
    if name_col not in df.columns:
        name_col = "permit_number" if "permit_number" in df.columns else df.columns[0]
    df["name"] = df[name_col].astype(str).str.slice(0, 80)
    logger.info("  Fetched %d permits", len(df))
    return df[["date", "event_type", "name"]].dropna(subset=["date"]) if "date" in df.columns else pd.DataFrame()


def validate_after_fetch(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    if "date" not in df.columns:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch Building Permits for 311 event correlation")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output-dir", type=str, default="data/events")
    args = parser.parse_args()

    end = datetime.now().date()
    if args.end_date:
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    start = end - timedelta(days=args.days)
    if args.start_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    client = get_client()
    df = fetch_permits(client, start_str, end_str)
    if df.empty:
        logger.warning("No permits data")
        return 1

    if not validate_after_fetch(df):
        logger.warning("Validation failed; writing anyway")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "permits.parquet"
    df.to_parquet(out_path, index=False, compression="gzip")
    logger.info("Saved %s (merge by re-running fetch_events.py)", out_path)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
