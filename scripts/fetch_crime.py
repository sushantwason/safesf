"""
Fetch SFPD incident reports from DataSF (wg3w-h783) for 311 correlation.

Output: data/crime/crime_raw.parquet, data/aggregates/crime_agg.parquet
crime_agg columns: date, district (if available), count; or date, count (citywide).

Run periodically. Use same date range as 311 data.
"""
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
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
CRIME_DATASET = "wg3w-h783"


def get_client():
    token = os.getenv("SF_OPEN_DATA_APP_TOKEN")
    return Socrata(
        DOMAIN,
        app_token=token if token and token != "YOUR_TOKEN_HERE" else None,
        timeout=60,
    )


def fetch_crime(client, start_date: str, end_date: str, limit: int = 100000) -> pd.DataFrame:
    """Fetch SFPD Incident Reports. Uses incident_date (or Report Date)."""
    logger.info("Fetching SFPD Incident Reports...")
    # SFPD dataset uses "incident_date" (ISO date string)
    try:
        results = client.get(
            CRIME_DATASET,
            where=f"incident_date >= '{start_date}' AND incident_date <= '{end_date}'",
            limit=limit,
        )
    except Exception as e:
        logger.warning("Date filter failed: %s; trying without filter", e)
        try:
            results = client.get(CRIME_DATASET, limit=min(limit, 50000))
        except Exception as e2:
            logger.error("Could not fetch crime: %s", e2)
            return pd.DataFrame()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(results)
    for col in ["incident_date", "report_datetime", "Report Date"]:
        if col in df.columns:
            df["date"] = pd.to_datetime(df[col], errors="coerce").dt.date
            break
    if "date" not in df.columns and len(df) > 0:
        df["date"] = pd.NaT
    # District: police_district or supervisor district if present
    if "police_district" in df.columns:
        df["district"] = pd.to_numeric(df["police_district"], errors="coerce")
    elif "supervisor_district" in df.columns:
        df["district"] = pd.to_numeric(df["supervisor_district"], errors="coerce")
    else:
        df["district"] = np.nan
    logger.info("  Fetched %d incidents", len(df))
    return df


def build_crime_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Daily counts by date, optionally by district."""
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["date", "district", "count"])
    df = df.dropna(subset=["date"])
    if "district" in df.columns and df["district"].notna().any():
        agg = df.groupby(["date", "district"]).size().reset_index(name="count")
    else:
        agg = df.groupby("date").size().reset_index(name="count")
        agg["district"] = None
    return agg


def validate_after_fetch(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    if "date" not in df.columns:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch SFPD crime data for 311 correlation")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output-dir", type=str, default="data/crime")
    parser.add_argument("--incremental", action="store_true", help="Only fetch last N days (use with --days)")
    args = parser.parse_args()

    end = datetime.now().date()
    if args.end_date:
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    start = end - timedelta(days=args.days)
    if args.start_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()

    if args.incremental:
        start = end - timedelta(days=min(args.days, 31))
        logger.info("Incremental: fetching %s to %s", start, end)

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    client = get_client()
    df = fetch_crime(client, start_str, end_str)
    if df.empty:
        logger.warning("No crime data")
        return 1

    if not validate_after_fetch(df):
        logger.warning("Validation failed; writing anyway")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "crime_raw.parquet", index=False, compression="gzip")

    agg = build_crime_agg(df)
    if not agg.empty:
        agg_path = Path("data/aggregates")
        agg_path.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(agg_path / "crime_agg.parquet", index=False, compression="gzip")
        logger.info("Saved crime_agg: %d rows", len(agg))

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
