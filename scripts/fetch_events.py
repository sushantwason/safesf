"""
Fetch event data from DataSF for 311 request correlation.

Datasets:
- Our415 Events and Activities (8i3s-ih2a)
- Temporary Street Closures (8x25-yybr)

Run periodically to keep event data current. Used by build_aggregates for correlation.
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
OUR415_DATASET = "8i3s-ih2a"
STREET_CLOSURES_DATASET = "8x25-yybr"


def get_client():
    token = os.getenv("SF_OPEN_DATA_APP_TOKEN")
    return Socrata(
        DOMAIN,
        app_token=token if token and token != "YOUR_TOKEN_HERE" else None,
        timeout=60,
    )


def fetch_our415_events(client, start_date: str, end_date: str, limit: int = 50000) -> pd.DataFrame:
    """Fetch Our415 Events and Activities from DataSF."""
    logger.info("Fetching Our415 Events...")
    try:
        results = client.get(
            OUR415_DATASET,
            where=f"event_start_date >= '{start_date}' AND event_start_date <= '{end_date}'",
            limit=limit,
        )
    except Exception as e:
        logger.warning(f"Our415 date filter failed, trying without: {e}")
        try:
            results = client.get(OUR415_DATASET, limit=min(limit, 10000))
        except Exception as e2:
            logger.error(f"Could not fetch Our415: {e2}")
            return pd.DataFrame()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(results)
    # Normalize date column (Our415 uses event_start_date)
    for col in ["event_start_date", "start_date", "start_datetime", "date"]:
        if col in df.columns:
            df["event_date"] = pd.to_datetime(df[col], errors="coerce").dt.date
            break
    if "event_date" not in df.columns and len(df) > 0:
        df["event_date"] = pd.NaT
    logger.info(f"  Fetched {len(df)} Our415 events")
    return df


def fetch_street_closures(client, start_date: str, end_date: str, limit: int = 50000) -> pd.DataFrame:
    """Fetch Temporary Street Closures from DataSF."""
    logger.info("Fetching Temporary Street Closures...")
    try:
        results = client.get(
            STREET_CLOSURES_DATASET,
            where=f"start_date >= '{start_date}' AND start_date <= '{end_date}'",
            limit=limit,
        )
    except Exception as e:
        logger.warning(f"Street closures date filter failed: {e}")
        try:
            results = client.get(
                STREET_CLOSURES_DATASET,
                limit=min(limit, 10000),
            )
        except Exception as e2:
            logger.warning(f"Alternative column? {e2}")
            # Try generic date column
            try:
                results = client.get(STREET_CLOSURES_DATASET, limit=5000)
            except Exception:
                return pd.DataFrame()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(results)
    for col in ["start_date", "closure_start_date", "closure_date", "date"]:
        if col in df.columns:
            df["closure_date"] = pd.to_datetime(df[col], errors="coerce").dt.date
            break
    if "closure_date" not in df.columns and len(df) > 0:
        df["closure_date"] = pd.NaT
    logger.info(f"  Fetched {len(df)} street closures")
    return df


def build_events_agg(our415: pd.DataFrame, closures: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily event counts for correlation with 311.
    Output: date, event_count, street_closure_count
    """
    rows = []
    all_dates = set()

    if not our415.empty and "event_date" in our415.columns:
        our415 = our415.dropna(subset=["event_date"])
        for d, cnt in our415["event_date"].value_counts().items():
            all_dates.add(d)
            rows.append({"date": d, "event_count": int(cnt), "street_closure_count": 0})
    if not closures.empty and "closure_date" in closures.columns:
        closures = closures.dropna(subset=["closure_date"])
        for d, cnt in closures["closure_date"].value_counts().items():
            all_dates.add(d)
            rows.append({"date": d, "event_count": 0, "street_closure_count": int(cnt)})

    if not rows:
        return pd.DataFrame(columns=["date", "event_count", "street_closure_count"])

    agg = pd.DataFrame(rows)
    agg = agg.groupby("date").agg({"event_count": "sum", "street_closure_count": "sum"}).reset_index()
    return agg


def build_events_by_date(our415: pd.DataFrame, closures: pd.DataFrame) -> pd.DataFrame:
    """
    Build day-by-day event names for annotations (e.g. "Noise spike maybe due to: Super Bowl").
    Output: date, event_names (JSON list of strings)
    """
    date_events = {}

    if not our415.empty and "event_date" in our415.columns:
        name_col = "event_name" if "event_name" in our415.columns else "event_description"
        if name_col not in our415.columns:
            name_col = our415.columns[0]
        our415 = our415.dropna(subset=["event_date"])
        for _, row in our415.iterrows():
            d = row["event_date"]
            name = str(row.get(name_col, "")).strip() if pd.notna(row.get(name_col)) else ""
            if name and len(name) < 120:
                date_events.setdefault(d, []).append(("event", name))

    if not closures.empty and "closure_date" in closures.columns:
        name_col = "case_name" if "case_name" in closures.columns else "type"
        if name_col not in closures.columns:
            name_col = "loc_desc" if "loc_desc" in closures.columns else closures.columns[0]
        street_col = "street" if "street" in closures.columns else None
        closures = closures.dropna(subset=["closure_date"])
        for _, row in closures.iterrows():
            d = row["closure_date"]
            name = str(row.get(name_col, "")).strip() if pd.notna(row.get(name_col)) else ""
            if street_col and pd.notna(row.get(street_col)):
                name = (name + " (" + str(row[street_col]) + ")").strip() if name else str(row[street_col])
            if name and len(name) < 100:
                date_events.setdefault(d, []).append(("closure", name))

    # Dedupe and limit per date
    rows = []
    for d, items in sorted(date_events.items()):
        seen = set()
        names = []
        for src, n in items:
            n_clean = n[:80] + "…" if len(n) > 80 else n
            if n_clean not in seen:
                seen.add(n_clean)
                names.append(n_clean)
                if len(names) >= 5:
                    break
        rows.append({"date": d, "event_names": "|".join(names)})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "event_names"])


def main():
    parser = argparse.ArgumentParser(description="Fetch DataSF event data for 311 correlation")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--output-dir", type=str, default="data/events", help="Output directory")
    parser.add_argument("--days", type=int, default=365, help="Days of history if dates not set")
    args = parser.parse_args()

    end = datetime.now().date()
    if args.end_date:
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    start = end - timedelta(days=args.days)
    if args.start_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    logger.info(f"Fetching events from {start_str} to {end_str}")

    client = get_client()

    our415 = fetch_our415_events(client, start_str, end_str)
    closures = fetch_street_closures(client, start_str, end_str)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not our415.empty:
        our415.to_parquet(out_dir / "our415_events.parquet", index=False, compression="gzip")
    if not closures.empty:
        closures.to_parquet(out_dir / "street_closures.parquet", index=False, compression="gzip")

    agg_path = Path("data/aggregates")
    agg_path.mkdir(parents=True, exist_ok=True)

    agg = build_events_agg(our415, closures)
    if not agg.empty:
        agg.to_parquet(out_dir / "events_daily_agg.parquet", index=False, compression="gzip")
        agg.to_parquet(agg_path / "events_agg.parquet", index=False, compression="gzip")
        logger.info(f"Saved events_agg: {len(agg)} days")

    events_by_date = build_events_by_date(our415, closures)
    if not events_by_date.empty:
        events_by_date.to_parquet(out_dir / "events_by_date.parquet", index=False, compression="gzip")
        events_by_date.to_parquet(agg_path / "events_by_date.parquet", index=False, compression="gzip")
        logger.info(f"Saved events_by_date: {len(events_by_date)} days with event names")

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
