"""
Fetch daily weather for SF (Open-Meteo API) for 311 correlation.

Output: data/weather/weather_daily.parquet and data/aggregates/weather_agg.parquet
Columns: date, precip_mm, temp_max_c, temp_min_c (optional: rain flag).

Run periodically; same date range as your 311/events data.
"""
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# SF approximate
LAT, LON = 37.7749, -122.4194
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily weather from Open-Meteo (historical)."""
    logger.info("Fetching weather from Open-Meteo for %s to %s", start_date, end_date)
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "America/Los_Angeles",
    }
    try:
        r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.error("Open-Meteo request failed: %s", e)
        return pd.DataFrame()

    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        logger.warning("No daily data in response")
        return pd.DataFrame()

    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]).date,
        "temp_max_c": daily.get("temperature_2m_max", [None] * len(daily["time"])),
        "temp_min_c": daily.get("temperature_2m_min", [None] * len(daily["time"])),
        "precip_mm": daily.get("precipitation_sum", [0] * len(daily["time"])),
    })
    df["rain_day"] = (df["precip_mm"].fillna(0) > 0.1).astype(int)
    logger.info("  Fetched %d days", len(df))
    return df


def validate_after_fetch(df: pd.DataFrame) -> bool:
    """Light validation: non-empty, date column present, reasonable row count."""
    if df is None or df.empty:
        logger.warning("Validation: empty dataframe")
        return False
    if "date" not in df.columns:
        logger.warning("Validation: missing date column")
        return False
    if len(df) < 1:
        logger.warning("Validation: no rows")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Fetch SF weather for 311 correlation")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=365, help="Days of history if dates not set")
    parser.add_argument("--output-dir", type=str, default="data/weather", help="Output directory")
    args = parser.parse_args()

    end = datetime.now().date()
    if args.end_date:
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    start = end - timedelta(days=args.days)
    if args.start_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    df = fetch_weather(start_str, end_str)
    if df.empty:
        logger.warning("No weather data; skipping write")
        return 1

    if not validate_after_fetch(df):
        logger.warning("Validation failed; writing anyway")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "weather_daily.parquet"
    df.to_parquet(out_path, index=False, compression="gzip")
    logger.info("Saved %s", out_path)

    agg_path = Path("data/aggregates")
    agg_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(agg_path / "weather_agg.parquet", index=False, compression="gzip")
    logger.info("Saved %s/weather_agg.parquet", agg_path)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
