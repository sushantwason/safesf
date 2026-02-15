"""
Build pre-aggregated tables from raw 311 data for 5+ year lookback with minimal storage.

Output tables:
- daily_agg.parquet: date, category, district, count, avg_resolution_hours
- monthly_agg.parquet: year_month, category, district, count
- grid_agg.parquet: lat_cell, lon_cell, category, month, count (for map heatmap)
- hourly_agg.parquet: hour, day_of_week, category, count
- resolution_agg.parquet: category, month, avg_hours, median_hours, count

Run after fetch_data.py. Use with data/311_raw.parquet or GCS sample.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Grid cell size in degrees (~1km in SF)
GRID_SIZE = 0.01

# Cutoff: raw data for last N days; aggregates for the rest
RAW_CUTOFF_DAYS = 90


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names across different data sources."""
    df = df.copy()

    # Date column
    if "requested_datetime" in df.columns and "opened" not in df.columns:
        df["opened"] = pd.to_datetime(df["requested_datetime"], errors="coerce")
    elif "opened" not in df.columns:
        raise ValueError("No date column (opened or requested_datetime) found")

    df["opened"] = pd.to_datetime(df["opened"], errors="coerce")
    df = df[df["opened"].notna()]

    # Coordinates: lat/long vs latitude/longitude
    if "lat" in df.columns and "long" in df.columns:
        df["latitude"] = pd.to_numeric(df["lat"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["long"], errors="coerce")
    elif "lat" in df.columns and "lon" in df.columns:
        df["latitude"] = pd.to_numeric(df["lat"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["lon"], errors="coerce")
    elif "point" in df.columns:
        df["latitude"] = df["point"].apply(
            lambda x: float(x["coordinates"][1]) if isinstance(x, dict) and x and "coordinates" in x else np.nan
        )
        df["longitude"] = df["point"].apply(
            lambda x: float(x["coordinates"][0]) if isinstance(x, dict) and x and "coordinates" in x else np.nan
        )

    # Category
    if "service_name" in df.columns and "category" not in df.columns:
        df["category"] = df["service_name"].fillna("Unknown")
    elif "category" not in df.columns:
        df["category"] = "Unknown"
    else:
        df["category"] = df["category"].fillna("Unknown")

    # District
    if "supervisor_district" in df.columns:
        df["district"] = pd.to_numeric(df["supervisor_district"], errors="coerce")
    else:
        df["district"] = np.nan

    # Resolution time
    if "closed_date" in df.columns and "closed" not in df.columns:
        df["closed"] = pd.to_datetime(df["closed_date"], errors="coerce")
    if "closed" in df.columns:
        df["resolution_hours"] = (df["closed"] - df["opened"]).dt.total_seconds() / 3600
    else:
        df["resolution_hours"] = np.nan

    return df


def build_daily_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Daily counts by category and district."""
    df["date"] = df["opened"].dt.date
    agg = df.groupby(["date", "category", "district"]).agg(
        count=("category", "size"),
        avg_resolution_hours=("resolution_hours", "mean"),
    ).reset_index()
    agg["avg_resolution_hours"] = agg["avg_resolution_hours"].round(2)
    return agg


def build_monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly counts by category and district."""
    df["year_month"] = df["opened"].dt.to_period("M").astype(str)
    agg = df.groupby(["year_month", "category", "district"]).size().reset_index(name="count")
    return agg


def build_grid_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Grid-cell aggregates for map heatmap. Cells ~1km."""
    valid = df[df["latitude"].notna() & df["longitude"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=["lat_cell", "lon_cell", "category", "year_month", "count"])

    valid = valid[(valid["latitude"] >= 37.5) & (valid["latitude"] <= 38.0)]
    valid = valid[(valid["longitude"] >= -123.0) & (valid["longitude"] <= -122.0)]

    valid["lat_cell"] = (valid["latitude"] / GRID_SIZE).astype(int) * GRID_SIZE
    valid["lon_cell"] = (valid["longitude"] / GRID_SIZE).astype(int) * GRID_SIZE
    valid["year_month"] = valid["opened"].dt.to_period("M").astype(str)

    agg = valid.groupby(["lat_cell", "lon_cell", "category", "year_month"]).size().reset_index(name="count")
    return agg


def build_hourly_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Hour and day-of-week patterns by category."""
    df["hour"] = df["opened"].dt.hour
    df["day_of_week"] = df["opened"].dt.dayofweek
    agg = df.groupby(["hour", "day_of_week", "category"]).size().reset_index(name="count")
    return agg


def build_resolution_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Resolution time stats by category and month."""
    df["year_month"] = df["opened"].dt.to_period("M").astype(str)
    valid = df[df["resolution_hours"].notna() & (df["resolution_hours"] > 0)]
    if valid.empty:
        return pd.DataFrame(columns=["category", "year_month", "avg_hours", "median_hours", "count"])

    agg = valid.groupby(["category", "year_month"]).agg(
        avg_hours=("resolution_hours", "mean"),
        median_hours=("resolution_hours", "median"),
        count=("resolution_hours", "size"),
    ).reset_index()
    agg["avg_hours"] = agg["avg_hours"].round(2)
    agg["median_hours"] = agg["median_hours"].round(2)
    return agg


def build_recent_raw(df: pd.DataFrame, cutoff_days: int = RAW_CUTOFF_DAYS) -> pd.DataFrame:
    """Keep only recent raw records for detailed map/recent activity."""
    max_date = df["opened"].max()
    if pd.isna(max_date):
        return pd.DataFrame()
    cutoff = max_date - pd.Timedelta(days=cutoff_days)
    return df[df["opened"] >= cutoff].copy()


def main():
    parser = argparse.ArgumentParser(description="Build pre-aggregated tables from raw 311 data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/311_raw.parquet",
        help="Input parquet file (raw 311 data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/aggregates",
        help="Output directory for aggregate parquet files",
    )
    parser.add_argument(
        "--gcs-url",
        type=str,
        default=None,
        help="Alternatively, fetch from GCS HTTP URL (e.g. sample)",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=RAW_CUTOFF_DAYS,
        help="Number of days to keep as raw recent data",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load input
    if args.gcs_url:
        logger.info(f"Loading from GCS: {args.gcs_url}")
        df = pd.read_parquet(args.gcs_url)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1
        logger.info(f"Loading from {input_path}")
        df = pd.read_parquet(input_path)

    logger.info(f"Loaded {len(df):,} records")

    df = _normalize_schema(df)
    logger.info(f"Normalized. Date range: {df['opened'].min()} to {df['opened'].max()}")

    # Build aggregates
    logger.info("Building daily_agg...")
    daily = build_daily_agg(df)
    daily_path = out_dir / "daily_agg.parquet"
    daily.to_parquet(daily_path, index=False, compression="gzip")
    logger.info(f"  Saved {len(daily):,} rows to {daily_path}")

    logger.info("Building monthly_agg...")
    monthly = build_monthly_agg(df)
    monthly_path = out_dir / "monthly_agg.parquet"
    monthly.to_parquet(monthly_path, index=False, compression="gzip")
    logger.info(f"  Saved {len(monthly):,} rows to {monthly_path}")

    logger.info("Building grid_agg...")
    grid = build_grid_agg(df)
    grid_path = out_dir / "grid_agg.parquet"
    grid.to_parquet(grid_path, index=False, compression="gzip")
    logger.info(f"  Saved {len(grid):,} rows to {grid_path}")

    logger.info("Building hourly_agg...")
    hourly = build_hourly_agg(df)
    hourly_path = out_dir / "hourly_agg.parquet"
    hourly.to_parquet(hourly_path, index=False, compression="gzip")
    logger.info(f"  Saved {len(hourly):,} rows to {hourly_path}")

    logger.info("Building resolution_agg...")
    res = build_resolution_agg(df)
    res_path = out_dir / "resolution_agg.parquet"
    res.to_parquet(res_path, index=False, compression="gzip")
    logger.info(f"  Saved {len(res):,} rows to {res_path}")

    logger.info("Building recent_raw (last %d days)...", args.recent_days)
    recent = build_recent_raw(df, args.recent_days)
    recent_path = out_dir / "recent_raw.parquet"
    recent.to_parquet(recent_path, index=False, compression="gzip")
    logger.info(f"  Saved {len(recent):,} rows to {recent_path}")

    # Metadata
    meta = {
        "date_range": {"start": str(df["opened"].min().date()), "end": str(df["opened"].max().date())},
        "total_records": int(len(df)),
        "aggregate_rows": {
            "daily_agg": int(len(daily)),
            "monthly_agg": int(len(monthly)),
            "grid_agg": int(len(grid)),
            "hourly_agg": int(len(hourly)),
            "resolution_agg": int(len(res)),
            "recent_raw": int(len(recent)),
        },
        "recent_cutoff_days": args.recent_days,
    }
    import json
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
