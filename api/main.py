"""
FastAPI backend for SF 311 Prediction Service
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import json
import io
import time
import requests
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISTRICT_NAMES = {
    1: 'Richmond',
    2: 'Marina, Cow Hollow',
    3: 'North Beach, Chinatown',
    4: 'Sunset',
    5: 'Haight-Ashbury, Fillmore',
    6: 'SOMA, Tenderloin',
    7: 'Mission, Bernal Heights',
    8: 'Castro, Noe Valley',
    9: 'Mission, Portola',
    10: 'Potrero Hill, Bayview',
    11: 'Excelsior, Outer Mission',
}

CATEGORY_COLORS = [
    '#e74c3c',  # red
    '#3498db',  # blue
    '#2ecc71',  # green
    '#f39c12',  # orange
    '#9b59b6',  # purple
    '#1abc9c',  # teal
    '#e67e22',  # dark orange
    '#34495e',  # dark blue-gray
    '#e91e63',  # pink
    '#00bcd4',  # cyan
]

CACHE_TTL_SECONDS = 3600  # 1 hour

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

MODELS: Dict[str, Any] = {}
METADATA: Dict[str, Any] = {}
DATA_CACHE: Optional[pd.DataFrame] = None
ANALYTICS_CACHE: Optional[Dict[str, Any]] = None
CACHE_LOADED_AT: Optional[float] = None  # monotonic timestamp
AGGREGATES_CACHE: Optional[Dict[str, pd.DataFrame]] = None  # daily_agg, monthly_agg, grid_agg, etc.
RAW_CUTOFF_DAYS = 90  # Use raw data for ranges <= this; aggregates for longer


# ---------------------------------------------------------------------------
# Data / model loading helpers
# ---------------------------------------------------------------------------

def load_aggregates():
    """Load pre-aggregated tables for 5+ year lookback (from GCS or local)."""
    global AGGREGATES_CACHE

    gcs_base = "https://storage.googleapis.com/sf-311-data-personal/aggregates"
    agg_files = [
        "daily_agg", "monthly_agg", "grid_agg", "hourly_agg", "resolution_agg", "recent_raw",
        "events_agg", "events_by_date", "crime_agg", "weather_agg",
    ]

    # Try GCS first
    for name in agg_files:
        try:
            url = f"{gcs_base}/{name}.parquet"
            df = pd.read_parquet(url)
            if AGGREGATES_CACHE is None:
                AGGREGATES_CACHE = {}
            AGGREGATES_CACHE[name] = df
            logger.info(f"Loaded aggregate {name}: {len(df):,} rows")
        except Exception as e:
            logger.debug(f"Could not load {name} from GCS: {e}")

    if AGGREGATES_CACHE:
        return

    # Fallback: local data/aggregates/
    agg_paths = [
        Path("/app") / "data" / "aggregates",
        Path(__file__).parent.parent / "data" / "aggregates",
    ]
    for agg_dir in agg_paths:
        if not agg_dir.exists():
            continue
        AGGREGATES_CACHE = {}
        for name in agg_files:
            p = agg_dir / f"{name}.parquet"
            if p.exists():
                try:
                    AGGREGATES_CACHE[name] = pd.read_parquet(p)
                    logger.info(f"Loaded aggregate {name} from {p}: {len(AGGREGATES_CACHE[name]):,} rows")
                except Exception as e:
                    logger.warning(f"Error loading {p}: {e}")
        if AGGREGATES_CACHE:
            break


def ensure_aggregates_loaded():
    """Lazy load aggregates on first use."""
    global AGGREGATES_CACHE
    if AGGREGATES_CACHE is None:
        load_aggregates()


def _days_in_range(start_date: Optional[str], end_date: Optional[str]) -> Optional[int]:
    """Return approximate days in range, or None if unbounded."""
    if not start_date and not end_date:
        return None
    try:
        sd = pd.to_datetime(start_date) if start_date else pd.Timestamp("2000-01-01")
        ed = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        return max(0, (ed - sd).days)
    except Exception:
        return None


def _filter_agg_date_range(
    df: pd.DataFrame,
    date_col: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    """Filter aggregate DataFrame by date range."""
    if not start_date and not end_date:
        return df
    try:
        sd = pd.to_datetime(start_date) if start_date else pd.Timestamp.min
        ed = pd.to_datetime(end_date) if end_date else pd.Timestamp.max
        if date_col == "date":
            df = df.copy()
            df["_dt"] = pd.to_datetime(df[date_col])
            df = df[(df["_dt"] >= sd) & (df["_dt"] <= ed)]
            df = df.drop(columns=["_dt"], errors="ignore")
        elif date_col == "year_month":
            # Include month if it overlaps [start_date, end_date]
            df = df.copy()
            df["_first"] = pd.to_datetime(df[date_col] + "-01")
            df["_last"] = df["_first"] + pd.offsets.MonthEnd(0)
            df = df[(df["_last"] >= sd) & (df["_first"] <= ed)]
            df = df.drop(columns=["_first", "_last"], errors="ignore")
        else:
            df = df.copy()
            df["_dt"] = pd.to_datetime(df[date_col])
            df = df[(df["_dt"] >= sd) & (df["_dt"] <= ed)]
            df = df.drop(columns=["_dt"], errors="ignore")
    except Exception:
        pass
    return df


def load_data(data_dir: str = "data"):
    """Load sample 311 data from GCS for fast loading (9.5x faster)"""
    global DATA_CACHE

    gcs_http_url = "https://storage.googleapis.com/sf-311-data-personal/311_sample.parquet"

    try:
        logger.info(f"Loading sample dataset from GCS: {gcs_http_url}")
        DATA_CACHE = pd.read_parquet(gcs_http_url)
        logger.info(f"Loaded {len(DATA_CACHE):,} records (stratified sample) from GCS")
        logger.info(f"Date range: {DATA_CACHE['opened'].min()} to {DATA_CACHE['opened'].max()}")
        logger.info(f"Categories: {DATA_CACHE['category'].nunique()}")
        return
    except Exception as e:
        logger.warning(f"Could not load sample from GCS: {e}")
        logger.info("Falling back to local file...")

    # Fallback to local files
    possible_paths = [
        Path(data_dir),
        Path("/app") / data_dir,
        Path(__file__).parent.parent / data_dir,
    ]

    data_files: list = []
    for data_path in possible_paths:
        if data_path.exists():
            logger.info(f"Checking {data_path} for data files...")
            files = list(data_path.glob("*.parquet"))
            if files:
                logger.info(f"Found {len(files)} parquet files in {data_path}")
                data_files.extend(files)
                break

    if not data_files:
        logger.warning("No data files found in any location")
        return

    data_file = max(data_files, key=lambda p: p.stat().st_size)

    try:
        DATA_CACHE = pd.read_parquet(data_file)
        logger.info(f"Loaded {len(DATA_CACHE):,} records from {data_file.name}")
        logger.info(f"Date range: {DATA_CACHE['opened'].min()} to {DATA_CACHE['opened'].max()}")
        logger.info(f"Categories: {DATA_CACHE['category'].nunique()}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")


def load_models(models_dir: str = "models"):
    """Load trained models on startup"""
    global MODELS, METADATA

    models_path = Path(models_dir)

    if not models_path.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return

    metadata_path = models_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            METADATA = json.load(f)
        logger.info(f"Loaded metadata for {len(METADATA.get('categories', []))} categories")

    for model_file in models_path.glob("xgb_*.pkl"):
        category = model_file.stem.replace("xgb_", "").replace("_", " ")
        try:
            MODELS[f"xgb_{category}"] = joblib.load(model_file)
            logger.info(f"Loaded XGBoost model for: {category}")
        except Exception as e:
            logger.error(f"Error loading {model_file}: {e}")

    for model_file in models_path.glob("prophet_*.pkl"):
        category = model_file.stem.replace("prophet_", "").replace("_", " ")
        try:
            MODELS[f"prophet_{category}"] = joblib.load(model_file)
            logger.info(f"Loaded Prophet model for: {category}")
        except Exception as e:
            logger.error(f"Error loading {model_file}: {e}")

    logger.info(f"Total models loaded: {len(MODELS)}")


def load_analytics_cache():
    """Load pre-computed analytics from GCS for instant responses"""
    global ANALYTICS_CACHE, CACHE_LOADED_AT

    gcs_http_url = "https://storage.googleapis.com/sf-311-data-personal/analytics_cache.json"

    try:
        logger.info("Loading analytics cache from GCS...")
        response = requests.get(gcs_http_url, timeout=15)
        response.raise_for_status()
        ANALYTICS_CACHE = response.json()
        CACHE_LOADED_AT = time.monotonic()
        logger.info("Analytics cache loaded")
    except Exception as e:
        logger.warning(f"Could not load analytics cache: {e}")
        ANALYTICS_CACHE = {}
        CACHE_LOADED_AT = time.monotonic()


def ensure_data_loaded():
    """Lazy load data on first request to speed up startup"""
    global DATA_CACHE
    if DATA_CACHE is None:
        logger.info("Lazy loading data on first request...")
        load_data()
        logger.info("Data loaded successfully")


def ensure_analytics_loaded():
    """Lazy load analytics cache on first request; reload if stale (TTL-based)."""
    global ANALYTICS_CACHE, CACHE_LOADED_AT
    now = time.monotonic()
    if ANALYTICS_CACHE is None or CACHE_LOADED_AT is None or (now - CACHE_LOADED_AT) > CACHE_TTL_SECONDS:
        load_analytics_cache()


def _sanitize_value(val):
    """Convert NaN / NaT / inf to None for JSON safety."""
    if val is None:
        return None
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
    if isinstance(val, pd.Timestamp):
        if pd.isna(val):
            return None
        return val.isoformat()
    return val


def _safe_json_response(data: Any) -> JSONResponse:
    """Return a JSONResponse after cleaning NaN/NaT values."""
    cleaned = json.loads(
        pd.io.json.dumps(data) if hasattr(pd.io.json, 'dumps')
        else json.dumps(data, default=str)
    )
    return JSONResponse(content=cleaned)


def _filter_date_range(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """Filter a DataFrame by optional start/end date strings (YYYY-MM-DD)."""
    if start_date:
        try:
            sd = pd.to_datetime(start_date)
            df = df[df['opened'] >= sd]
        except Exception:
            pass
    if end_date:
        try:
            ed = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[df['opened'] <= ed]
        except Exception:
            pass
    return df


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------

def _get_allowed_origins() -> List[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "")
    if raw.strip():
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        logger.info(f"CORS allowed origins from env: {origins}")
        return origins
    logger.warning("ALLOWED_ORIGINS not set - defaulting to ['*']. Configure for production!")
    return ["*"]


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting SF 311 Predictor API...")
    logger.info("Data will be loaded on first request (lazy loading for faster startup)")
    load_models()
    logger.info("Application ready!")
    yield
    logger.info("Shutting down SF 311 Predictor API...")


# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SF 311 Predictor API",
    description="Predict San Francisco 311 service request volumes",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    category: str = Field(..., description="311 request category")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


class PredictionResponse(BaseModel):
    category: str
    predictions: List[Dict[str, Any]]
    model_used: str
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    categories_available: List[str]
    version: str


class StatsResponse(BaseModel):
    total_requests: int
    categories: Dict[str, int]
    date_range: Dict[str, str]


# ---------------------------------------------------------------------------
# Endpoints – health / categories / metrics
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    categories = METADATA.get('categories', [])
    return HealthResponse(
        status="healthy" if MODELS else "no_models_loaded",
        models_loaded=len(MODELS),
        categories_available=categories,
        version="1.0.0",
    )


@app.get("/api/categories")
async def get_categories():
    """Get available categories"""
    ensure_data_loaded()

    if DATA_CACHE is not None and not DATA_CACHE.empty:
        categories = DATA_CACHE['category'].unique().tolist()
        category_counts = DATA_CACHE['category'].value_counts().to_dict()
        return {
            "categories": categories,
            "total": len(categories),
            "counts": category_counts,
        }

    categories = METADATA.get('categories', [])
    if not categories:
        return {"categories": [], "total": 0, "message": "No data loaded yet"}
    return {"categories": categories, "total": len(categories)}


@app.get("/metrics")
async def get_metrics():
    """Prometheus-style metrics endpoint"""
    return {
        "models_loaded": len(MODELS),
        "categories_available": len(METADATA.get('categories', [])),
        "uptime_seconds": 0,
    }


# ---------------------------------------------------------------------------
# Endpoints – predictions (kept as-is)
# ---------------------------------------------------------------------------

@app.get("/api/predict/volume")
async def predict_volume(
    category: str = Query(..., description="311 request category"),
    days: int = Query(7, ge=1, le=30, description="Number of days to predict"),
    model: str = Query("xgboost", description="Model to use (xgboost or prophet)"),
):
    """Predict 311 request volume for a category"""
    available_categories = METADATA.get('categories', [])
    if category not in available_categories:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category}' not found. Available: {available_categories}",
        )

    model_key = f"{model}_{category}"
    if model_key not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model not found for category: {category}")

    loaded_model = MODELS[model_key]

    try:
        predictions = []
        today = datetime.now().date()

        if model == "prophet":
            future_dates = pd.DataFrame({'ds': pd.date_range(start=today, periods=days, freq='D')})
            forecast = loaded_model.predict(future_dates)
            for _, row in forecast.iterrows():
                predictions.append({
                    'date': row['ds'].strftime('%Y-%m-%d'),
                    'predicted_volume': max(0, int(round(row['yhat']))),
                    'lower_bound': max(0, int(round(row['yhat_lower']))),
                    'upper_bound': max(0, int(round(row['yhat_upper']))),
                })
        else:
            feature_columns = METADATA.get('feature_columns', [])
            for day_offset in range(days):
                pred_date = today + timedelta(days=day_offset)
                features = np.zeros(len(feature_columns))
                features[0] = pred_date.year
                features[1] = pred_date.month
                features[2] = pred_date.day
                features[3] = pred_date.weekday()
                features[4] = pred_date.isocalendar()[1]
                features[5] = 1 if pred_date.weekday() >= 5 else 0
                volume = loaded_model.predict([features])[0]
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'predicted_volume': max(0, int(round(volume))),
                    'confidence': 'medium',
                })

        return PredictionResponse(
            category=category,
            predictions=predictions,
            model_used=model,
            generated_at=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/predict/trends")
async def predict_trends(
    category: str = Query("all", description="Category or 'all'"),
    timeframe: str = Query("30d", description="Timeframe (7d, 30d, 90d)"),
):
    """Get predicted trends for categories"""
    days_map = {"7d": 7, "30d": 30, "90d": 90}
    days = days_map.get(timeframe, 30)
    categories = [category] if category != "all" else METADATA.get('categories', [])[:5]

    trends = {}
    for cat in categories:
        if f"prophet_{cat}" not in MODELS:
            continue
        try:
            mdl = MODELS[f"prophet_{cat}"]
            today = datetime.now().date()
            future_dates = pd.DataFrame({'ds': pd.date_range(start=today, periods=days, freq='D')})
            forecast = mdl.predict(future_dates)
            values = forecast['yhat'].values
            trend_direction = "increasing" if values[-1] > values[0] else "decreasing"
            trend_magnitude = abs((values[-1] - values[0]) / values[0] * 100)
            trends[cat] = {
                'direction': trend_direction,
                'magnitude_percent': round(trend_magnitude, 2),
                'predicted_total': int(values.sum()),
                'daily_average': int(values.mean()),
            }
        except Exception as e:
            logger.error(f"Error calculating trend for {cat}: {e}")

    return {'timeframe': timeframe, 'trends': trends, 'generated_at': datetime.now().isoformat()}


@app.get("/api/predict/hotspots")
async def predict_hotspots(
    date: Optional[str] = Query(None, description="Date for prediction (YYYY-MM-DD)"),
    top_n: int = Query(10, description="Number of top hotspots to return"),
):
    """Predict geographic hotspots for 311 requests (placeholder)."""
    hotspots = [
        {
            'neighborhood': 'Tenderloin',
            'predicted_volume': 245,
            'latitude': 37.7841,
            'longitude': -122.4131,
            'top_categories': ['Street Cleaning', 'Graffiti', 'Illegal Dumping'],
        },
        {
            'neighborhood': 'Mission',
            'predicted_volume': 198,
            'latitude': 37.7599,
            'longitude': -122.4148,
            'top_categories': ['Graffiti', 'Street Cleaning', 'Parking'],
        },
        {
            'neighborhood': 'Financial District/South Beach',
            'predicted_volume': 156,
            'latitude': 37.7946,
            'longitude': -122.3999,
            'top_categories': ['Street Cleaning', 'Parking', 'Tree Maintenance'],
        },
    ]

    return {
        'date': date or datetime.now().strftime('%Y-%m-%d'),
        'hotspots': hotspots[:top_n],
        'methodology': 'Historical pattern analysis',
        'generated_at': datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Endpoints – data
# ---------------------------------------------------------------------------

@app.get("/api/data/recent")
async def get_recent_data(limit: int = Query(100, ge=1, le=1000)):
    """Get recent 311 service requests"""
    ensure_data_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.sort_values('opened', ascending=False).head(limit)

    records = []
    for _, row in df.iterrows():
        district_val = row.get('supervisor_district')
        if pd.notna(district_val):
            try:
                district_val = int(float(district_val))
            except (ValueError, TypeError):
                district_val = None
        else:
            district_val = None

        records.append({
            'id': str(row.get('service_request_id', 'N/A')),
            'category': str(row.get('category', 'Unknown')),
            'status': str(row.get('status_description', 'Unknown')),
            'opened': row['opened'].isoformat() if pd.notna(row.get('opened')) else None,
            'closed': row['closed'].isoformat() if pd.notna(row.get('closed')) else None,
            'address': str(row.get('address', 'N/A')),
            'district': district_val,
            'latitude': None,   # Disabled to avoid JSON serialization issues with NaN
            'longitude': None,  # Disabled to avoid JSON serialization issues with NaN
        })

    return {'total': len(records), 'limit': limit, 'data': records}


@app.get("/api/data/stats")
async def get_data_stats():
    """Get statistics about the underlying data"""
    ensure_data_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        metrics = METADATA.get('metrics', {})
        return {
            'models_trained': len(metrics),
            'categories': [],
            'total_requests': 0,
            'date_range': {},
        }

    total = len(DATA_CACHE)
    categories = DATA_CACHE['category'].value_counts().to_dict()

    date_range = {
        'start': DATA_CACHE['opened'].min().isoformat() if pd.notna(DATA_CACHE['opened'].min()) else None,
        'end': DATA_CACHE['opened'].max().isoformat() if pd.notna(DATA_CACHE['opened'].max()) else None,
    }

    if 'opened' in DATA_CACHE.columns:
        days = (DATA_CACHE['opened'].max() - DATA_CACHE['opened'].min()).days + 1
        daily_avg = total / max(days, 1)
    else:
        daily_avg = 0

    return {
        'total_requests': total,
        'categories': categories,
        'date_range': date_range,
        'daily_average': round(daily_avg, 2),
        'top_categories': dict(list(categories.items())[:10]),
    }


@app.get("/api/data/export")
async def export_data(
    start_date: Optional[str] = Query(None, description="Filter start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Filter end date YYYY-MM-DD"),
):
    """Export dataset as CSV (optionally filtered by date range)."""
    ensure_data_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()
    if start_date or end_date:
        df = _filter_date_range(df, start_date, end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="No rows in selected date range")

    def _iter_csv():
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        yield from buf

    return StreamingResponse(
        _iter_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sf_311_data.csv"},
    )


# ---------------------------------------------------------------------------
# Endpoints – analytics (with optional date range filtering)
# ---------------------------------------------------------------------------

@app.get("/api/data/daily-timeseries")
async def get_daily_timeseries(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Get daily aggregated request counts (uses aggregates for long ranges). Includes events_by_date for annotations."""
    if start_date is None and end_date is None:
        ensure_analytics_loaded()
        if ANALYTICS_CACHE and 'daily_timeseries' in ANALYTICS_CACHE:
            cached = ANALYTICS_CACHE['daily_timeseries'].copy()
            events_map = _get_events_by_date_dict()
            if events_map and 'dates' in cached:
                cached['events_by_date'] = {d: events_map.get(d, []) for d in cached.get('dates', [])}
            weather_map = _get_weather_by_date_dict()
            if weather_map and 'dates' in cached:
                cached['weather_by_date'] = {d: weather_map.get(d) for d in cached.get('dates', []) if weather_map.get(d)}
            return cached

    days = _days_in_range(start_date, end_date)
    use_aggregates = days is not None and days > RAW_CUTOFF_DAYS

    if use_aggregates:
        ensure_aggregates_loaded()
        if AGGREGATES_CACHE and 'daily_agg' in AGGREGATES_CACHE:
            agg = AGGREGATES_CACHE['daily_agg'].copy()
            agg = _filter_agg_date_range(agg, 'date', start_date, end_date)
            if not agg.empty:
                daily_counts = agg.groupby('date')['count'].sum().reset_index()
                daily_counts = daily_counts.sort_values('date')
                dates_list = [str(d) for d in daily_counts['date'].tolist()]
                events_map = _get_events_by_date_dict()
                weather_map = _get_weather_by_date_dict()
                return {
                    'dates': dates_list,
                    'counts': daily_counts['count'].astype(int).tolist(),
                    'total_days': len(daily_counts),
                    'events_by_date': {d: events_map.get(d, []) for d in dates_list},
                    'weather_by_date': {d: weather_map.get(d) for d in dates_list if weather_map.get(d)},
                }

    ensure_data_loaded()
    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()
    df = _filter_date_range(df, start_date, end_date)
    df['date'] = pd.to_datetime(df['opened']).dt.date

    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_counts = daily_counts.sort_values('date')

    dates_list = [str(d) for d in daily_counts['date'].tolist()]
    events_map = _get_events_by_date_dict()
    events_for_dates = {d: events_map.get(d, []) for d in dates_list}
    weather_map = _get_weather_by_date_dict()
    weather_for_dates = {d: weather_map.get(d) for d in dates_list if weather_map.get(d)}

    return {
        'dates': dates_list,
        'counts': daily_counts['count'].tolist(),
        'total_days': len(daily_counts),
        'events_by_date': events_for_dates,
        'weather_by_date': weather_for_dates,
    }


@app.get("/api/analytics/day-of-week")
async def get_day_of_week_analytics(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Get request patterns by day of week (uses aggregates for long ranges)."""
    if start_date is None and end_date is None:
        ensure_analytics_loaded()
        if ANALYTICS_CACHE and 'day_of_week' in ANALYTICS_CACHE:
            return ANALYTICS_CACHE['day_of_week']

    days = _days_in_range(start_date, end_date)
    use_aggregates = days is not None and days > RAW_CUTOFF_DAYS

    if use_aggregates:
        ensure_aggregates_loaded()
    if use_aggregates and AGGREGATES_CACHE and 'daily_agg' in AGGREGATES_CACHE:
        agg = AGGREGATES_CACHE['daily_agg'].copy()
        agg = _filter_agg_date_range(agg, 'date', start_date, end_date)
        if not agg.empty:
            agg['date_dt'] = pd.to_datetime(agg['date'])
            agg['day_of_week'] = agg['date_dt'].dt.dayofweek
            dow_counts = agg.groupby('day_of_week')['count'].sum()
            names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            total = dow_counts.sum()
            return {
                'days': names,
                'counts': [int(dow_counts.get(i, 0)) for i in range(7)],
                'percentages': [round(dow_counts.get(i, 0) / total * 100, 1) if total else 0 for i in range(7)],
            }

    ensure_data_loaded()
    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()
    df = _filter_date_range(df, start_date, end_date)
    df['day_of_week'] = pd.to_datetime(df['opened']).dt.day_name()

    dow_counts = df['day_of_week'].value_counts()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    total = len(df)

    return {
        'days': [d for d in dow_order if d in dow_counts.index],
        'counts': [int(dow_counts[d]) for d in dow_order if d in dow_counts.index],
        'percentages': [round(dow_counts.get(d, 0) / total * 100, 1) if total else 0 for d in dow_order],
    }


@app.get("/api/analytics/hourly-pattern")
async def get_hourly_pattern(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Get request patterns by hour of day (cached when no date filter)."""
    if start_date is None and end_date is None:
        ensure_analytics_loaded()
        if ANALYTICS_CACHE and 'hourly_pattern' in ANALYTICS_CACHE:
            return ANALYTICS_CACHE['hourly_pattern']

    ensure_data_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()
    df = _filter_date_range(df, start_date, end_date)
    df['hour'] = pd.to_datetime(df['opened']).dt.hour

    hourly_counts = df.groupby('hour').size().sort_index()

    return {
        'hours': [int(h) for h in hourly_counts.index],
        'counts': [int(c) for c in hourly_counts.values],
        'peak_hour': int(hourly_counts.idxmax()),
        'peak_count': int(hourly_counts.max()),
    }


@app.get("/api/analytics/weather-impact")
async def get_weather_impact():
    """Analyze weather impact on requests (cached)."""
    ensure_analytics_loaded()

    if ANALYTICS_CACHE and 'weather_impact' in ANALYTICS_CACHE:
        return ANALYTICS_CACHE['weather_impact']

    ensure_data_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    if 'is_rainy' not in DATA_CACHE.columns:
        raise HTTPException(status_code=503, detail="Weather data not available")

    df = DATA_CACHE.copy()
    df['date'] = pd.to_datetime(df['opened']).dt.date

    rainy_avg = df[df['is_rainy'] == True].groupby('date').size().mean()  # noqa: E712
    dry_avg = df[df['is_rainy'] == False].groupby('date').size().mean()  # noqa: E712
    overall_impact = (rainy_avg / dry_avg - 1) * 100 if dry_avg > 0 else 0

    top_cats = df['category'].value_counts().head(10).index
    category_impacts = []

    for cat in top_cats:
        cat_data = df[df['category'] == cat]
        rainy_count = cat_data[cat_data['is_rainy'] == True].groupby('date').size().mean()  # noqa: E712
        dry_count = cat_data[cat_data['is_rainy'] == False].groupby('date').size().mean()  # noqa: E712

        if pd.notna(rainy_count) and pd.notna(dry_count) and dry_count > 0:
            impact = (rainy_count / dry_count - 1) * 100
            category_impacts.append({'category': cat, 'impact_percent': round(impact, 1)})

    category_impacts.sort(key=lambda x: abs(x['impact_percent']), reverse=True)

    return {
        'overall_impact_percent': round(overall_impact, 1),
        'rainy_day_avg': round(rainy_avg, 0) if pd.notna(rainy_avg) else 0,
        'dry_day_avg': round(dry_avg, 0) if pd.notna(dry_avg) else 0,
        'category_impacts': category_impacts[:10],
    }


@app.get("/api/analytics/monthly-trends")
async def get_monthly_trends(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Get monthly request volume trends (uses aggregates for long ranges)."""
    if start_date is None and end_date is None:
        ensure_analytics_loaded()
        if ANALYTICS_CACHE and 'monthly_trends' in ANALYTICS_CACHE:
            return ANALYTICS_CACHE['monthly_trends']

    days = _days_in_range(start_date, end_date)
    use_aggregates = days is not None and days > RAW_CUTOFF_DAYS

    if use_aggregates:
        ensure_aggregates_loaded()
    if use_aggregates and AGGREGATES_CACHE and 'monthly_agg' in AGGREGATES_CACHE:
        agg = AGGREGATES_CACHE['monthly_agg'].copy()
        agg = _filter_agg_date_range(agg, 'year_month', start_date, end_date)
        if not agg.empty:
            monthly_counts = agg.groupby('year_month')['count'].sum().sort_index()
            return {
                'months': [str(m) for m in monthly_counts.index],
                'counts': [int(c) for c in monthly_counts.values],
                'average': round(float(monthly_counts.mean()), 0),
            }

    ensure_data_loaded()
    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()
    df = _filter_date_range(df, start_date, end_date)
    df['month'] = pd.to_datetime(df['opened']).dt.to_period('M').astype(str)

    monthly_counts = df.groupby('month').size().sort_index()

    return {
        'months': [str(m) for m in monthly_counts.index],
        'counts': [int(c) for c in monthly_counts.values],
        'average': round(float(monthly_counts.mean()), 0),
    }


@app.get("/api/analytics/category-insights")
async def get_category_insights():
    """Get category-specific deep insights (growth trends, hotspots)"""
    return {
        "category_growth_trends": [
            {"category": "Street and Sidewalk Cleaning", "trend": "DECLINING", "growth_pct": -14.7},
            {"category": "Graffiti Public", "trend": "DECLINING", "growth_pct": -35.0},
            {"category": "Encampment", "trend": "DECLINING", "growth_pct": -18.3},
            {"category": "Graffiti Private", "trend": "DECLINING", "growth_pct": -25.2},
            {"category": "Tree Maintenance", "trend": "DECLINING", "growth_pct": -18.3},
            {"category": "Blocked Street and Sidewalk", "trend": "DECLINING", "growth_pct": -14.4},
        ],
        "district_hotspots": [
            {"category": "Graffiti Public", "district": 9, "concentration": 25.3},
            {"category": "Encampment", "district": 6, "concentration": 33.4},
        ],
    }


def _get_events_by_date_dict():
    """Return {date_str: [event_names]} from events_by_date aggregate."""
    ensure_aggregates_loaded()
    df = None
    if AGGREGATES_CACHE and "events_by_date" in AGGREGATES_CACHE:
        df = AGGREGATES_CACHE["events_by_date"].copy()
    else:
        for base in [Path("/app") / "data" / "aggregates", Path(__file__).parent.parent / "data" / "aggregates"]:
            p = base / "events_by_date.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                break
    if df is None or df.empty or "event_names" not in df.columns:
        return {}
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["names"] = df["event_names"].fillna("").str.split("|").apply(lambda x: [n.strip() for n in x if n and n.strip()])
    return dict(zip(df["date"].astype(str), df["names"].tolist()))


@app.get("/api/analytics/events-by-date")
async def get_events_by_date(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Return event names per date for day-by-day annotations (e.g. 'Noise spike maybe due to: Super Bowl')."""
    events = _get_events_by_date_dict()
    if not events:
        return {"events_by_date": {}}
    dates = sorted(events.keys())
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]
    return {"events_by_date": {d: events[d] for d in dates}}


# ---------------------------------------------------------------------------
# Event correlation (DataSF: Our415 + Street Closures)
# ---------------------------------------------------------------------------

@app.get("/api/analytics/event-correlation")
async def get_event_correlation(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """
    Correlate 311 request volume with DataSF events (Our415 + Street Closures).
    Returns: event days vs non-event days volume, top categories on event days.
    """
    ensure_data_loaded()
    ensure_aggregates_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    # Get events_agg if available
    events_df = None
    if AGGREGATES_CACHE and "events_agg" in AGGREGATES_CACHE:
        events_df = AGGREGATES_CACHE["events_agg"].copy()
    else:
        # Try loading from local
        for base in [Path("/app") / "data" / "aggregates", Path(__file__).parent.parent / "data" / "aggregates"]:
            p = base / "events_agg.parquet"
            if p.exists():
                events_df = pd.read_parquet(p)
                break

    if events_df is None or events_df.empty:
        return {
            "message": "Event data not loaded. Run: python scripts/fetch_events.py --days 365",
            "event_days_avg_requests": None,
            "non_event_days_avg_requests": None,
            "lift_percent": None,
            "top_categories_on_event_days": [],
        }

    # Build daily 311 counts
    df = DATA_CACHE.copy()
    df = _filter_date_range(df, start_date, end_date)
    df["date"] = pd.to_datetime(df["opened"]).dt.date
    daily_311 = df.groupby("date").agg(
        total_requests=("category", "size"),
        categories=("category", lambda x: x.value_counts().to_dict()),
    ).reset_index()

    # Merge with events (include construction_count if present)
    events_df["date"] = pd.to_datetime(events_df["date"], errors="coerce").dt.date
    ec = events_df["event_count"].fillna(0)
    sc = events_df["street_closure_count"].fillna(0)
    cc = events_df["construction_count"].fillna(0) if "construction_count" in events_df.columns else 0
    events_df["has_events"] = (ec + sc + cc) > 0
    merge_cols = ["date", "event_count", "street_closure_count", "has_events"]
    if "construction_count" in events_df.columns:
        merge_cols.append("construction_count")
    merged = daily_311.merge(events_df[merge_cols], on="date", how="left")
    merged["has_events"] = merged["has_events"].fillna(False)

    event_days = merged[merged["has_events"]]
    non_event_days = merged[~merged["has_events"]]

    event_avg = float(event_days["total_requests"].mean()) if len(event_days) > 0 else None
    non_event_avg = float(non_event_days["total_requests"].mean()) if len(non_event_days) > 0 else None
    lift = None
    if event_avg and non_event_avg and non_event_avg > 0:
        lift = round((event_avg - non_event_avg) / non_event_avg * 100, 1)

    # Top categories on event days (aggregate category counts)
    cat_totals = {}
    for cats in event_days["categories"].dropna():
        for c, cnt in cats.items():
            cat_totals[c] = cat_totals.get(c, 0) + cnt
    top_on_events = sorted(cat_totals.items(), key=lambda x: -x[1])[:10]

    return {
        "event_days_avg_requests": round(event_avg, 1) if event_avg else None,
        "non_event_days_avg_requests": round(non_event_avg, 1) if non_event_avg else None,
        "lift_percent": lift,
        "event_days_count": int(len(event_days)),
        "non_event_days_count": int(len(non_event_days)),
        "top_categories_on_event_days": [{"category": c, "count": int(n)} for c, n in top_on_events],
    }


@app.get("/api/analytics/correlation")
async def get_correlation(
    dataset: str = Query("events", description="Secondary dataset: events, crime"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """
    Correlate 311 volume with another dataset (events or crime).
    For crime: returns daily 311 vs daily incident count (citywide or by district).
    """
    ensure_data_loaded()
    ensure_aggregates_loaded()
    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()
    df = _filter_date_range(df, start_date, end_date)
    df["date"] = pd.to_datetime(df["opened"]).dt.date
    daily_311 = df.groupby("date").size().reset_index(name="total_requests")

    if dataset == "crime" and AGGREGATES_CACHE and "crime_agg" in AGGREGATES_CACHE:
        crime_df = AGGREGATES_CACHE["crime_agg"].copy()
        crime_df["date"] = pd.to_datetime(crime_df["date"], errors="coerce").dt.date
        if "district" in crime_df.columns and crime_df["district"].notna().any():
            crime_daily = crime_df.groupby("date")["count"].sum().reset_index(name="incident_count")
        else:
            crime_daily = crime_df.groupby("date")["count"].sum().reset_index(name="incident_count")
        merged = daily_311.merge(crime_daily, on="date", how="inner")
        if merged.empty or len(merged) < 2:
            return {"message": "Insufficient crime/311 overlap", "correlation": None, "dataset": "crime"}
        corr = float(merged["total_requests"].corr(merged["incident_count"]))
        return {
            "dataset": "crime",
            "correlation": round(corr, 4),
            "days_matched": int(len(merged)),
            "avg_311": round(float(merged["total_requests"].mean()), 1),
            "avg_incidents": round(float(merged["incident_count"].mean()), 1),
        }
    if dataset == "events":
        return await get_event_correlation(start_date=start_date, end_date=end_date)
    raise HTTPException(status_code=400, detail="dataset must be 'events' or 'crime'")


def _get_weather_by_date_dict():
    """Return {date_str: {precip_mm, temp_max_c, rain_day}} from weather_agg (JSON-serializable)."""
    def _sanitize(row):
        out = {}
        for k, v in row.items():
            if pd.isna(v):
                out[k] = None
            elif isinstance(v, (np.integer, np.floating)):
                out[k] = float(v)
            else:
                out[k] = v
        return out

    ensure_aggregates_loaded()
    if not AGGREGATES_CACHE or "weather_agg" not in AGGREGATES_CACHE:
        for base in [Path("/app") / "data" / "aggregates", Path(__file__).parent.parent / "data" / "aggregates"]:
            p = base / "weather_agg.parquet"
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    return {str(d): _sanitize(row) for d, row in df.set_index("date").to_dict("index").items()}
                except Exception:
                    pass
        return {}
    df = AGGREGATES_CACHE["weather_agg"].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return {str(d): _sanitize(row) for d, row in df.set_index("date").to_dict("index").items()}


@app.get("/api/data/weather")
async def get_weather(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Return daily weather for SF (precip, temp) for chart annotations."""
    weather = _get_weather_by_date_dict()
    if not weather:
        return {"weather_by_date": {}, "message": "Run scripts/fetch_weather.py --days 365"}
    dates = sorted(weather.keys())
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]
    return {"weather_by_date": {d: weather[d] for d in dates}}


@app.get("/api/data/freshness")
async def get_freshness():
    """Return last-updated dates for 311, events, crime, weather (for dashboard badge)."""
    ensure_aggregates_loaded()
    out = {"311": None, "events": None, "crime": None, "weather": None}
    for name, key in [("311", "daily_agg"), ("events", "events_agg"), ("crime", "crime_agg"), ("weather", "weather_agg")]:
        if AGGREGATES_CACHE and key in AGGREGATES_CACHE:
            df = AGGREGATES_CACHE[key]
            if "date" in df.columns and not df.empty:
                out[name] = str(pd.to_datetime(df["date"]).max().date()) if pd.notna(df["date"].max()) else None
    return out


# ---------------------------------------------------------------------------
# NEW endpoints
# ---------------------------------------------------------------------------

@app.get("/api/analytics/district-breakdown")
async def get_district_breakdown():
    """Compute district breakdown from the FULL dataset."""
    ensure_data_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()
    df['supervisor_district'] = pd.to_numeric(df['supervisor_district'], errors='coerce')
    df = df[pd.notna(df['supervisor_district'])]
    df['supervisor_district'] = df['supervisor_district'].astype(int)

    total = len(df)
    district_counts = df['supervisor_district'].value_counts().sort_index()

    breakdown = []
    for district_num, count in district_counts.items():
        breakdown.append({
            'district': int(district_num),
            'name': DISTRICT_NAMES.get(int(district_num), f'District {district_num}'),
            'count': int(count),
            'percentage': round(count / total * 100, 2),
        })

    # Sort by count descending
    breakdown.sort(key=lambda x: x['count'], reverse=True)

    return {
        'total_with_district': total,
        'districts': breakdown,
    }


@app.get("/api/analytics/resolution-time")
async def get_resolution_time():
    """Compute average resolution time by category."""
    ensure_data_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()

    # Ensure datetime types
    df['opened'] = pd.to_datetime(df['opened'], errors='coerce')
    df['closed'] = pd.to_datetime(df['closed'], errors='coerce')

    # Only rows with both opened and closed
    df = df[pd.notna(df['opened']) & pd.notna(df['closed'])]
    df['resolution_hours'] = (df['closed'] - df['opened']).dt.total_seconds() / 3600
    # Exclude negative or absurdly large resolution times
    df = df[(df['resolution_hours'] >= 0) & (df['resolution_hours'] < 8760)]  # < 1 year

    if df.empty:
        return {'overall_avg_hours': None, 'overall_median_hours': None, 'categories': []}

    overall_avg = float(df['resolution_hours'].mean())
    overall_median = float(df['resolution_hours'].median())

    by_cat = df.groupby('category')['resolution_hours'].agg(['mean', 'median', 'count'])
    by_cat = by_cat.rename(columns={'mean': 'avg_hours', 'median': 'median_hours'})
    by_cat = by_cat[by_cat['count'] >= 10]  # only categories with enough data
    by_cat = by_cat.sort_values('avg_hours', ascending=False)

    categories = []
    for cat_name, row in by_cat.iterrows():
        categories.append({
            'category': cat_name,
            'avg_hours': round(float(row['avg_hours']), 2),
            'median_hours': round(float(row['median_hours']), 2),
            'sample_size': int(row['count']),
        })

    return {
        'overall_avg_hours': round(overall_avg, 2),
        'overall_median_hours': round(overall_median, 2),
        'categories': categories,
    }


@app.get("/api/analytics/anomalies")
async def get_anomalies():
    """Detect anomalies in the last 7 days vs rolling 30-day average."""
    ensure_data_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()
    df['opened'] = pd.to_datetime(df['opened'], errors='coerce')
    df = df[pd.notna(df['opened'])]
    df['date'] = df['opened'].dt.date

    max_date = df['date'].max()
    # Last 7 days
    recent_start = max_date - timedelta(days=6)
    # Need 30 days of history before that for rolling average
    history_start = recent_start - timedelta(days=30)

    # Daily counts by category
    daily_cat = df[df['date'] >= history_start].groupby(['date', 'category']).size().reset_index(name='count')

    anomalies = []
    recent_dates = [recent_start + timedelta(days=i) for i in range(7)]
    recent_dates = [d for d in recent_dates if d <= max_date]

    # Get categories with enough data
    top_categories = df['category'].value_counts().head(20).index.tolist()

    for cat in top_categories:
        cat_daily = daily_cat[daily_cat['category'] == cat].set_index('date')['count']

        for check_date in recent_dates:
            # Rolling 30-day window ending the day before check_date
            window_start = check_date - timedelta(days=30)
            window_end = check_date - timedelta(days=1)

            window_vals = []
            current = window_start
            while current <= window_end:
                window_vals.append(cat_daily.get(current, 0))
                current += timedelta(days=1)

            if len(window_vals) < 7:
                continue

            window_arr = np.array(window_vals, dtype=float)
            avg = window_arr.mean()
            std = window_arr.std()

            if std == 0:
                continue

            actual = cat_daily.get(check_date, 0)
            deviation = (actual - avg) / std

            if deviation > 2.0:
                events_map = _get_events_by_date_dict()
                day_events = events_map.get(str(check_date), [])
                possible_cause = None
                if day_events:
                    possible_cause = "Maybe due to: " + ", ".join(day_events[:3])
                    if len(day_events) > 3:
                        possible_cause += "..."
                anomalies.append({
                    'date': str(check_date),
                    'category': cat,
                    'actual_count': int(actual),
                    'expected_count': round(float(avg), 1),
                    'deviation_factor': round(float(deviation), 2),
                    'possible_event_cause': possible_cause,
                })

    # Sort by deviation factor descending
    anomalies.sort(key=lambda x: x['deviation_factor'], reverse=True)

    return {
        'period': {'start': str(recent_start), 'end': str(max_date)},
        'anomalies': anomalies,
    }


@app.get("/api/analytics/map-data")
async def get_map_data(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    format: str = Query("points", description="points or heatmap"),
    category: Optional[str] = Query(None, description="Filter by category (drill-down)"),
    district: Optional[int] = Query(None, description="Filter by district (drill-down)"),
):
    """Return lat/lon data grouped by category. Supports points or heatmap (grid)."""
    days = _days_in_range(start_date, end_date)
    use_heatmap = format == "heatmap" or (days is not None and days > RAW_CUTOFF_DAYS)

    # Try heatmap from grid_agg for long ranges
    if use_heatmap:
        ensure_aggregates_loaded()
        if AGGREGATES_CACHE and 'grid_agg' in AGGREGATES_CACHE:
            grid = AGGREGATES_CACHE['grid_agg'].copy()
            grid = _filter_agg_date_range(grid, 'year_month', start_date, end_date)
            if category:
                grid = grid[grid['category'] == category]
            if not grid.empty:
                top_cats = grid['category'].value_counts().head(10).index.tolist()
                color_map = {c: CATEGORY_COLORS[i] for i, c in enumerate(top_cats)}
                result = []
                for cat in top_cats:
                    g = grid[grid['category'] == cat]
                    cells = [
                        {'lat': float(r['lat_cell']), 'lon': float(r['lon_cell']), 'count': int(r['count'])}
                        for _, r in g.iterrows()
                    ]
                    result.append({
                        'category': cat,
                        'cells': cells,
                        'count': int(g['count'].sum()),
                        'color': color_map[cat],
                        'format': 'heatmap',
                    })
                return {'categories': result, 'total_points': int(grid['count'].sum()), 'format': 'heatmap'}

    # Use raw/recent data for points
    ensure_data_loaded()
    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()
    df = _filter_date_range(df, start_date, end_date)
    if category:
        df = df[df['category'] == category]
    if district is not None:
        dist_col = 'supervisor_district' if 'supervisor_district' in df.columns else 'district'
        if dist_col in df.columns:
            df['district_num'] = pd.to_numeric(df[dist_col], errors='coerce')
            df = df[df['district_num'] == district]

    lat_col, lon_col = None, None
    if 'lat' in df.columns and 'long' in df.columns:
        lat_col, lon_col = 'lat', 'long'
    elif 'lat' in df.columns and 'lon' in df.columns:
        lat_col, lon_col = 'lat', 'lon'
    elif 'latitude' in df.columns and 'longitude' in df.columns:
        lat_col, lon_col = 'latitude', 'longitude'
    elif 'point' in df.columns:
        df['latitude'] = df['point'].apply(
            lambda x: float(x['coordinates'][1]) if isinstance(x, dict) and x and 'coordinates' in x else np.nan
        )
        df['longitude'] = df['point'].apply(
            lambda x: float(x['coordinates'][0]) if isinstance(x, dict) and x and 'coordinates' in x else np.nan
        )
        lat_col, lon_col = 'latitude', 'longitude'
    else:
        return {'categories': [], 'total_points': 0, 'message': 'No coordinate columns found'}

    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df[pd.notna(df[lat_col]) & pd.notna(df[lon_col])]
    df = df[(df[lat_col] > 37.5) & (df[lat_col] < 38.0) &
            (df[lon_col] > -123.0) & (df[lon_col] < -122.0)]

    if df.empty:
        return {'categories': [], 'total_points': 0}

    top_cats = df['category'].value_counts().head(10).index.tolist()
    color_map = {cat: CATEGORY_COLORS[i] for i, cat in enumerate(top_cats)}

    result = []
    total_points = 0
    for cat in top_cats:
        cat_df = df[df['category'] == cat]
        points = [
            {'lat': float(row[lat_col]), 'lon': float(row[lon_col])}
            for _, row in cat_df.iterrows()
        ]
        total_points += len(points)
        result.append({
            'category': cat,
            'points': points,
            'count': len(points),
            'color': color_map[cat],
        })

    return {'categories': result, 'total_points': total_points, 'format': 'points'}


@app.get("/api/analytics/compare-periods")
async def get_compare_periods(
    base_start: str = Query(..., description="Base period start YYYY-MM-DD"),
    base_end: str = Query(..., description="Base period end YYYY-MM-DD"),
    compare_start: Optional[str] = Query(None, description="Compare period start (e.g. same month last year)"),
    compare_end: Optional[str] = Query(None, description="Compare period end"),
    category: Optional[str] = Query(None, description="Filter by category"),
):
    """Compare two time periods (e.g. this month vs same month last year)."""
    ensure_data_loaded()
    ensure_aggregates_loaded()

    def get_totals(sd: str, ed: str) -> int:
        days = (pd.to_datetime(ed) - pd.to_datetime(sd)).days
        if AGGREGATES_CACHE and 'daily_agg' in AGGREGATES_CACHE and days > RAW_CUTOFF_DAYS:
            agg = AGGREGATES_CACHE['daily_agg'].copy()
            agg = _filter_agg_date_range(agg, 'date', sd, ed)
            if category:
                agg = agg[agg['category'] == category]
            return int(agg['count'].sum())
        if DATA_CACHE is None or DATA_CACHE.empty:
            return 0
        df = DATA_CACHE.copy()
        df = _filter_date_range(df, sd, ed)
        if category:
            df = df[df['category'] == category]
        return len(df)

    base_total = get_totals(base_start, base_end)
    compare_total = 0
    if compare_start and compare_end:
        compare_total = get_totals(compare_start, compare_end)

    pct_change = None
    if compare_total and compare_total > 0:
        pct_change = round((base_total - compare_total) / compare_total * 100, 1)

    return {
        'base_period': {'start': base_start, 'end': base_end, 'total': base_total},
        'compare_period': {'start': compare_start, 'end': compare_end, 'total': compare_total} if compare_start and compare_end else None,
        'change_percent': pct_change,
        'category': category,
    }


@app.get("/api/analytics/category-timeseries")
async def get_category_timeseries(
    category: str = Query(..., description="Category name"),
):
    """Return daily counts for a single category."""
    ensure_data_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()
    df = df[df['category'] == category]

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for category: {category}")

    df['date'] = pd.to_datetime(df['opened']).dt.date
    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_counts = daily_counts.sort_values('date')

    return {
        'category': category,
        'dates': [str(d) for d in daily_counts['date'].tolist()],
        'counts': daily_counts['count'].tolist(),
        'total_days': len(daily_counts),
        'total_count': int(daily_counts['count'].sum()),
    }


# ---------------------------------------------------------------------------
# Frontend static files
# ---------------------------------------------------------------------------

frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    logger.info(f"Serving frontend from {frontend_path}")

    @app.get("/")
    async def serve_frontend():
        """Serve the frontend dashboard"""
        index_file = frontend_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {
            "service": "SF 311 Predictor API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
