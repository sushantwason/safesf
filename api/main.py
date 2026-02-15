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


# ---------------------------------------------------------------------------
# Data / model loading helpers
# ---------------------------------------------------------------------------

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
async def export_data():
    """Export the full dataset as a CSV download."""
    ensure_data_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    def _iter_csv():
        buf = io.StringIO()
        DATA_CACHE.to_csv(buf, index=False)
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
    """Get daily aggregated request counts (cached when no date filter)."""
    # Use cache only when no date filters are applied
    if start_date is None and end_date is None:
        ensure_analytics_loaded()
        if ANALYTICS_CACHE and 'daily_timeseries' in ANALYTICS_CACHE:
            return ANALYTICS_CACHE['daily_timeseries']

    ensure_data_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()
    df = _filter_date_range(df, start_date, end_date)
    df['date'] = pd.to_datetime(df['opened']).dt.date

    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_counts = daily_counts.sort_values('date')

    return {
        'dates': [str(d) for d in daily_counts['date'].tolist()],
        'counts': daily_counts['count'].tolist(),
        'total_days': len(daily_counts),
    }


@app.get("/api/analytics/day-of-week")
async def get_day_of_week_analytics(
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Get request patterns by day of week (cached when no date filter)."""
    if start_date is None and end_date is None:
        ensure_analytics_loaded()
        if ANALYTICS_CACHE and 'day_of_week' in ANALYTICS_CACHE:
            return ANALYTICS_CACHE['day_of_week']

    ensure_data_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()
    df = _filter_date_range(df, start_date, end_date)
    df['day_of_week'] = pd.to_datetime(df['opened']).dt.day_name()

    dow_counts = df['day_of_week'].value_counts()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    return {
        'days': [d for d in dow_order if d in dow_counts.index],
        'counts': [int(dow_counts[d]) for d in dow_order if d in dow_counts.index],
        'percentages': [round(dow_counts[d] / len(df) * 100, 1) for d in dow_order if d in dow_counts.index],
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
    """Get monthly request volume trends (cached when no date filter)."""
    if start_date is None and end_date is None:
        ensure_analytics_loaded()
        if ANALYTICS_CACHE and 'monthly_trends' in ANALYTICS_CACHE:
            return ANALYTICS_CACHE['monthly_trends']

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
                anomalies.append({
                    'date': str(check_date),
                    'category': cat,
                    'actual_count': int(actual),
                    'expected_count': round(float(avg), 1),
                    'deviation_factor': round(float(deviation), 2),
                })

    # Sort by deviation factor descending
    anomalies.sort(key=lambda x: x['deviation_factor'], reverse=True)

    return {
        'period': {'start': str(recent_start), 'end': str(max_date)},
        'anomalies': anomalies,
    }


@app.get("/api/analytics/map-data")
async def get_map_data():
    """Return lat/lon data grouped by category for the interactive map."""
    ensure_data_loaded()

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    df = DATA_CACHE.copy()

    # Determine lat/lon column names.
    # The parquet has 'lat'/'long' (string, populated) and 'latitude'/'longitude' (empty).
    lat_col = None
    lon_col = None
    if 'lat' in df.columns and 'long' in df.columns:
        lat_col, lon_col = 'lat', 'long'
    elif 'lat' in df.columns and 'lon' in df.columns:
        lat_col, lon_col = 'lat', 'lon'
    elif 'latitude' in df.columns and 'longitude' in df.columns:
        lat_col, lon_col = 'latitude', 'longitude'
    elif 'point' in df.columns:
        # Extract from nested point column
        df['latitude'] = df['point'].apply(
            lambda x: float(x['coordinates'][1]) if isinstance(x, dict) and 'coordinates' in x else None
        )
        df['longitude'] = df['point'].apply(
            lambda x: float(x['coordinates'][0]) if isinstance(x, dict) and 'coordinates' in x else None
        )
        lat_col, lon_col = 'latitude', 'longitude'
    else:
        return {'categories': [], 'total_points': 0, 'message': 'No coordinate columns found'}

    # Convert to numeric and filter out NaN
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df[pd.notna(df[lat_col]) & pd.notna(df[lon_col])]

    # Basic sanity filter for SF area
    df = df[(df[lat_col] > 37.5) & (df[lat_col] < 38.0) &
            (df[lon_col] > -123.0) & (df[lon_col] < -122.0)]

    if df.empty:
        return {'categories': [], 'total_points': 0}

    # Top 10 categories by count
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

    return {'categories': result, 'total_points': total_points}


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
