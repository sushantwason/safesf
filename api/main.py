"""
FastAPI backend for SF 311 Prediction Service
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SF 311 Predictor API",
    description="Predict San Francisco 311 service request volumes",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
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


# Global model cache
MODELS = {}
METADATA = {}
DATA_CACHE = None


def load_data(data_dir: str = "data"):
    """Load 311 data from parquet file on startup"""
    global DATA_CACHE

    # Try multiple possible locations
    possible_paths = [
        Path(data_dir),
        Path("/app") / data_dir,
        Path(__file__).parent.parent / data_dir
    ]

    data_files = []
    for data_path in possible_paths:
        if data_path.exists():
            logger.info(f"Checking {data_path} for data files...")
            files = list(data_path.glob("*.parquet"))
            if files:
                logger.info(f"Found {len(files)} parquet files in {data_path}")
                data_files.extend(files)
                break
        else:
            logger.warning(f"Path does not exist: {data_path}")

    if not data_files:
        logger.warning(f"No data files found in any location")
        return

    # Load the largest file (most data)
    data_file = max(data_files, key=lambda p: p.stat().st_size)

    try:
        DATA_CACHE = pd.read_parquet(data_file)
        logger.info(f"✅ Loaded {len(DATA_CACHE):,} records from {data_file.name}")
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

    # Load metadata
    metadata_path = models_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            METADATA = json.load(f)
        logger.info(f"Loaded metadata for {len(METADATA.get('categories', []))} categories")

    # Load XGBoost models
    for model_file in models_path.glob("xgb_*.pkl"):
        category = model_file.stem.replace("xgb_", "").replace("_", " ")
        try:
            MODELS[f"xgb_{category}"] = joblib.load(model_file)
            logger.info(f"Loaded XGBoost model for: {category}")
        except Exception as e:
            logger.error(f"Error loading {model_file}: {e}")

    # Load Prophet models
    for model_file in models_path.glob("prophet_*.pkl"):
        category = model_file.stem.replace("prophet_", "").replace("_", " ")
        try:
            MODELS[f"prophet_{category}"] = joblib.load(model_file)
            logger.info(f"Loaded Prophet model for: {category}")
        except Exception as e:
            logger.error(f"Error loading {model_file}: {e}")

    logger.info(f"Total models loaded: {len(MODELS)}")


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting SF 311 Predictor API...")
    load_data()
    load_models()
    logger.info("✅ Application ready!")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "SF 311 Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    categories = METADATA.get('categories', [])

    return HealthResponse(
        status="healthy" if MODELS else "no_models_loaded",
        models_loaded=len(MODELS),
        categories_available=categories,
        version="1.0.0"
    )


@app.get("/api/categories")
async def get_categories():
    """Get available categories"""
    global DATA_CACHE

    # Try to get from loaded data first
    if DATA_CACHE is not None and not DATA_CACHE.empty:
        categories = DATA_CACHE['category'].unique().tolist()
        category_counts = DATA_CACHE['category'].value_counts().to_dict()

        return {
            "categories": categories,
            "total": len(categories),
            "counts": category_counts
        }

    # Fallback to metadata
    categories = METADATA.get('categories', [])

    if not categories:
        # Return empty list instead of error
        return {
            "categories": [],
            "total": 0,
            "message": "No data loaded yet"
        }

    return {
        "categories": categories,
        "total": len(categories)
    }


@app.get("/api/predict/volume")
async def predict_volume(
    category: str = Query(..., description="311 request category"),
    days: int = Query(7, ge=1, le=30, description="Number of days to predict"),
    model: str = Query("xgboost", description="Model to use (xgboost or prophet)")
):
    """
    Predict 311 request volume for a category

    Returns daily predictions for the next N days
    """
    # Validate category
    available_categories = METADATA.get('categories', [])
    if category not in available_categories:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category}' not found. Available: {available_categories}"
        )

    # Get model
    model_key = f"{model}_{category}"
    if model_key not in MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found for category: {category}"
        )

    loaded_model = MODELS[model_key]

    # Generate predictions
    try:
        predictions = []
        today = datetime.now().date()

        if model == "prophet":
            # Prophet prediction
            future_dates = pd.DataFrame({
                'ds': pd.date_range(start=today, periods=days, freq='D')
            })
            forecast = loaded_model.predict(future_dates)

            for idx, row in forecast.iterrows():
                predictions.append({
                    'date': row['ds'].strftime('%Y-%m-%d'),
                    'predicted_volume': max(0, int(round(row['yhat']))),
                    'lower_bound': max(0, int(round(row['yhat_lower']))),
                    'upper_bound': max(0, int(round(row['yhat_upper'])))
                })

        else:
            # XGBoost prediction
            # For simplicity, using recent average patterns
            # In production, you'd fetch recent data to create proper features
            feature_columns = METADATA.get('feature_columns', [])

            for day_offset in range(days):
                pred_date = today + timedelta(days=day_offset)

                # Create dummy features (in production, use actual historical data)
                features = np.zeros(len(feature_columns))

                # Set temporal features
                features[0] = pred_date.year  # year
                features[1] = pred_date.month  # month
                features[2] = pred_date.day  # day
                features[3] = pred_date.weekday()  # day_of_week
                features[4] = pred_date.isocalendar()[1]  # week_of_year
                features[5] = 1 if pred_date.weekday() >= 5 else 0  # is_weekend

                # Predict
                volume = loaded_model.predict([features])[0]

                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'predicted_volume': max(0, int(round(volume))),
                    'confidence': 'medium'  # Placeholder
                })

        return PredictionResponse(
            category=category,
            predictions=predictions,
            model_used=model,
            generated_at=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/predict/trends")
async def predict_trends(
    category: str = Query("all", description="Category or 'all'"),
    timeframe: str = Query("30d", description="Timeframe (7d, 30d, 90d)")
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
            model = MODELS[f"prophet_{cat}"]
            today = datetime.now().date()

            future_dates = pd.DataFrame({
                'ds': pd.date_range(start=today, periods=days, freq='D')
            })
            forecast = model.predict(future_dates)

            # Calculate trend
            values = forecast['yhat'].values
            trend_direction = "increasing" if values[-1] > values[0] else "decreasing"
            trend_magnitude = abs((values[-1] - values[0]) / values[0] * 100)

            trends[cat] = {
                'direction': trend_direction,
                'magnitude_percent': round(trend_magnitude, 2),
                'predicted_total': int(values.sum()),
                'daily_average': int(values.mean())
            }

        except Exception as e:
            logger.error(f"Error calculating trend for {cat}: {e}")

    return {
        'timeframe': timeframe,
        'trends': trends,
        'generated_at': datetime.now().isoformat()
    }


@app.get("/api/predict/hotspots")
async def predict_hotspots(
    date: Optional[str] = Query(None, description="Date for prediction (YYYY-MM-DD)"),
    top_n: int = Query(10, description="Number of top hotspots to return")
):
    """
    Predict geographic hotspots for 311 requests

    Note: This is a simplified version. Production would use actual geospatial clustering.
    """
    # Placeholder implementation
    # In production, this would:
    # 1. Load historical location data
    # 2. Apply DBSCAN clustering
    # 3. Predict likely hotspot areas

    hotspots = [
        {
            'neighborhood': 'Tenderloin',
            'predicted_volume': 245,
            'latitude': 37.7841,
            'longitude': -122.4131,
            'top_categories': ['Street Cleaning', 'Graffiti', 'Illegal Dumping']
        },
        {
            'neighborhood': 'Mission',
            'predicted_volume': 198,
            'latitude': 37.7599,
            'longitude': -122.4148,
            'top_categories': ['Graffiti', 'Street Cleaning', 'Parking']
        },
        {
            'neighborhood': 'Financial District/South Beach',
            'predicted_volume': 156,
            'latitude': 37.7946,
            'longitude': -122.3999,
            'top_categories': ['Street Cleaning', 'Parking', 'Tree Maintenance']
        }
    ]

    return {
        'date': date or datetime.now().strftime('%Y-%m-%d'),
        'hotspots': hotspots[:top_n],
        'methodology': 'Historical pattern analysis',
        'generated_at': datetime.now().isoformat()
    }


@app.get("/api/data/recent")
async def get_recent_data(limit: int = Query(100, ge=1, le=1000)):
    """Get recent 311 service requests"""
    global DATA_CACHE

    if DATA_CACHE is None or DATA_CACHE.empty:
        raise HTTPException(status_code=503, detail="No data available")

    # Get most recent records
    df = DATA_CACHE.sort_values('opened', ascending=False).head(limit)

    records = []
    for _, row in df.iterrows():
        records.append({
            'id': row.get('service_request_id', 'N/A'),
            'category': row.get('category', 'Unknown'),
            'status': row.get('status_description', 'Unknown'),
            'opened': row['opened'].isoformat() if pd.notna(row.get('opened')) else None,
            'closed': row['closed'].isoformat() if pd.notna(row.get('closed')) else None,
            'address': row.get('address', 'N/A'),
            'district': row.get('supervisor_district', 'N/A'),
            'latitude': float(row['latitude']) if pd.notna(row.get('latitude')) else None,
            'longitude': float(row['longitude']) if pd.notna(row.get('longitude')) else None,
        })

    return {
        'total': len(records),
        'limit': limit,
        'data': records
    }


@app.get("/api/data/stats")
async def get_data_stats():
    """Get statistics about the underlying data"""
    global DATA_CACHE

    if DATA_CACHE is None or DATA_CACHE.empty:
        # Return stats from metadata if no data loaded
        metrics = METADATA.get('metrics', {})
        return {
            'models_trained': len(metrics),
            'categories': [],
            'total_requests': 0,
            'date_range': {}
        }

    # Calculate stats from actual data
    total = len(DATA_CACHE)
    categories = DATA_CACHE['category'].value_counts().to_dict()

    date_range = {
        'start': DATA_CACHE['opened'].min().isoformat() if pd.notna(DATA_CACHE['opened'].min()) else None,
        'end': DATA_CACHE['opened'].max().isoformat() if pd.notna(DATA_CACHE['opened'].max()) else None
    }

    # Calculate daily average
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
        'top_categories': dict(list(categories.items())[:10])
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus-style metrics endpoint"""
    return {
        "models_loaded": len(MODELS),
        "categories_available": len(METADATA.get('categories', [])),
        "uptime_seconds": 0  # Placeholder
    }


# Mount static files for frontend
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
            "health": "/health"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
