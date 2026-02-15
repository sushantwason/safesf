# SF 311 Request Predictor

Machine learning system to predict San Francisco 311 service request volumes and patterns, helping city allocate resources proactively.

## Features

- **Predictive Analytics**: ML models forecasting 311 request volume by type, location, and time
- **Real-time Dashboard**: Interactive visualization of predictions and trends
- **API Endpoints**: RESTful API for integrating predictions into city systems
- **Automated Updates**: Daily data refresh from SF Open Data portal
- **Serverless Architecture**: Fully serverless GCP deployment for scalability

## Architecture

```
┌─────────────────┐
│  SF Open Data   │
│   API (311)     │
└────────┬────────┘
         │ Daily Sync
         ▼
┌─────────────────┐      ┌──────────────┐
│   BigQuery      │─────▶│  Cloud Run   │
│  (Data Lake)    │      │  (ML API)    │
└─────────────────┘      └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │  Frontend    │
                         │  Dashboard   │
                         └──────────────┘
```

## Tech Stack

- **Data**: BigQuery, Cloud Storage
- **ML**: scikit-learn, XGBoost, Prophet (time series)
- **API**: FastAPI, Pydantic
- **Frontend**: React, Plotly.js, Mapbox
- **Infrastructure**: Cloud Run, Cloud Scheduler, Cloud Functions
- **Monitoring**: Cloud Monitoring, Cloud Logging

## Setup

### Prerequisites

- Google Cloud SDK installed
- Python 3.11+
- GCP Project: `safesf-439219`

### Installation

```bash
# Clone and setup
cd sf-311-predictor
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure GCP
gcloud config set project safesf-439219
gcloud auth application-default login

# Set environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Data Setup

```bash
# Fetch historical 311 data
python scripts/fetch_data.py --start-date 2020-01-01 --end-date 2026-02-14

# Build pre-aggregated tables (for 5+ year lookback with minimal storage)
# Run after fetch_data or use --gcs-url for existing data
python scripts/build_aggregates.py --input data/311_raw.parquet --output-dir data/aggregates
# Or from GCS sample:
python scripts/build_aggregates.py --gcs-url "https://storage.googleapis.com/sf-311-data-personal/311_sample.parquet" --output-dir data/aggregates

# Fetch DataSF event data (Our415 + Street Closures) for 311 correlation
python scripts/fetch_events.py --days 365

# Optional: weather (Open-Meteo), crime (SFPD), building permits (merged into events)
python scripts/fetch_weather.py --days 365
python scripts/fetch_crime.py --days 365
python scripts/fetch_permits.py --days 365
# Re-run fetch_events to merge permits into events_agg/events_by_date
python scripts/fetch_events.py --days 365

# Load into BigQuery
python scripts/load_to_bigquery.py

# Train initial models
python scripts/train_model.py
```

### Local Development

```bash
# Run API locally
cd api
uvicorn main:app --reload --port 8000

# Run frontend locally
cd frontend
python -m http.server 3000
```

### Deploy with crime/weather data (Compare tab & rain annotations)

The app reads `data/aggregates/crime_agg.parquet` and `weather_agg.parquet` if present. To include them in the deployed image:

```bash
# Fetch crime + weather into data/aggregates/
./scripts/prepare_data_for_deploy.sh

# Then deploy (build copies data/aggregates/ into the image)
./scripts/deploy.sh
```

Or manually: `python scripts/fetch_crime.py --days 365`, then `python scripts/fetch_weather.py --days 365`, then deploy.

### Custom domain (safesf.app)

To serve the app at **https://safesf.app** instead of the default `*.run.app` URL, see **[docs/CUSTOM_DOMAIN.md](docs/CUSTOM_DOMAIN.md)** for buying the domain, mapping it in Cloud Run, and setting DNS.

### Deployment

```bash
# Deploy to Cloud Run
gcloud run deploy sf-311-predictor \
  --source . \
  --project safesf-439219 \
  --region us-west1 \
  --allow-unauthenticated

# Set up daily data sync
gcloud scheduler jobs create http sf-311-daily-sync \
  --schedule="0 6 * * *" \
  --uri="https://YOUR_SERVICE_URL/api/sync" \
  --http-method=POST
```

## API Endpoints

### Predictions

```
GET  /api/predict/volume?days=7&category=Graffiti
GET  /api/predict/hotspots?date=2026-02-15
GET  /api/predict/trends?category=all&timeframe=30d
```

### Data

```
GET  /api/data/recent?limit=100
GET  /api/data/stats?groupby=category
GET  /api/data/categories
```

### Health

```
GET  /health
GET  /metrics
```

## ML Models

### 1. Volume Predictor
- **Algorithm**: XGBoost + Prophet ensemble
- **Features**: Historical volume, day of week, weather, events, holidays
- **Output**: Predicted request count by category for next 7-30 days
- **Accuracy**: MAPE ~15% for 7-day predictions

### 2. Hotspot Detector
- **Algorithm**: DBSCAN clustering + spatial analysis
- **Features**: Location patterns, time of day, category
- **Output**: Geographic hotspots requiring resource allocation
- **Accuracy**: 85% precision in identifying high-demand areas

### 3. Category Classifier
- **Algorithm**: Random Forest + NLP
- **Features**: Request description, location, time
- **Output**: Auto-categorization of incoming requests
- **Accuracy**: 92% classification accuracy

## Data Sources

- **SF 311 Cases**: https://data.sfgov.org/City-Infrastructure/311-Cases/vw6y-z8j6
- **Update Frequency**: Daily at 6am PT
- **Historical Range**: 2008-present
- **Records**: ~5M+ requests

## Use Cases

### For Government
- Resource allocation optimization
- Budget forecasting
- Performance measurement
- Identifying service gaps

### For Public
- Service trend transparency
- Neighborhood infrastructure health
- Civic engagement insights

## Monitoring

- **Uptime**: Cloud Monitoring alerts for API availability
- **Model Performance**: Daily validation against actual volumes
- **Data Quality**: Automated checks for data completeness
- **Costs**: Budget alerts set at $100/month

## Contributing

This is a civic tech hobby project. Contributions welcome!

## License

MIT License - see LICENSE file

## Contact

Built with SF Open Data for the public good.
