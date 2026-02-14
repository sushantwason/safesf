# SF 311 Predictor - Complete Setup Guide

Step-by-step guide to deploy the SF 311 Request Predictor to Google Cloud Platform.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **GCP Project**: `safesf-439219` (or create your own)
3. **Local Tools**:
   - Python 3.11+
   - Google Cloud SDK (gcloud CLI)
   - Git
   - Docker (optional, for local testing)

## Step 1: Initial Setup

### 1.1 Install Google Cloud SDK

```bash
# macOS
brew install google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Windows
# Download from: https://cloud.google.com/sdk/docs/install
```

### 1.2 Authenticate with GCP

```bash
# Login to GCP
gcloud auth login

# Set project
gcloud config set project safesf-439219

# Enable application default credentials
gcloud auth application-default login
```

### 1.3 Clone and Setup Project

```bash
# Navigate to project directory
cd sf-311-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### 1.4 Get SF Open Data API Token (Optional but Recommended)

1. Visit https://data.sfgov.org/
2. Sign up for a free account
3. Go to Profile > App Tokens
4. Create a new app token
5. Add to `.env` file:
   ```
   SF_OPEN_DATA_APP_TOKEN=your_token_here
   ```

## Step 2: Data Collection & Model Training

### 2.1 Fetch Historical 311 Data

```bash
# Fetch last 2 years of data (adjust dates as needed)
python scripts/fetch_data.py \
    --start-date 2024-01-01 \
    --end-date 2026-02-14 \
    --output data/311_raw.parquet

# This will take 5-15 minutes depending on data volume
```

Expected output:
- `data/311_raw.parquet` - Cleaned and processed 311 data
- Should contain 200k-500k records for 2 years

### 2.2 Train ML Models

```bash
# Train prediction models for top categories
python scripts/train_model.py \
    --data data/311_raw.parquet \
    --output models

# This will take 10-30 minutes depending on your machine
```

Expected output:
- `models/xgb_*.pkl` - XGBoost models for each category
- `models/prophet_*.pkl` - Prophet models for each category
- `models/metadata.json` - Model metadata and metrics

### 2.3 Test Locally (Optional)

```bash
# Run API locally
cd api
uvicorn main:app --reload --port 8000

# In another terminal, test
curl http://localhost:8000/health

# View dashboard
# Open browser to http://localhost:8000/frontend/index.html
```

## Step 3: Set Up GCP Infrastructure

### 3.1 Enable Required APIs

```bash
# Enable all necessary GCP APIs
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    bigquery.googleapis.com \
    storage-api.googleapis.com \
    cloudscheduler.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com
```

### 3.2 Set Up BigQuery

```bash
# Create BigQuery dataset and tables
python scripts/setup_bigquery.py
```

Expected output:
- Dataset: `safesf-439219.sf_311_data`
- Tables: `requests_raw`, `predictions`, `daily_aggregates`
- Views: `recent_requests_by_category`

### 3.3 Load Data to BigQuery (Optional)

```bash
# Load historical data to BigQuery
python scripts/load_to_bigquery.py \
    --data data/311_raw.parquet \
    --project safesf-439219 \
    --dataset sf_311_data \
    --table requests_raw
```

### 3.4 Create Cloud Storage Bucket for Models

```bash
# Create bucket for model storage
gsutil mb -p safesf-439219 -l us-west1 gs://sf-311-predictor-models

# Upload trained models
gsutil -m cp -r models/* gs://sf-311-predictor-models/
```

## Step 4: Deploy to Cloud Run

### 4.1 Quick Deploy (Automated)

```bash
# Make deploy script executable
chmod +x scripts/deploy.sh

# Run deployment
./scripts/deploy.sh
```

This script will:
1. Enable required APIs
2. Build container image
3. Deploy to Cloud Run
4. Set up Cloud Scheduler for daily updates
5. Output service URL

### 4.2 Manual Deploy (Alternative)

```bash
# Build and submit container
gcloud builds submit --tag gcr.io/safesf-439219/sf-311-predictor

# Deploy to Cloud Run
gcloud run deploy sf-311-predictor \
    --image gcr.io/safesf-439219/sf-311-predictor \
    --region us-west1 \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --set-env-vars "GCP_PROJECT_ID=safesf-439219,GCP_REGION=us-west1"

# Get service URL
gcloud run services describe sf-311-predictor \
    --region us-west1 \
    --format 'value(status.url)'
```

### 4.3 Set Up Automated Data Sync

```bash
# Get your Cloud Run service URL
SERVICE_URL=$(gcloud run services describe sf-311-predictor \
    --region us-west1 --format 'value(status.url)')

# Create Cloud Scheduler job for daily sync at 6am PT
gcloud scheduler jobs create http sf-311-daily-sync \
    --location us-west1 \
    --schedule "0 6 * * *" \
    --uri "${SERVICE_URL}/api/sync" \
    --http-method POST \
    --time-zone "America/Los_Angeles"
```

## Step 5: Verification & Testing

### 5.1 Test Deployed Service

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe sf-311-predictor \
    --region us-west1 --format 'value(status.url)')

# Test health endpoint
curl ${SERVICE_URL}/health

# Test categories
curl ${SERVICE_URL}/api/categories

# Test prediction (replace category with actual one)
curl "${SERVICE_URL}/api/predict/volume?category=Graffiti&days=7&model=prophet"
```

### 5.2 Access Dashboard

Visit your service URL in a browser:
```
https://sf-311-predictor-XXXXX-uw.a.run.app/frontend/index.html
```

## Step 6: Monitoring & Maintenance

### 6.1 View Logs

```bash
# Stream logs
gcloud logs tail --project safesf-439219 \
    --resource-names sf-311-predictor

# View in console
# https://console.cloud.google.com/logs
```

### 6.2 Set Up Alerts

```bash
# Create alert for high error rate
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="SF 311 Predictor High Error Rate" \
    --condition-display-name="Error rate > 5%" \
    --condition-threshold-value=5 \
    --condition-threshold-duration=300s
```

### 6.3 Monitor Costs

```bash
# Check current spend
gcloud billing accounts list

# Set budget alerts in GCP Console:
# https://console.cloud.google.com/billing/budgets
```

Recommended budget: $50-100/month for moderate usage

### 6.4 Update Models

To retrain models with new data:

```bash
# Fetch latest data
python scripts/fetch_data.py --start-date 2026-01-01 --end-date 2026-02-14

# Retrain models
python scripts/train_model.py --data data/311_raw.parquet

# Upload to GCS
gsutil -m cp -r models/* gs://sf-311-predictor-models/

# Redeploy service
./scripts/deploy.sh
```

## Troubleshooting

### Issue: "Permission denied" errors

**Solution**: Ensure you have necessary IAM roles:
```bash
gcloud projects add-iam-policy-binding safesf-439219 \
    --member="user:YOUR_EMAIL" \
    --role="roles/editor"
```

### Issue: "API not enabled" errors

**Solution**: Enable required APIs:
```bash
gcloud services enable SERVICE_NAME.googleapis.com
```

### Issue: Models not loading in Cloud Run

**Solution**:
1. Check if models directory is included in Docker image
2. Verify models uploaded to GCS
3. Check Cloud Run logs for errors

### Issue: BigQuery quota exceeded

**Solution**:
1. Check quota limits in GCP Console
2. Request quota increase if needed
3. Optimize queries to reduce data scanned

### Issue: Data fetch timing out

**Solution**:
1. Reduce date range for fetch
2. Use pagination with SF Open Data API
3. Run fetch in smaller batches

## Cost Optimization

### Expected Costs (Approximate)

- **Cloud Run**: $10-30/month (depends on traffic)
- **BigQuery**: $5-15/month (storage + queries)
- **Cloud Storage**: $1-5/month
- **Cloud Scheduler**: $0.10/month
- **Total**: $20-50/month for low-moderate usage

### Cost Reduction Tips

1. **Reduce Cloud Run minimum instances**: Set to 0 for dev
2. **Use BigQuery partitioning**: Already configured
3. **Set up budget alerts**: Get notified at $50, $75, $100
4. **Clean old data**: Delete old predictions after 90 days
5. **Optimize model size**: Use quantization or pruning

## Next Steps

1. **Customize Dashboard**: Edit `frontend/index.html` to add features
2. **Add Authentication**: Implement Firebase Auth or Cloud IAM
3. **Integrate with City Systems**: Use API for official dashboards
4. **Add More Features**:
   - Weather data integration
   - Event calendar correlation
   - Real-time notifications
5. **Share with Community**: Open source on GitHub

## Resources

- **SF Open Data Portal**: https://data.sfgov.org/
- **GCP Documentation**: https://cloud.google.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Prophet**: https://facebook.github.io/prophet/
- **XGBoost**: https://xgboost.readthedocs.io/

## Support

For issues or questions:
1. Check logs: `gcloud logs tail`
2. Review GCP Console dashboards
3. Test locally first
4. Check SF Open Data status page

---

**🎉 Congratulations! Your SF 311 Predictor is now live!**

Share your service URL with city officials, community groups, or fellow civic tech enthusiasts!
