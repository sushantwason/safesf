#!/bin/bash

# Deployment script for SF 311 Predictor to GCP Cloud Run

set -e

# Configuration
PROJECT_ID="safesf-439219"
REGION="us-west1"
SERVICE_NAME="sf-311-predictor"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying SF 311 Predictor to GCP Cloud Run"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found. Please install it first."
    exit 1
fi

# Set project
echo "📝 Setting GCP project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required GCP APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    bigquery.googleapis.com \
    storage-api.googleapis.com \
    cloudscheduler.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com

# Build container image
echo "🏗️  Building container image..."
gcloud builds submit --tag ${IMAGE_NAME}:latest .

# Deploy to Cloud Run
echo "☁️  Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --region ${REGION} \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0 \
    --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION}"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --region ${REGION} \
    --format 'value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo "📊 Dashboard: ${SERVICE_URL}/frontend/index.html"
echo "📖 API Docs: ${SERVICE_URL}/docs"
echo ""

# Set up Cloud Scheduler for daily data sync (optional; skip if job exists or creation fails)
echo "⏰ Setting up daily data sync (Cloud Scheduler)..."
if gcloud scheduler jobs describe sf-311-daily-sync --location "${REGION}" &>/dev/null; then
    echo "  Scheduler job sf-311-daily-sync already exists; skipping."
else
    # No OIDC/OAuth flags = unauthenticated POST (Cloud Run allows unauthenticated).
    gcloud scheduler jobs create http sf-311-daily-sync \
        --location "${REGION}" \
        --schedule "0 6 * * *" \
        --uri "${SERVICE_URL}/api/sync" \
        --http-method POST \
        --time-zone "America/Los_Angeles" \
        && echo "  Created sf-311-daily-sync." \
        || echo "⚠️  Scheduler creation failed (e.g. API or location). Create manually in Console if needed."
fi

echo ""
echo "🎉 All done! Your SF 311 Predictor is live!"
echo ""
echo "Next steps:"
echo "1. Visit ${SERVICE_URL} to view the dashboard"
echo "2. Set up BigQuery tables for data storage"
echo "3. Run initial data fetch and model training"
echo "4. Configure monitoring and alerts"
