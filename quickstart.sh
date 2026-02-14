#!/bin/bash

# Quick Start Script for SF 311 Predictor
# This script sets up the environment and fetches initial data

set -e

echo "🚀 SF 311 Predictor - Quick Start"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "📝 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    echo ""
    echo "📄 Creating .env file..."
    cp .env.example .env
    echo "${YELLOW}⚠️  Please edit .env and add your SF Open Data API token${NC}"
else
    echo ""
    echo "${GREEN}✓ .env file already exists${NC}"
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p data models logs

# Fetch sample data (last 30 days for quick start)
echo ""
echo "📊 Fetching sample 311 data (last 30 days)..."
end_date=$(date +%Y-%m-%d)
start_date=$(date -d "30 days ago" +%Y-%m-%d 2>/dev/null || date -v-30d +%Y-%m-%d)

python scripts/fetch_data.py \
    --start-date "$start_date" \
    --end-date "$end_date" \
    --output data/311_sample.parquet \
    --limit 50000

# Train models on sample data
echo ""
echo "🤖 Training ML models on sample data..."
python scripts/train_model.py \
    --data data/311_sample.parquet \
    --output models

# Test API
echo ""
echo "🧪 Starting API for testing..."
echo "${YELLOW}Starting server on http://localhost:8000${NC}"
echo "Press Ctrl+C to stop"
echo ""
echo "Once server starts, visit:"
echo "  - API docs: http://localhost:8000/docs"
echo "  - Dashboard: http://localhost:8000/frontend/index.html"
echo "  - Health: http://localhost:8000/health"
echo ""

cd api
uvicorn main:app --host 0.0.0.0 --port 8000
