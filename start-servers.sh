#!/bin/bash

# SF 311 Predictor - Start Script
# This script starts both the API and Frontend servers

set -e

echo "🚀 Starting SF 311 Predictor..."
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Kill any existing servers
echo "🧹 Cleaning up existing servers..."
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "http.server 8080" 2>/dev/null || true
lsof -ti:8000 -ti:8080 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 2

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run quickstart.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Start API server (port 8000)
echo ""
echo "🔌 Starting API server on http://localhost:8000..."
cd api
nohup uvicorn main:app --host 0.0.0.0 --port 8000 --reload > /tmp/sf311-api.log 2>&1 &
API_PID=$!
cd ..
sleep 3

# Check if API started
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API server started successfully${NC}"
else
    echo -e "${YELLOW}⚠️  API server might be starting up...${NC}"
fi

# Start Frontend server (port 8080)
echo ""
echo "🎨 Starting Frontend server on http://localhost:8080..."
nohup python3 -m http.server 8080 > /tmp/sf311-frontend.log 2>&1 &
FRONTEND_PID=$!
sleep 2

# Check if Frontend started
if curl -s http://localhost:8080/frontend/index.html > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Frontend server started successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Frontend server might be starting up...${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ SF 311 Predictor is running!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${BLUE}📱 Dashboard:${NC}      http://localhost:8080/frontend/index.html"
echo -e "${BLUE}🔧 API Docs:${NC}       http://localhost:8000/docs"
echo -e "${BLUE}❤️  Health Check:${NC}   http://localhost:8000/health"
echo ""
echo "📊 Data: 50,000 SF 311 requests loaded"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 Tips:"
echo "  - View logs: tail -f /tmp/sf311-api.log"
echo "  - Stop servers: ./stop-servers.sh"
echo ""
echo "🌐 Open your browser to: http://localhost:8080/frontend/index.html"
echo ""
