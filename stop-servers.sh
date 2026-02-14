#!/bin/bash

# SF 311 Predictor - Stop Script
# This script stops both the API and Frontend servers

echo "🛑 Stopping SF 311 Predictor servers..."
echo ""

# Kill API server
echo "Stopping API server (port 8000)..."
pkill -f "uvicorn main:app" 2>/dev/null && echo "✓ API server stopped" || echo "API server not running"

# Kill Frontend server
echo "Stopping Frontend server (port 8080)..."
pkill -f "http.server 8080" 2>/dev/null && echo "✓ Frontend server stopped" || echo "Frontend server not running"

# Force kill any remaining processes on these ports
echo ""
echo "Cleaning up ports..."
lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:8080 2>/dev/null | xargs kill -9 2>/dev/null || true

echo ""
echo "✅ All servers stopped"
