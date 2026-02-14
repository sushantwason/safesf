"""
Tests for SF 311 Predictor API
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "SF 311 Predictor API"


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data
    assert "categories_available" in data


def test_get_categories():
    """Test categories endpoint"""
    response = client.get("/api/categories")
    assert response.status_code in [200, 503]  # 503 if no models loaded
    if response.status_code == 200:
        data = response.json()
        assert "categories" in data
        assert "total" in data


def test_predict_volume_missing_category():
    """Test prediction with missing category"""
    response = client.get("/api/predict/volume?days=7")
    assert response.status_code == 422  # Validation error


def test_predict_trends():
    """Test trends endpoint"""
    response = client.get("/api/predict/trends?category=all&timeframe=30d")
    assert response.status_code == 200
    data = response.json()
    assert "timeframe" in data
    assert "trends" in data


def test_predict_hotspots():
    """Test hotspots endpoint"""
    response = client.get("/api/predict/hotspots?top_n=5")
    assert response.status_code == 200
    data = response.json()
    assert "date" in data
    assert "hotspots" in data
    assert len(data["hotspots"]) <= 5


def test_data_stats():
    """Test data stats endpoint"""
    response = client.get("/api/data/stats")
    assert response.status_code == 200
    data = response.json()
    assert "models_trained" in data
    assert "categories" in data


def test_metrics():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models_loaded" in data
    assert "categories_available" in data


def test_invalid_endpoint():
    """Test invalid endpoint returns 404"""
    response = client.get("/api/invalid")
    assert response.status_code == 404


@pytest.mark.parametrize("days", [1, 7, 15, 30])
def test_predict_volume_different_days(days):
    """Test prediction with different day values"""
    # This will fail if no models are loaded, which is expected in CI
    response = client.get(f"/api/predict/volume?category=TestCategory&days={days}")
    assert response.status_code in [200, 404]  # 404 if category doesn't exist


@pytest.mark.parametrize("timeframe", ["7d", "30d", "90d"])
def test_predict_trends_different_timeframes(timeframe):
    """Test trends with different timeframes"""
    response = client.get(f"/api/predict/trends?category=all&timeframe={timeframe}")
    assert response.status_code == 200
    data = response.json()
    assert data["timeframe"] == timeframe
