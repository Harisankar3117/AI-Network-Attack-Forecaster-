import pytest
from fastapi.testclient import TestClient
from api.main import app
import json

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] == True
    assert data["scaler_loaded"] == True

def test_inference_wednesday():
    response = client.get("/api/v1/forecast/wednesday?k_steps=5")
    assert response.status_code == 200
    data = response.json()
    
    # Check core structure
    assert "scenario" in data
    assert "current_probability" in data
    assert "current_risk_level" in data
    assert "forecast_windows" in data
    assert "explainability" in data
    assert "mitre_context" in data
    assert "raw_features" in data
    assert "summary" in data

    assert data["scenario"] == "wednesday"
    assert len(data["forecast_windows"]) == 5
    
    # Verify probability bounds
    assert 0.0 <= data["current_probability"] <= 1.0
    
    for window in data["forecast_windows"]:
        assert 0.0 <= window["probability"] <= 1.0
        assert window["risk_level"] in ["Low", "Medium", "High"]

def test_inference_friday():
    response = client.get("/api/v1/forecast/friday?k_steps=3")
    assert response.status_code == 200
    data = response.json()
    assert data["scenario"] == "friday"
    assert len(data["forecast_windows"]) == 3

def test_benchmark():
    response = client.get("/api/v1/benchmark")
    assert response.status_code == 200
    data = response.json()
    
    assert "lstm_results" in data
    assert "Wednesday_validation" in data["lstm_results"]
    assert "Accuracy" in data["lstm_results"]["Wednesday_validation"]
    
def test_invalid_scenario():
    response = client.get("/api/v1/forecast/unknown_scenario")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
