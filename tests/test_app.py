import sys
import os
import json
import io
from pathlib import Path
import pytest
import pandas as pd

# Set test environment
os.environ["DB_TYPE"] = "sqlite"
os.environ["FLASK_DEBUG"] = "0"

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import create_app
from database.db import db, Accident, Prediction, User
from ml.predict import load_prediction_artifacts, predict_accident_risk
from utils.helpers import validate_accident_csv

@pytest.fixture
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"
    })

    with app.app_context():
        db.create_all()
        # Seed a test accident record
        test_acc = Accident(
            accident_date=pd.to_datetime("2024-05-15").date(),
            accident_time="18:30",
            latitude=12.9176,
            longitude=77.6234,
            location="Silk Board Junction",
            weather="Rainy",
            road_condition="Wet",
            road_type="Major Intersection",
            traffic_volume=950,
            vehicle_count=2,
            visibility=2.0,
            speed_limit=50,
            previous_accidents=6,
            severity="Severe",
            risk_level="High"
        )
        db.session.add(test_acc)
        db.session.commit()

        yield app

        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

def test_database_connection(app):
    """1. Tests database connection and query execution."""
    with app.app_context():
        count = Accident.query.count()
        assert count >= 1
        user_count = User.query.count()
        assert user_count >= 1

def test_model_loading():
    """2. Tests loading of trained Random Forest model and preprocessor."""
    model, preprocessor, metrics = load_prediction_artifacts()
    assert model is not None
    assert preprocessor is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

def test_risk_prediction_function():
    """3. Tests real-time risk prediction pipeline function."""
    sample_payload = {
        "latitude": 12.9176,
        "longitude": 77.6234,
        "traffic_volume": 1100,
        "hour": 19,
        "day_of_week": 4,
        "month": 6,
        "weather": "Rainy",
        "road_condition": "Wet",
        "road_type": "Major Intersection",
        "visibility": 1.5,
        "speed_limit": 50,
        "previous_accidents": 8,
        "vehicle_count": 2
    }
    result = predict_accident_risk(sample_payload)
    assert "risk_level" in result
    assert result["risk_level"] in ["Low", "Medium", "High"]
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert len(result["contributing_factors"]) > 0

def test_api_predict_endpoint(client):
    """4. Tests POST /api/predict JSON endpoint."""
    sample_payload = {
        "latitude": 12.8452,
        "longitude": 77.6602,
        "traffic_volume": 850,
        "hour": 9,
        "day_of_week": 1,
        "month": 4,
        "weather": "Clear",
        "road_condition": "Dry",
        "road_type": "Elevated Expressway",
        "visibility": 8.0,
        "speed_limit": 80,
        "previous_accidents": 2,
        "vehicle_count": 1
    }
    response = client.post(
        "/api/predict",
        data=json.dumps(sample_payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert "probability" in data
    assert "contributing_factors" in data

def test_api_accidents_endpoint(client):
    """5. Tests GET /api/accidents endpoint."""
    response = client.get("/api/accidents")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "data" in data
    assert len(data["data"]) >= 1

def test_api_hotspots_endpoint(client):
    """6. Tests GET /api/hotspots endpoint."""
    response = client.get("/api/hotspots")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "data" in data

def test_api_analytics_data_endpoint(client):
    """7. Tests GET /api/analytics-data endpoint."""
    response = client.get("/api/analytics-data")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    for chart_key in ["by_hour", "by_day", "by_month", "by_weather", "by_road_condition", "by_severity"]:
        assert chart_key in data

def test_data_validation_helper():
    """8. Tests CSV validation function."""
    valid_csv = io.StringIO(
        "accident_date,accident_time,latitude,longitude,location,weather,road_condition,road_type,traffic_volume,vehicle_count,visibility,speed_limit,severity,risk_level\n"
        "2024-01-01,12:00,12.9716,77.5946,MG Road,Clear,Dry,Highway,500,1,10.0,50,Minor,Low\n"
    )
    is_valid, msg, df = validate_accident_csv(valid_csv)
    assert is_valid is True
    assert df is not None
    assert len(df) == 1

    invalid_csv = io.StringIO("random_col_1,random_col_2\n1,2\n")
    is_valid_inv, msg_inv, df_inv = validate_accident_csv(invalid_csv)
    assert is_valid_inv is False
    assert df_inv is None

def test_web_routes(client):
    """9. Tests core web views render successfully."""
    routes = [
        "/",
        "/dashboard",
        "/map",
        "/predict",
        "/hotspots",
        "/analytics",
        "/model",
        "/data-management",
        "/login"
    ]
    for route in routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"Failed GET on {route}"
