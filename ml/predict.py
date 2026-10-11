import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.preprocessing import AccidentPreprocessor, RISK_CLASSES

# Global cache for loaded model and preprocessor
_MODEL_CACHE = {
    "model": None,
    "preprocessor": None,
    "metrics": None
}

def load_prediction_artifacts(model_path=None, preprocessor_path=None, metrics_path=None):
    """Loads and caches the model, preprocessor, and metrics."""
    if _MODEL_CACHE["model"] is not None and _MODEL_CACHE["preprocessor"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["preprocessor"], _MODEL_CACHE["metrics"]

    if model_path is None:
        model_path = BASE_DIR / "models" / "accident_hotspot_model.pkl"
    if preprocessor_path is None:
        preprocessor_path = BASE_DIR / "models" / "preprocessing.pkl"
    if metrics_path is None:
        metrics_path = BASE_DIR / "models" / "model_metrics.pkl"

    if not Path(model_path).exists() or not Path(preprocessor_path).exists():
        raise FileNotFoundError(
            f"Model or Preprocessor artifact not found. Please run ml/train_model.py first."
        )

    _MODEL_CACHE["model"] = joblib.load(model_path)
    _MODEL_CACHE["preprocessor"] = joblib.load(preprocessor_path)
    
    if Path(metrics_path).exists():
        _MODEL_CACHE["metrics"] = joblib.load(metrics_path)

    return _MODEL_CACHE["model"], _MODEL_CACHE["preprocessor"], _MODEL_CACHE["metrics"]


def reload_prediction_artifacts():
    """Forces reload of cached model artifacts after retraining."""
    _MODEL_CACHE["model"] = None
    _MODEL_CACHE["preprocessor"] = None
    _MODEL_CACHE["metrics"] = None
    return load_prediction_artifacts()


def explain_contributing_factors(input_dict, predicted_class, probability):
    """
    Generates intelligent, domain-specific rule and tree-informed factor explanations
    for MCA viva demonstration.
    """
    factors = []

    traffic = float(input_dict.get('traffic_volume', 500))
    weather = str(input_dict.get('weather', 'Clear')).title()
    road_cond = str(input_dict.get('road_condition', 'Dry')).title()
    speed_limit = float(input_dict.get('speed_limit', 50))
    visibility = float(input_dict.get('visibility', 5.0))
    prev_acc = int(input_dict.get('previous_accidents', 0))
    hour = int(input_dict.get('hour', 12))
    vehicle_count = int(input_dict.get('vehicle_count', 1))

    # Traffic factor
    if traffic >= 850:
        factors.append({
            "factor": "Severe Traffic Density",
            "impact": "High Positive",
            "description": f"High vehicle congestion ({int(traffic)} vph) dramatically amplifies friction points and multi-vehicle conflict opportunities."
        })
    elif traffic >= 600:
        factors.append({
            "factor": "Moderate Traffic Load",
            "impact": "Medium Positive",
            "description": f"Vehicle density of {int(traffic)} vph maintains steady traffic flow with elevated intersection exposure."
        })

    # Weather factor
    if weather in ["Rainy", "Thunderstorm"]:
        factors.append({
            "factor": "Adverse Weather Condition",
            "impact": "High Positive",
            "description": f"{weather} weather substantially lowers tire grip and increases stopping distances."
        })
    elif weather == "Foggy":
        factors.append({
            "factor": "Foggy Atmospheric Condition",
            "impact": "High Positive",
            "description": "Dense fog limits forward sight distance and early brake reaction times."
        })

    # Road Condition
    if road_cond in ["Wet", "Damaged", "Potholed"]:
        factors.append({
            "factor": "Hazardous Road Surface",
            "impact": "High Positive",
            "description": f"Surface condition '{road_cond}' increases skidding hazard and emergency swerve probability."
        })

    # Speed limit & Time
    if speed_limit >= 70:
        factors.append({
            "factor": "High Speed Limit Corridor",
            "impact": "Moderate Positive",
            "description": f"Designated speed limit of {int(speed_limit)} km/h increases crash impact energy."
        })
    if (22 <= hour or hour <= 4):
        factors.append({
            "factor": "Late Night / Reduced Driver Alertness",
            "impact": "Moderate Positive",
            "description": f"Off-peak night hour ({hour}:00) exhibits higher likelihood of driver fatigue and speeding violations."
        })
    elif (8 <= hour <= 11) or (17 <= hour <= 21):
        factors.append({
            "factor": "Peak Rush Hour Convergence",
            "impact": "Moderate Positive",
            "description": f"Hour ({hour}:00) coincides with peak metropolitan commuter movements."
        })

    # Visibility
    if visibility < 2.5:
        factors.append({
            "factor": "Low Sight Distance",
            "impact": "High Positive",
            "description": f"Reported visibility of {visibility} km is well below the safe highway threshold of 5.0 km."
        })

    # Accident History
    if prev_acc >= 5:
        factors.append({
            "factor": "Chronic Accident History",
            "impact": "High Positive",
            "description": f"Zone has recorded {prev_acc} prior accidents, indicating recurring geometric or infrastructural vulnerability."
        })

    if not factors:
        factors.append({
            "factor": "Favorable Environmental & Traffic Baseline",
            "impact": "Mitigating",
            "description": "Clear weather, moderate speeds, and dry road surface provide favorable operating safety parameters."
        })

    return factors


def predict_accident_risk(input_data):
    """
    Validates input parameters, executes Random Forest inference,
    computes probabilities and explanations.
    """
    model, preprocessor, _ = load_prediction_artifacts()

    # Validation & parsing
    try:
        lat = float(input_data.get('latitude', 12.9716))
        lon = float(input_data.get('longitude', 77.5946))
        traffic_vol = int(input_data.get('traffic_volume', 600))
        hour = int(input_data.get('hour', 12))
        day_of_week = int(input_data.get('day_of_week', 2))
        month = int(input_data.get('month', 5))
        visibility = float(input_data.get('visibility', 6.0))
        speed_limit = int(input_data.get('speed_limit', 50))
        prev_accidents = int(input_data.get('previous_accidents', 2))
        vehicle_count = int(input_data.get('vehicle_count', 1))
        weather = str(input_data.get('weather', 'Clear'))
        road_condition = str(input_data.get('road_condition', 'Dry'))
        road_type = str(input_data.get('road_type', 'Arterial Road'))
    except (ValueError, TypeError) as err:
        raise ValueError(f"Invalid input parameter format: {err}")

    # Prepare single row DataFrame for preprocessor
    row_dict = {
        'latitude': lat,
        'longitude': lon,
        'traffic_volume': traffic_vol,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'visibility': visibility,
        'speed_limit': speed_limit,
        'previous_accidents': prev_accidents,
        'vehicle_count': vehicle_count,
        'weather': weather,
        'road_condition': road_condition,
        'road_type': road_type
    }

    # Transform input
    X_input = preprocessor.transform(pd.DataFrame([row_dict]))

    # Inference
    pred_class = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]

    # Map classes to probabilities
    class_probs = {}
    for cls_name, prob_val in zip(model.classes_, probabilities):
        class_probs[str(cls_name)] = round(float(prob_val), 4)

    primary_probability = class_probs.get(pred_class, round(float(np.max(probabilities)), 4))

    # Factor explanations
    factors = explain_contributing_factors(row_dict, pred_class, primary_probability)

    return {
        "risk_level": pred_class,
        "probability": primary_probability,
        "class_probabilities": class_probs,
        "latitude": lat,
        "longitude": lon,
        "contributing_factors": factors,
        "input_features": row_dict
    }

if __name__ == "__main__":
    test_sample = {
        "latitude": 12.9176,
        "longitude": 77.6234,
        "traffic_volume": 1150,
        "hour": 19,
        "day_of_week": 4,
        "month": 7,
        "weather": "Rainy",
        "road_condition": "Wet",
        "road_type": "Major Intersection",
        "visibility": 2.1,
        "speed_limit": 60,
        "previous_accidents": 8,
        "vehicle_count": 2
    }
    result = predict_accident_risk(test_sample)
    print("Test Prediction Output:")
    print(f"Risk Level : {result['risk_level']}")
    print(f"Probability: {result['probability'] * 100:.2f}%")
    print(f"Probabilities: {result['class_probabilities']}")
    print("Contributing Factors:")
    for f in result['contributing_factors']:
        print(f" - [{f['factor']}]: {f['description']}")
