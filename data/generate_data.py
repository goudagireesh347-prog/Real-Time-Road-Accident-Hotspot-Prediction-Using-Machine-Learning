import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_accident_dataset(num_records=6000, seed=42):
    """
    Generates realistic, reproducible synthetic road accident data with spatial,
    temporal, environmental, and behavioral features for MCA final year demonstration.
    """
    np.random.seed(seed)
    random.seed(seed)

    # 10 Major Real-World Metropolitan Hubs with coordinates & baseline risk tendencies
    zones = [
        {
            "name": "Silk Board Junction",
            "lat": 12.9176, "lon": 77.6234,
            "type": "Major Intersection",
            "risk_bias": 0.45, # high risk tendency
            "base_traffic": 950,
            "speed_limit": 50
        },
        {
            "name": "Electronic City Flyover",
            "lat": 12.8452, "lon": 77.6602,
            "type": "Elevated Expressway",
            "risk_bias": 0.40,
            "base_traffic": 850,
            "speed_limit": 80
        },
        {
            "name": "Marathahalli Bridge",
            "lat": 12.9562, "lon": 77.7011,
            "type": "Arterial Bottleneck",
            "risk_bias": 0.38,
            "base_traffic": 900,
            "speed_limit": 60
        },
        {
            "name": "Hebbal Flyover Junction",
            "lat": 13.0358, "lon": 77.5970,
            "type": "Highway Interchange",
            "risk_bias": 0.35,
            "base_traffic": 880,
            "speed_limit": 80
        },
        {
            "name": "KR Puram Hanging Bridge",
            "lat": 13.0006, "lon": 77.6835,
            "type": "Freight / Transit Corridor",
            "risk_bias": 0.42,
            "base_traffic": 920,
            "speed_limit": 50
        },
        {
            "name": "Outer Ring Road - Bellandur",
            "lat": 12.9304, "lon": 77.6784,
            "type": "IT Corridor Express",
            "risk_bias": 0.35,
            "base_traffic": 820,
            "speed_limit": 60
        },
        {
            "name": "Majestic Central Bus Terminus",
            "lat": 12.9767, "lon": 77.5713,
            "type": "Transit Terminal Zone",
            "risk_bias": 0.25,
            "base_traffic": 780,
            "speed_limit": 40
        },
        {
            "name": "Koramangala 80ft Road",
            "lat": 12.9344, "lon": 77.6291,
            "type": "Commercial Corridor",
            "risk_bias": 0.20,
            "base_traffic": 650,
            "speed_limit": 40
        },
        {
            "name": "Yeshwanthpur Circle",
            "lat": 13.0238, "lon": 77.5503,
            "type": "Industrial Highway Gate",
            "risk_bias": 0.30,
            "base_traffic": 750,
            "speed_limit": 60
        },
        {
            "name": "Whitefield ITPL Main Road",
            "lat": 12.9870, "lon": 77.7310,
            "type": "Suburban Tech Hub",
            "risk_bias": 0.22,
            "base_traffic": 700,
            "speed_limit": 50
        }
    ]

    weather_conditions = ["Clear", "Rainy", "Foggy", "Overcast", "Thunderstorm"]
    weather_weights = [0.48, 0.22, 0.12, 0.13, 0.05]

    road_conditions = ["Dry", "Wet", "Damaged", "Under Construction", "Potholed"]
    road_cond_weights = [0.50, 0.22, 0.12, 0.08, 0.08]

    road_types = ["Highway", "Arterial Road", "Intersection", "Local Street", "Flyover"]

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    total_days = (end_date - start_date).days

    records = []

    for i in range(num_records):
        # Pick a geographic hotspot center with realistic dispersion (std dev ~ 500m to 1km)
        zone = random.choice(zones)
        lat = zone["lat"] + np.random.normal(0, 0.0075)
        lon = zone["lon"] + np.random.normal(0, 0.0075)

        # Random date & realistic time
        rand_days = random.randint(0, total_days)
        acc_date = start_date + timedelta(days=rand_days)
        
        # Bimodal peak hour probability (morning peak 8-10, evening peak 17-21, night speeding 22-02)
        hour_prob = np.array([
            0.03, 0.03, 0.02, 0.01, 0.01, 0.02, # 00 - 05
            0.04, 0.06, 0.08, 0.08, 0.06, 0.04, # 06 - 11
            0.04, 0.04, 0.04, 0.05, 0.06, 0.08, # 12 - 17
            0.09, 0.08, 0.07, 0.05, 0.04, 0.03  # 18 - 23
        ])
        hour_prob = hour_prob / hour_prob.sum()
        hour = int(np.random.choice(range(24), p=hour_prob))
        minute = random.randint(0, 59)
        acc_time = f"{hour:02d}:{minute:02d}"

        # Weather & road conditions
        # Severe weather slightly more frequent during monsoon months (June-September)
        month = acc_date.month
        if month in [6, 7, 8, 9, 10]:
            w_weights = [0.30, 0.38, 0.12, 0.12, 0.08]
        elif month in [11, 12, 1]: # Winter fog
            w_weights = [0.45, 0.10, 0.32, 0.10, 0.03]
        else:
            w_weights = weather_weights
        
        w_weights = np.array(w_weights) / np.sum(w_weights)
        weather = np.random.choice(weather_conditions, p=w_weights)

        # Road condition correlated with weather
        if weather in ["Rainy", "Thunderstorm"]:
            rc_weights = [0.15, 0.55, 0.15, 0.08, 0.07]
        else:
            rc_weights = road_cond_weights
        
        rc_weights = np.array(rc_weights) / np.sum(rc_weights)
        road_condition = np.random.choice(road_conditions, p=rc_weights)
        road_type = zone["type"] if random.random() < 0.6 else random.choice(road_types)

        # Visibility (in km)
        if weather == "Foggy":
            visibility = round(np.random.uniform(0.3, 2.5), 1)
        elif weather in ["Rainy", "Thunderstorm"]:
            visibility = round(np.random.uniform(1.5, 5.0), 1)
        else:
            visibility = round(np.random.uniform(5.5, 10.0), 1)

        speed_limit = zone["speed_limit"]
        
        # Traffic volume calculation
        is_rush_hour = (8 <= hour <= 11) or (17 <= hour <= 21)
        rush_multiplier = 1.35 if is_rush_hour else (0.45 if hour < 6 else 0.85)
        traffic_volume = int(max(100, np.random.normal(zone["base_traffic"] * rush_multiplier, 120)))
        
        # Vehicle count involved
        vehicle_count = np.random.choice([1, 2, 3, 4, 5], p=[0.40, 0.42, 0.13, 0.04, 0.01])

        # Previous accidents in the vicinity
        previous_accidents = int(np.clip(np.random.poisson(lam=zone["risk_bias"] * 10), 0, 15))

        # Risk Score Calculation (Ground Truth Logic for realistic ML learnability)
        risk_score = 0.0
        
        # Zone baseline bias
        risk_score += zone["risk_bias"] * 28.0

        # Traffic volume contribution
        if traffic_volume > 1000:
            risk_score += 24.0
        elif traffic_volume > 650:
            risk_score += 15.0
        else:
            risk_score += 6.0

        # Environmental factors
        if weather in ["Rainy", "Thunderstorm"]:
            risk_score += 18.0
        elif weather == "Foggy":
            risk_score += 15.0
        
        if road_condition in ["Wet", "Potholed", "Damaged"]:
            risk_score += 16.0
        elif road_condition == "Under Construction":
            risk_score += 12.0

        # Time & Speeding
        if hour in [22, 23, 0, 1, 2, 3] and speed_limit >= 60:
            risk_score += 14.0 # Night speeding risk
        elif is_rush_hour:
            risk_score += 10.0 # Dense congestion bump risk

        if visibility < 2.0:
            risk_score += 12.0

        # History
        risk_score += min(previous_accidents * 2.5, 20.0)

        # Random noise to simulate real-world variability
        risk_score += np.random.normal(0, 6.0)

        # Categorize into Risk Level & Severity
        if risk_score >= 68.0:
            risk_level = "High"
            severity = np.random.choice(["Severe", "Fatal", "Major"], p=[0.55, 0.25, 0.20])
        elif risk_score >= 44.0:
            risk_level = "Medium"
            severity = np.random.choice(["Moderate", "Minor", "Major"], p=[0.60, 0.25, 0.15])
        else:
            risk_level = "Low"
            severity = np.random.choice(["Minor", "Negligible", "Moderate"], p=[0.75, 0.20, 0.05])

        records.append({
            "accident_date": acc_date.strftime("%Y-%m-%d"),
            "accident_time": acc_time,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "location": zone["name"],
            "weather": weather,
            "road_condition": road_condition,
            "road_type": road_type,
            "traffic_volume": traffic_volume,
            "vehicle_count": vehicle_count,
            "visibility": visibility,
            "speed_limit": speed_limit,
            "previous_accidents": previous_accidents,
            "severity": severity,
            "risk_level": risk_level
        })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_csv = data_dir / "accident_data.csv"

    print("Generating 6,000 realistic synthetic accident records with reproducible seed 42...")
    df = generate_synthetic_accident_dataset(num_records=6000, seed=42)
    df.to_csv(out_csv, index=False)
    print(f"Successfully generated and saved {len(df)} records to: {out_csv}")
    print("\nClass distribution:")
    print(df['risk_level'].value_counts())
    print("\nSample records:")
    print(df.head(3))
