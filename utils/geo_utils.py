import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap, MarkerCluster

from ml.predict import predict_accident_risk

# Standard Coordinate reference system
CRS_WGS84 = "EPSG:4326"

# 10 Metropolitan Zones for Bengaluru City Corridors
METRO_ZONES = [
    {"name": "Silk Board Junction", "lat": 12.9176, "lon": 77.6234, "speed_limit": 50, "road_type": "Major Intersection", "road_cond": "Wet"},
    {"name": "Electronic City Flyover", "lat": 12.8452, "lon": 77.6602, "speed_limit": 80, "road_type": "Elevated Expressway", "road_cond": "Dry"},
    {"name": "Marathahalli Bridge", "lat": 12.9562, "lon": 77.7011, "speed_limit": 60, "road_type": "Arterial Bottleneck", "road_cond": "Potholed"},
    {"name": "Hebbal Flyover Junction", "lat": 13.0358, "lon": 77.5970, "speed_limit": 80, "road_type": "Highway Interchange", "road_cond": "Wet"},
    {"name": "KR Puram Hanging Bridge", "lat": 13.0006, "lon": 77.6835, "speed_limit": 50, "road_type": "Freight / Transit Corridor", "road_cond": "Damaged"},
    {"name": "Outer Ring Road - Bellandur", "lat": 12.9304, "lon": 77.6784, "speed_limit": 60, "road_type": "IT Corridor Express", "road_cond": "Dry"},
    {"name": "Majestic Central Bus Terminus", "lat": 12.9767, "lon": 77.5713, "speed_limit": 40, "road_type": "Transit Terminal Zone", "road_cond": "Dry"},
    {"name": "Koramangala 80ft Road", "lat": 12.9344, "lon": 77.6291, "speed_limit": 40, "road_type": "Commercial Corridor", "road_cond": "Dry"},
    {"name": "Yeshwanthpur Circle", "lat": 13.0238, "lon": 77.5503, "speed_limit": 60, "road_type": "Industrial Highway Gate", "road_cond": "Dry"},
    {"name": "Whitefield ITPL Main Road", "lat": 12.9870, "lon": 77.7310, "speed_limit": 50, "road_type": "Suburban Tech Hub", "road_cond": "Under Construction"}
]

RISK_COLORS = {
    'High': '#dc3545',    # Red
    'Medium': '#fd7e14',  # Orange
    'Low': '#198754'      # Green
}

def df_to_geodataframe(df):
    """Converts a standard Pandas DataFrame with latitude/longitude to GeoDataFrame."""
    if df is None or len(df) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=CRS_WGS84)

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
        crs=CRS_WGS84
    )
    return gdf


def aggregate_spatial_hotspots(df):
    """
    Groups spatial accident records by location clusters and aggregates risk statistics.
    """
    if df is None or len(df) == 0:
        return []

    gdf = df_to_geodataframe(df)
    
    # Group by location name
    hotspots = []
    grouped = gdf.groupby('location')

    for name, group in grouped:
        total_accidents = len(group)
        mean_lat = float(group['latitude'].mean())
        mean_lon = float(group['longitude'].mean())
        avg_traffic = int(group['traffic_volume'].mean())
        
        # Risk counts
        high_cnt = int((group['risk_level'] == 'High').sum())
        med_cnt = int((group['risk_level'] == 'Medium').sum())
        low_cnt = int((group['risk_level'] == 'Low').sum())

        # Dominant risk level
        if high_cnt >= med_cnt and high_cnt >= low_cnt:
            dom_risk = 'High'
            prob_est = round(0.70 + (high_cnt / total_accidents) * 0.25, 2)
        elif med_cnt >= low_cnt:
            dom_risk = 'Medium'
            prob_est = round(0.45 + (med_cnt / total_accidents) * 0.20, 2)
        else:
            dom_risk = 'Low'
            prob_est = round(0.20 + (low_cnt / total_accidents) * 0.15, 2)

        prob_est = min(prob_est, 0.98)
        mode_weather = group['weather'].mode()[0] if not group['weather'].empty else 'Clear'

        hotspots.append({
            'location': name,
            'latitude': round(mean_lat, 6),
            'longitude': round(mean_lon, 6),
            'total_accidents': total_accidents,
            'high_count': high_cnt,
            'medium_count': med_cnt,
            'low_count': low_cnt,
            'dominant_risk': dom_risk,
            'probability': prob_est,
            'avg_traffic': avg_traffic,
            'primary_weather': mode_weather
        })

    hotspots.sort(key=lambda x: (x['dominant_risk'] == 'High', x['total_accidents']), reverse=True)
    return hotspots


def generate_interactive_map(
    accidents_data=None,
    hotspots_data=None,
    center_lat=12.9716,
    center_lon=77.5946,
    zoom_start=11,
    include_heatmap=True,
    filter_risk=None,
    selected_marker=None
):
    """
    Creates an interactive Folium map supporting HeatMap, CircleMarkers,
    custom color-coded popups, and layer toggles.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap',
        control_scale=True
    )

    # Convert to DataFrame if list of dicts
    if isinstance(accidents_data, list):
        acc_df = pd.DataFrame(accidents_data)
    elif isinstance(accidents_data, pd.DataFrame):
        acc_df = accidents_data.copy()
    else:
        acc_df = pd.DataFrame()

    if not acc_df.empty and filter_risk:
        acc_df = acc_df[acc_df['risk_level'] == filter_risk]

    # 1. HeatMap Layer
    if include_heatmap and not acc_df.empty:
        heat_data = []
        for _, row in acc_df.iterrows():
            weight = 1.0 if row.get('risk_level') == 'High' else (0.6 if row.get('risk_level') == 'Medium' else 0.3)
            heat_data.append([float(row['latitude']), float(row['longitude']), weight])
        
        if heat_data:
            HeatMap(
                heat_data,
                name='Accident Heatmap',
                radius=14,
                blur=12,
                max_zoom=14,
                gradient={0.2: '#2b83ba', 0.5: '#abdda4', 0.7: '#fdae61', 0.9: '#d7191c'}
            ).add_to(m)

    # 2. Hotspots Layer with custom detailed Popups
    hotspot_group = folium.FeatureGroup(name='Hotspot Clusters').add_to(m)
    
    if hotspots_data:
        for spot in hotspots_data:
            risk = spot.get('dominant_risk', 'Medium')
            color = RISK_COLORS.get(risk, '#fd7e14')
            
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 210px; font-size: 13px;">
                <div style="background-color: {color}; color: white; padding: 6px 10px; border-radius: 4px 4px 0 0; font-weight: bold;">
                    {spot.get('location', 'Zone Hotspot')}
                </div>
                <div style="padding: 10px; border: 1px solid #ddd; border-top: none; background: #fff;">
                    <p style="margin: 3px 0;"><b>Risk Level:</b> <span style="color: {color}; font-weight: bold;">{risk}</span></p>
                    <p style="margin: 3px 0;"><b>Probability:</b> {float(spot.get('probability', 0.5))*100:.1f}%</p>
                    <p style="margin: 3px 0;"><b>Accident Count:</b> {spot.get('total_accidents', 0)}</p>
                    <p style="margin: 3px 0;"><b>Avg Traffic:</b> {spot.get('avg_traffic', 'N/A')} vph</p>
                    <p style="margin: 3px 0;"><b>Weather:</b> {spot.get('primary_weather', 'Clear')}</p>
                    <p style="margin: 3px 0; color: #666; font-size: 11px;">Coords: {spot.get('latitude')}, {spot.get('longitude')}</p>
                </div>
            </div>
            """
            folium.Circle(
                location=[spot['latitude'], spot['longitude']],
                radius=450,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.35,
                weight=2,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{spot.get('location')} - {risk} Risk"
            ).add_to(hotspot_group)

    # 3. Individual Accident Points (sampled to avoid browser lag if many)
    points_group = folium.FeatureGroup(name='Sampled Accident Points').add_to(m)
    if not acc_df.empty:
        sample_df = acc_df.sample(min(len(acc_df), 400), random_state=42)
        for _, row in sample_df.iterrows():
            r_level = row.get('risk_level', 'Medium')
            p_color = RISK_COLORS.get(r_level, '#fd7e14')
            
            p_popup = f"""
            <div style="font-size: 12px; font-family: sans-serif;">
                <b>{row.get('location', 'Location')}</b><br>
                Date: {row.get('accident_date', '')} {row.get('accident_time', '')}<br>
                Risk: <b style="color: {p_color};">{r_level}</b><br>
                Traffic: {row.get('traffic_volume', '')} vph<br>
                Weather: {row.get('weather', '')}<br>
                Severity: {row.get('severity', '')}
            </div>
            """
            folium.CircleMarker(
                location=[float(row['latitude']), float(row['longitude'])],
                radius=4,
                color=p_color,
                fill=True,
                fill_color=p_color,
                fill_opacity=0.7,
                popup=folium.Popup(p_popup, max_width=250),
                tooltip=f"{row.get('location')} ({r_level})"
            ).add_to(points_group)

    # 4. Highlighted Selected Point (e.g. from real-time prediction)
    if selected_marker:
        s_lat = selected_marker.get('latitude')
        s_lon = selected_marker.get('longitude')
        s_risk = selected_marker.get('risk_level', 'High')
        s_prob = selected_marker.get('probability', 0.8)
        s_color = RISK_COLORS.get(s_risk, '#dc3545')

        folium.Marker(
            location=[s_lat, s_lon],
            icon=folium.Icon(color='red' if s_risk == 'High' else ('orange' if s_risk == 'Medium' else 'green'), icon='exclamation-sign'),
            popup=folium.Popup(f"""
                <div style="font-size: 13px; font-weight: bold; color: {s_color};">
                    PREDICTED LOCATION<br>
                    Risk Level: {s_risk}<br>
                    Probability: {s_prob*100:.1f}%
                </div>
            """, max_width=250),
            tooltip=f"Predicted Target: {s_risk}"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m._repr_html_()


def calculate_scenario_hotspots(date_str, time_str, weather, traffic_level):
    """
    Dynamically recalculates risk levels and probabilities for all predefined zones
    under user-selected scenario conditions.
    """
    try:
        hour = int(time_str.split(':')[0]) if ':' in time_str else int(time_str)
    except:
        hour = 12

    try:
        dt = pd.to_datetime(date_str)
        day_of_week = int(dt.dayofweek)
        month = int(dt.month)
    except:
        day_of_week = 2
        month = 6

    # Traffic level multiplier
    traffic_map = {
        'Low': 350,
        'Medium': 650,
        'High': 1050,
        'Severe Congestion': 1350
    }
    base_traffic = traffic_map.get(traffic_level, 750)

    results = []

    for zone in METRO_ZONES:
        payload = {
            "latitude": zone["lat"],
            "longitude": zone["lon"],
            "traffic_volume": int(base_traffic * (1.2 if "Junction" in zone["name"] or "Bridge" in zone["name"] else 0.9)),
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "weather": weather,
            "road_condition": zone["road_cond"],
            "road_type": zone["road_type"],
            "visibility": 1.8 if weather in ["Foggy", "Rainy", "Thunderstorm"] else 7.5,
            "speed_limit": zone["speed_limit"],
            "previous_accidents": 6 if "Bridge" in zone["name"] or "Junction" in zone["name"] else 2,
            "vehicle_count": 2
        }

        pred_res = predict_accident_risk(payload)

        results.append({
            "location": zone["name"],
            "latitude": zone["lat"],
            "longitude": zone["lon"],
            "risk_level": pred_res["risk_level"],
            "probability": pred_res["probability"],
            "traffic_volume": payload["traffic_volume"],
            "weather": weather,
            "road_condition": zone["road_cond"],
            "speed_limit": zone["speed_limit"],
            "dominant_risk": pred_res["risk_level"], # For map compatibility
            "total_accidents": payload["previous_accidents"] + int(pred_res["probability"] * 10),
            "avg_traffic": payload["traffic_volume"],
            "primary_weather": weather
        })

    # Sort descending by probability
    results.sort(key=lambda x: x["probability"], reverse=True)
    
    # Assign ranks
    for rank_idx, item in enumerate(results, start=1):
        item["rank"] = rank_idx

    return results
