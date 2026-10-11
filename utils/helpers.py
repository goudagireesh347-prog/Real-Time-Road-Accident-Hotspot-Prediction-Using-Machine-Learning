from functools import wraps
from flask import session, redirect, url_for, flash
import pandas as pd

REQUIRED_CSV_COLUMNS = [
    'accident_date',
    'accident_time',
    'latitude',
    'longitude',
    'location',
    'weather',
    'road_condition',
    'road_type',
    'traffic_volume',
    'vehicle_count',
    'visibility',
    'speed_limit',
    'severity',
    'risk_level'
]

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please sign in to access this page.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please sign in to access this feature.", "warning")
            return redirect(url_for('login'))
        if session.get('role') != 'Admin':
            flash("Access denied: Administrator privileges required.", "danger")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function


def validate_accident_csv(file_stream):
    """
    Validates an uploaded CSV file ensuring required schema columns exist
    and basic types are sound.
    """
    try:
        df = pd.read_csv(file_stream)
        missing_cols = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns in CSV: {', '.join(missing_cols)}", None

        # Clean null values and invalid coordinates
        df = df.dropna(subset=['latitude', 'longitude', 'risk_level'])
        df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
        df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]

        if len(df) == 0:
            return False, "Uploaded CSV contains no valid records after coordinate filtering.", None

        return True, f"CSV validated successfully ({len(df)} records).", df
    except Exception as e:
        return False, f"Failed to parse CSV file: {str(e)}", None
