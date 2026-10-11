import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Feature specifications
NUMERICAL_FEATURES = [
    'latitude',
    'longitude',
    'traffic_volume',
    'hour',
    'day_of_week',
    'month',
    'visibility',
    'speed_limit',
    'previous_accidents',
    'vehicle_count'
]

CATEGORICAL_FEATURES = [
    'weather',
    'road_condition',
    'road_type'
]

ALL_INPUT_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = 'risk_level'
RISK_CLASSES = ['Low', 'Medium', 'High']

class AccidentPreprocessor:
    """
    Comprehensive data preprocessing and feature transformation pipeline
    incorporating Pandas, NumPy, and GeoPandas.
    """
    def __init__(self):
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
            ],
            remainder='passthrough'
        )
        self.feature_names = []
        self.is_fitted = False

    def clean_and_extract_features(self, df):
        """
        Cleans data, removes duplicates, validates coordinates,
        and extracts temporal and spatial features.
        """
        df = df.copy()

        # 1. Remove duplicate records
        df = df.drop_duplicates()

        # 2. Handle missing values
        df['traffic_volume'] = df['traffic_volume'].fillna(df['traffic_volume'].median())
        df['vehicle_count'] = df['vehicle_count'].fillna(1)
        df['visibility'] = df['visibility'].fillna(df['visibility'].median())
        df['speed_limit'] = df['speed_limit'].fillna(50)
        df['previous_accidents'] = df['previous_accidents'].fillna(0)
        df['weather'] = df['weather'].fillna('Clear')
        df['road_condition'] = df['road_condition'].fillna('Dry')
        df['road_type'] = df['road_type'].fillna('Arterial Road')

        # 3. Validate coordinates (Bengaluru / India latitude/longitude bounds or standard global)
        df = df[(df['latitude'] >= -90.0) & (df['latitude'] <= 90.0)]
        df = df[(df['longitude'] >= -180.0) & (df['longitude'] <= 180.0)]

        # 4. Temporal extraction
        if 'accident_date' in df.columns:
            date_series = pd.to_datetime(df['accident_date'], errors='coerce')
            df['month'] = date_series.dt.month.fillna(1).astype(int)
            df['day'] = date_series.dt.day.fillna(1).astype(int)
            df['day_of_week'] = date_series.dt.dayofweek.fillna(0).astype(int)
        else:
            if 'month' not in df.columns:
                df['month'] = 1
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = 0

        if 'accident_time' in df.columns:
            # Extract hour from HH:MM string or integer
            def extract_hour(val):
                try:
                    if pd.isna(val):
                        return 12
                    s = str(val).strip()
                    if ':' in s:
                        return int(s.split(':')[0])
                    return int(float(s))
                except:
                    return 12
            df['hour'] = df['accident_time'].apply(extract_hour)
        else:
            if 'hour' not in df.columns:
                df['hour'] = 12

        # 5. GeoPandas Integration: Create GeoDataFrame for spatial validity
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )

        return gdf

    def fit(self, df):
        """Fits the categorical encoder and builds feature names."""
        clean_df = self.clean_and_extract_features(df)
        X = clean_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
        self.column_transformer.fit(X)

        # Build feature names
        cat_encoder = self.column_transformer.named_transformers_['cat']
        cat_cols = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        self.feature_names = cat_cols + NUMERICAL_FEATURES
        self.is_fitted = True
        return self

    def transform(self, df):
        """Transforms input DataFrame into model-ready numpy array."""
        if not self.is_fitted:
            raise ValueError("AccidentPreprocessor must be fitted before calling transform().")

        clean_df = self.clean_and_extract_features(df)
        X = clean_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
        transformed = self.column_transformer.transform(X)
        return transformed

    def transform_single_record(self, record_dict):
        """Converts a single input dictionary from web form/API into model format."""
        temp_df = pd.DataFrame([record_dict])
        return self.transform(temp_df)
