import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd

db = SQLAlchemy()

class Accident(db.Model):
    __tablename__ = 'accidents'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    accident_date = db.Column(db.Date, nullable=False, index=True)
    accident_time = db.Column(db.String(10), nullable=False)
    latitude = db.Column(db.Float, nullable=False, index=True)
    longitude = db.Column(db.Float, nullable=False, index=True)
    location = db.Column(db.String(150), nullable=False)
    weather = db.Column(db.String(50), nullable=False)
    road_condition = db.Column(db.String(50), nullable=False)
    road_type = db.Column(db.String(50), nullable=False)
    traffic_volume = db.Column(db.Integer, nullable=False)
    vehicle_count = db.Column(db.Integer, nullable=False)
    visibility = db.Column(db.Float, nullable=False)
    speed_limit = db.Column(db.Integer, nullable=False)
    previous_accidents = db.Column(db.Integer, nullable=False, default=0)
    severity = db.Column(db.String(50), nullable=False)
    risk_level = db.Column(db.String(20), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'accident_date': self.accident_date.strftime('%Y-%m-%d') if self.accident_date else '',
            'accident_time': self.accident_time,
            'latitude': round(self.latitude, 6),
            'longitude': round(self.longitude, 6),
            'location': self.location,
            'weather': self.weather,
            'road_condition': self.road_condition,
            'road_type': self.road_type,
            'traffic_volume': self.traffic_volume,
            'vehicle_count': self.vehicle_count,
            'visibility': self.visibility,
            'speed_limit': self.speed_limit,
            'previous_accidents': self.previous_accidents,
            'severity': self.severity,
            'risk_level': self.risk_level,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }


class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    latitude = db.Column(db.Float, nullable=False, index=True)
    longitude = db.Column(db.Float, nullable=False, index=True)
    prediction_time = db.Column(db.DateTime, nullable=False, default=datetime.datetime.now)
    weather = db.Column(db.String(50), nullable=False)
    traffic_volume = db.Column(db.Integer, nullable=False)
    predicted_risk = db.Column(db.String(20), nullable=False, index=True)
    probability = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'latitude': round(self.latitude, 6),
            'longitude': round(self.longitude, 6),
            'prediction_time': self.prediction_time.strftime('%Y-%m-%d %H:%M:%S') if self.prediction_time else '',
            'weather': self.weather,
            'traffic_volume': self.traffic_volume,
            'predicted_risk': self.predicted_risk,
            'probability': round(self.probability, 4),
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='Planner') # 'Admin' or 'Planner'
    created_at = db.Column(db.DateTime, default=datetime.datetime.now)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
        seed_initial_users()


def seed_initial_users():
    """Seeds default demo users if they do not exist."""
    try:
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', email='admin@roadsafe.ai', role='Admin')
            admin.set_password('admin123')
            db.session.add(admin)

        if not User.query.filter_by(username='planner').first():
            planner = User(username='planner', email='planner@roadsafe.ai', role='Planner')
            planner.set_password('planner123')
            db.session.add(planner)

        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"[Seed Notice] Error initializing users: {e}")


def import_csv_to_db(csv_path):
    """Imports accident CSV into the database if the table is empty or for synchronization."""
    try:
        df = pd.read_csv(csv_path)
        records_to_insert = []
        for _, row in df.iterrows():
            rec = Accident(
                accident_date=pd.to_datetime(row['accident_date']).date(),
                accident_time=str(row['accident_time']),
                latitude=float(row['latitude']),
                longitude=float(row['longitude']),
                location=str(row['location']),
                weather=str(row['weather']),
                road_condition=str(row['road_condition']),
                road_type=str(row['road_type']),
                traffic_volume=int(row['traffic_volume']),
                vehicle_count=int(row['vehicle_count']),
                visibility=float(row['visibility']),
                speed_limit=int(row['speed_limit']),
                previous_accidents=int(row.get('previous_accidents', 0)),
                severity=str(row['severity']),
                risk_level=str(row['risk_level'])
            )
            records_to_insert.append(rec)
        
        db.session.bulk_save_objects(records_to_insert)
        db.session.commit()
        print(f"[Database Sync] Successfully loaded {len(records_to_insert)} records into database.")
        return len(records_to_insert)
    except Exception as e:
        db.session.rollback()
        print(f"[Database Sync Error] Failed importing CSV: {e}")
        return 0
