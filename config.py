import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

def build_database_uri():
    db_type = os.getenv("DB_TYPE", "mysql").lower()
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "accident_hotspot_db")

    mysql_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    sqlite_path = BASE_DIR / "data" / "accident_hotspot.db"
    sqlite_uri = f"sqlite:///{sqlite_path}"
    
    if db_type == "sqlite":
        return sqlite_uri
    
    # Test MySQL connection
    try:
        import pymysql
        conn = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            port=int(db_port),
            connect_timeout=2
        )
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARACTER SET utf8mb4;")
        conn.commit()
        conn.close()
        return mysql_uri
    except Exception as e:
        print(f"[Database Notice] MySQL server connection to '{db_host}:{db_port}' was unavailable ({e}).")
        print(f"[Database Notice] Seamlessly using SQLite fallback: {sqlite_uri}")
        return sqlite_uri

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "roadsafe_ai_default_secret_key_2026")
    DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"
    
    # Database Settings
    DB_TYPE = os.getenv("DB_TYPE", "mysql").lower()
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "3306")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "accident_hotspot_db")
    
    # Model and Data Paths
    DATA_DIR = BASE_DIR / "data"
    CSV_DATA_PATH = DATA_DIR / "accident_data.csv"
    MODELS_DIR = BASE_DIR / "models"
    MODEL_PATH = MODELS_DIR / "accident_hotspot_model.pkl"
    PREPROCESSING_PATH = MODELS_DIR / "preprocessing.pkl"
    METRICS_PATH = MODELS_DIR / "model_metrics.pkl"

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = build_database_uri()

