# RoadSafe AI – Real-Time Road Accident Hotspot Prediction and Visualization System
### MCA Final Year Major Project

A full-stack, map-based geospatial decision support and predictive analytics web application designed for municipal city planners, traffic police departments, and civil authorities.

---

## 🌟 Key Features

* **Executive Dashboard**: Key performance indicators (Total Accidents, High/Medium/Low Risk Counts, Active Zones), embedded interactive Folium map, and diurnality trend charts.
* **Interactive Cartography (Folium + GeoPandas)**: Spatial clustering, density heatmaps (`folium.plugins.HeatMap`), risk color coding (Red = High, Orange = Medium, Green = Low), and custom popups.
* **Machine Learning Engine**: Random Forest Classifier with 120 estimators, depth regularization, balanced weights, and feature importance analysis, achieving **81.08% Test Accuracy**.
* **Real-Time Risk Prediction**: Interactive form with single-coordinate inference, class probabilities, factor impact explanations, and dynamic GIS marker mapping.
* **Dynamic Hotspot Scenario Simulator**: Test "what-if" scenarios across date, time, weather, and traffic density parameters across 10 major metropolitan corridors.
* **Accident Analytics**: 8 dynamic Chart.js dashboards querying live Flask REST APIs (Accidents by Hour, Day, Month, Weather, Road Condition, Severity, Location, and High-Risk Blackspots).
* **Data Management**: Search, filter, create, and delete accident records, upload CSV datasets with column validation, and trigger pipeline retraining.
* **Role-Based Authentication**: Secure session-based authentication with `pbkdf2:sha256` password hashing for **Administrator** and **City Planner** roles.
* **Jupyter Notebook Integration**: Development, EDA, and demo notebook (`notebooks/accident_hotspot_project.ipynb`).

---

## 🏗️ Technology Stack

* **Backend Framework**: Python 3, Flask 3
* **Database**: MySQL (PyMySQL, Flask-SQLAlchemy) with seamless local SQLite fallback
* **Machine Learning**: Scikit-learn (Random Forest), NumPy, Pandas, Joblib
* **Geospatial Processing**: GeoPandas, Shapely (CRS EPSG:4326), Folium
* **Frontend**: HTML5, CSS3, JavaScript (ES6), Bootstrap 5.3, Font Awesome 6, Chart.js 4.4
* **Testing**: Pytest

---

## 📁 Project Structure

```
accident_hotspot_prediction/
│
├── app.py                            # Main Flask Application & Web/API Routes
├── config.py                         # App Configuration & Intelligent Database URI Builder
├── requirements.txt                  # Python Package Dependencies
├── .env.example                      # Environment Template
├── .env                              # Environment Configuration
├── README.md                         # Project Guide & Documentation
│
├── data/
│   ├── accident_data.csv             # 6,000 Realistic Accident Records
│   └── generate_data.py              # Reproducible Synthetic Dataset Generator (Seed 42)
│
├── models/
│   ├── accident_hotspot_model.pkl    # Serialized Random Forest Model
│   ├── preprocessing.pkl             # Fitted Preprocessor Pipeline & OneHotEncoder
│   └── model_metrics.pkl             # Accuracy, Confusion Matrix, and Feature Importances
│
├── ml/
│   ├── train_model.py                # Model Training & Evaluation Script
│   ├── predict.py                    # Inference Engine & Factor Explanations
│   └── preprocessing.py              # Cleaning, Temporal Extraction & Preprocessor Pipeline
│
├── database/
│   ├── db.py                         # SQLAlchemy Models (Accident, Prediction, User) & Seed
│   └── schema.sql                    # MySQL Schema DDL with Indexes
│
├── notebooks/
│   └── accident_hotspot_project.ipynb # Interactive EDA & Viva Demonstration Notebook
│
├── templates/
│   ├── base.html                     # Master Template with Sidebar & Navbar
│   ├── index.html                    # Welcome / Project Architecture Overview
│   ├── dashboard.html                # Executive Dashboard with Map & Trends
│   ├── map.html                      # Full-Screen Interactive Folium Map
│   ├── predict.html                  # Real-Time Risk Predictor
│   ├── hotspots.html                 # Dynamic Multi-Zone Scenario Simulator
│   ├── analytics.html                # 8 Dynamic Chart.js Analytics Dashboards
│   ├── model.html                    # Model Performance & Confusion Matrix
│   ├── data_management.html          # Records Management & CSV Upload
│   ├── login.html                    # Authentication Portal with 1-Click Demo Logins
│   ├── 404.html                      # 404 Error Page
│   └── 500.html                      # 500 Error Page
│
├── static/
│   ├── css/
│   │   └── style.css                 # Custom Styling
│   ├── js/
│   │   └── dashboard.js              # Chart.js Integration & API Fetching
│   └── images/
│
├── utils/
│   ├── geo_utils.py                  # GeoPandas & Folium Map Generators
│   └── helpers.py                    # Auth Decorators & CSV Validators
│
├── tests/
│   └── test_app.py                   # Automated Test Suite (9 Test Cases)
│
└── docs/
    └── MCA_FINAL_PROJECT_REPORT.md   # Complete Academic MCA Project Documentation
```

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
Ensure Python 3.10+ and pip are installed on your system.

### 2. Navigate to Project Directory
```powershell
cd "C:\Users\Deeksha bhat\.gemini\antigravity\scratch\accident_hotspot_prediction"
```

### 3. Create and Activate Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. MySQL Database Setup (Optional)
If running MySQL locally:
1. Open MySQL Command Line or Workbench:
```sql
CREATE DATABASE accident_hotspot_db;
```
2. Run `database/schema.sql`:
```bash
mysql -u root -p accident_hotspot_db < database/schema.sql
```
3. Update `.env` with your MySQL credentials:
```env
DB_TYPE=mysql
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=accident_hotspot_db
```
> **Note**: If MySQL is not running or credentials are not provided, the system **automatically and seamlessly falls back to local SQLite** (`data/accident_hotspot.db`), ensuring zero setup barriers during demonstrations!

### 6. Generate Dataset (If Not Present)
To generate the 6,000 spatial records with reproducible seed 42:
```powershell
python data/generate_data.py
```

### 7. Train the Machine Learning Model
```powershell
python ml/train_model.py
```
This evaluates the Random Forest model and outputs accuracy, classification report, confusion matrix, and feature importances to `models/`.

### 8. Run Automated Tests
```powershell
python -m pytest tests/test_app.py -v
```

### 9. Start the Flask Web Application
```powershell
python app.py
```
The application will launch at:
👉 **`http://127.0.0.1:5000`**

---

## 🔑 Demo Login Credentials

| Role | Username | Password | Privileges |
| :--- | :--- | :--- | :--- |
| **Administrator** | `admin` | `admin123` | Full CRUD, Upload CSV, Model Retraining, View Analytics |
| **City Planner** | `planner` | `planner123` | Real-Time Predictions, View Maps, Scenario Simulator, Analytics |

*Convenient 1-click credential auto-fill buttons are provided on the login page.*

---

## 📓 Running from Jupyter Notebook

Open and run the notebook for interactive development and viva presentation:
```powershell
jupyter notebook notebooks/accident_hotspot_project.ipynb
```
The notebook contains step-by-step cells for:
* Dependency installation (`!pip install -r requirements.txt`)
* Dataset generation (`!python data/generate_data.py`)
* EDA & GeoPandas visualizations
* Random Forest model training & metrics evaluation
* Starting Flask as a background thread directly inside the notebook!

---

## 🌐 REST API Endpoints

* `GET /api/accidents?limit=200&risk=High` - Retrieve accident records.
* `GET /api/hotspots` - Retrieve identified spatial cluster centroids and risk probabilities.
* `POST /api/predict` - Make real-time predictions.
  * Request Body:
    ```json
    {
      "latitude": 12.9176,
      "longitude": 77.6234,
      "traffic_volume": 1050,
      "hour": 19,
      "day_of_week": 4,
      "month": 6,
      "weather": "Rainy",
      "road_condition": "Wet",
      "road_type": "Major Intersection",
      "visibility": 2.0,
      "speed_limit": 50,
      "previous_accidents": 6,
      "vehicle_count": 2
    }
    ```
  * Response:
    ```json
    {
      "status": "success",
      "risk_level": "High",
      "probability": 0.9966,
      "latitude": 12.9176,
      "longitude": 77.6234,
      "contributing_factors": [...]
    }
    ```
* `GET /api/analytics-data` - Returns aggregated time-series and categorical data for the 8 Chart.js charts.
* `GET /api/model-metrics` - Returns accuracy, confusion matrix, and feature importances.
* `POST /api/retrain` - Retrains the model from existing database records.
