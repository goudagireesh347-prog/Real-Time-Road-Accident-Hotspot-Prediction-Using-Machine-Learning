import os
import sys
from pathlib import Path
import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.preprocessing import AccidentPreprocessor, RISK_CLASSES

def train_and_save_model(data_path=None, model_dir=None):
    """
    Trains the Random Forest accident risk prediction model,
    evaluates classification performance, and persists artifacts.
    """
    if data_path is None:
        data_path = BASE_DIR / "data" / "accident_data.csv"
    if model_dir is None:
        model_dir = BASE_DIR / "models"

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading accident dataset from: {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records. Risk level distribution:\n{df['risk_level'].value_counts()}\n")

    # 1. Initialize and fit preprocessor
    print("Fitting preprocessor & encoding spatial/temporal features...")
    preprocessor = AccidentPreprocessor()
    preprocessor.fit(df)
    X = preprocessor.transform(df)
    y = df['risk_level'].values

    # 2. Train-Test Split (80% train, 20% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

    # 3. Train Random Forest Classifier
    print("Training Random Forest Classifier (n_estimators=120, max_depth=16)...")
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 4. Model Evaluation
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', labels=RISK_CLASSES)
    rec = recall_score(y_test, y_pred, average='weighted', labels=RISK_CLASSES)
    f1 = f1_score(y_test, y_pred, average='weighted', labels=RISK_CLASSES)
    cm = confusion_matrix(y_test, y_pred, labels=RISK_CLASSES)

    print("\n" + "=" * 55)
    print("MODEL EVALUATION RESULTS (MCA PROJECT BENCHMARK)")
    print("=" * 55)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}% (Weighted)")
    print(f"Recall    : {rec * 100:.2f}% (Weighted)")
    print(f"F1-Score  : {f1 * 100:.2f}% (Weighted)")
    print("\nConfusion Matrix (Low, Medium, High):")
    print(cm)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=RISK_CLASSES))

    # 5. Extract Feature Importances
    importances = rf.feature_importances_
    feature_names = preprocessor.feature_names
    feat_imp = []
    for name, score in zip(feature_names, importances):
        feat_imp.append({"feature": name, "importance": round(float(score), 4)})
    feat_imp.sort(key=lambda x: x["importance"], reverse=True)

    # 6. Save Artifacts using Joblib
    model_path = model_dir / "accident_hotspot_model.pkl"
    preprocessor_path = model_dir / "preprocessing.pkl"
    metrics_path = model_dir / "model_metrics.pkl"

    joblib.dump(rf, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    metrics_payload = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "labels": RISK_CLASSES,
        "feature_importance": feat_imp[:12], # top 12 features for visualization
        "training_samples": int(X_train.shape[0]),
        "testing_samples": int(X_test.shape[0]),
        "total_samples": int(len(df)),
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(metrics_payload, metrics_path)

    print(f"\nArtifacts successfully persisted:")
    print(f" - Model: {model_path}")
    print(f" - Preprocessor: {preprocessor_path}")
    print(f" - Metrics: {metrics_path}")
    print("=" * 55)

    return metrics_payload

if __name__ == "__main__":
    train_and_save_model()
