"""
train_all_models.py — Retrain all models with improved features
================================================================
Run this script locally before deploying to Render or GitHub.

It retrains:
1. XGBoost temperature models (per city)
2. Heatwave classifiers (per city)  
3. Anomaly regressors (per city)

SARIMA models are NOT retrained here (too slow, they're pre-trained).
The script enables force_retrain to rebuild .pkl files with new features
(years_since_2000, ENSO proxy, warming trend etc.)

Usage:
    python train_all_models.py            # Train all cities
    python train_all_models.py Bangalore  # Train single city
"""

import sys
import os
import time

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_data, get_available_cities
from anomaly import detect_anomalies
from model_training import train_xgb_model
from advanced_heatwave_predictor import train_heatwave_classifier
from advanced_anomaly_predictor import train_anomaly_regressor


def train_city(city_name, temp_col="Temperature"):
    """Train all models for a single city."""
    print(f"\n{'='*60}")
    print(f"  TRAINING: {city_name}")
    print(f"{'='*60}")
    
    start = time.time()
    
    # Load and prepare data
    df = load_data(city_name)
    if df.empty:
        print(f"  ⚠ No data for {city_name}, skipping.")
        return
    
    print(f"  Data: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
    
    # Prepare continuous daily series
    model_df = df[[temp_col]].copy()
    model_df = model_df.sort_index()
    model_df = model_df.asfreq("D")
    model_df[temp_col] = model_df[temp_col].interpolate(method="linear", limit=7)
    model_df[temp_col] = model_df[temp_col].ffill().bfill()
    model_df = model_df.dropna(subset=[temp_col])
    
    # Detect anomalies (adds climatology, anomaly_val, etc.)
    model_df = detect_anomalies(model_df, temp_col)
    
    # 1. Train XGBoost temperature model
    print(f"\n  [1/3] Training XGBoost temperature model...")
    try:
        xgb_model, xgb_metrics = train_xgb_model(
            model_df, temp_col, city_name, force_retrain=True
        )
        print(f"  ✓ XGBoost — MAE: {xgb_metrics.get('mae', '?')}°C, "
              f"R²: {xgb_metrics.get('r2', '?')}")
    except Exception as e:
        print(f"  ✗ XGBoost failed: {e}")
    
    # 2. Train heatwave classifier
    print(f"\n  [2/3] Training heatwave classifier...")
    try:
        hw_clf, hw_scaler, hw_metrics = train_heatwave_classifier(
            model_df, temp_col, city_name, force_retrain=True
        )
        print(f"  ✓ Heatwave — Precision: {hw_metrics.get('precision', '?')}, "
              f"Recall: {hw_metrics.get('recall', '?')}, "
              f"F1: {hw_metrics.get('f1', '?')}")
    except Exception as e:
        print(f"  ✗ Heatwave classifier failed: {e}")
    
    # 3. Train anomaly regressor
    print(f"\n  [3/3] Training anomaly regressor...")
    try:
        anom_reg, anom_scaler, anom_metrics = train_anomaly_regressor(
            model_df, temp_col, city_name, force_retrain=True
        )
        print(f"  ✓ Anomaly — MAE: {anom_metrics.get('mae_test', '?')}°C, "
              f"R²: {anom_metrics.get('r2_test', '?')}")
    except Exception as e:
        print(f"  ✗ Anomaly regressor failed: {e}")
    
    elapsed = time.time() - start
    print(f"\n  ⏱ {city_name} done in {elapsed:.1f}s")


def main():
    """Train all or selected cities."""
    
    # Check if specific city was requested
    if len(sys.argv) > 1:
        cities = [sys.argv[1]]
    else:
        cities = get_available_cities()
    
    print("=" * 60)
    print("  HEATWAVE PREDICTION — MODEL TRAINING")
    print(f"  Cities: {', '.join(cities)}")
    print("=" * 60)
    
    total_start = time.time()
    
    for city in cities:
        try:
            train_city(city)
        except Exception as e:
            print(f"\n  ✗✗ FAILED for {city}: {e}")
            import traceback
            traceback.print_exc()
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL DONE in {total_elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n  Models saved to: models/")
    print(f"  Now commit & push to GitHub for Render deployment.")


if __name__ == "__main__":
    main()
