"""
prediction_engine.py — Unified Prediction Engine
=================================================
Integrates temperature forecasting, heatwave prediction, and anomaly detection
into a single coherent prediction pipeline.

Provides high-level functions for comprehensive future event forecasting
without modifying existing code.

This module coordinates:
1. Temperature forecasting (existing)
2. Heatwave probability prediction (new)
3. Anomaly magnitude prediction (new)
4. Risk assessment and early warning signals
"""

import warnings
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

warnings.filterwarnings("ignore")


def comprehensive_prediction(df_history, forecast_df, city_name, temp_col="Temperature",
                            force_retrain_classifiers=False):
    """
    Comprehensive prediction: temperature + heatwave + anomaly prediction.
    
    Integrates all prediction models into a single output with:
    - Temperature forecast with confidence intervals
    - Heatwave probability and risk level
    - Anomaly prediction (magnitude and direction)
    - Early warning signals
    - Composite risk score
    
    Args:
        df_history: Historical data with Temperature column (and climatology if available)
        forecast_df: DataFrame with forecast_temp from temperature model
        city_name: City identifier
        temp_col: Historical temperature column name
        force_retrain_classifiers: Force retraining ML models
    
    Returns:
        Enhanced forecast_df with all predictions
    """
    
    # Step 1: Ensure input is copy to avoid modifying originals
    forecast_df = forecast_df.copy()
    
    # Step 2: Add heatwave probability predictions
    try:
        from advanced_heatwave_predictor import (
            train_heatwave_classifier, predict_heatwave_probability
        )
        
        classifier, scaler, hw_metrics = train_heatwave_classifier(
            df_history, temp_col, city_name, 
            force_retrain=force_retrain_classifiers
        )
        forecast_df = predict_heatwave_probability(
            df_history, forecast_df, temp_col, city_name,
            classifier=classifier, scaler=scaler
        )
        print(f"✓ Heatwave predictions added for {city_name}")
    except Exception as e:
        print(f"⚠ Heatwave prediction failed: {e}")
        # Fallback columns
        forecast_df["hw_probability"] = 0.5
        forecast_df["hw_risk"] = "MODERATE"
        forecast_df["hw_confidence"] = 0.5
    
    # Step 3: Add anomaly magnitude predictions
    try:
        from advanced_anomaly_predictor import (
            train_anomaly_regressor, predict_anomalies_advanced
        )
        
        regressor, scaler, anom_metrics = train_anomaly_regressor(
            df_history, temp_col, city_name,
            force_retrain=force_retrain_classifiers
        )
        forecast_df = predict_anomalies_advanced(
            df_history, forecast_df, temp_col, city_name,
            regressor=regressor, scaler=scaler
        )
        print(f"✓ Anomaly predictions added for {city_name}")
    except Exception as e:
        print(f"⚠ Anomaly prediction failed: {e}")
        # Fallback columns
        forecast_df["predicted_anomaly"] = forecast_df.get("anomaly", 0)
        forecast_df["anomaly_confidence"] = 0.5
        forecast_df["extreme_hot_prob"] = 0.3
        forecast_df["anomaly_direction"] = "NORMAL"
    
    # Step 4: Add early warning signals
    forecast_df = _add_early_warning_signals(forecast_df)
    
    # Step 5: Add composite risk score
    forecast_df = _calculate_composite_risk(forecast_df)
    
    return forecast_df


def _add_early_warning_signals(forecast_df) -> pd.DataFrame:
    """
    Add early warning indicators for upcoming extreme events.
    
    Signals include:
    - Rapid temperature increase (setup phase)
    - Strong heatwave probability + high anomaly
    - Sustained high temperatures (persistence)
    - Acceleration into heatwave regime
    """
    forecast_df = forecast_df.copy()
    
    # === TEMPERATURE ACCELERATION ===
    if "forecast_temp" in forecast_df.columns:
        temp = forecast_df["forecast_temp"]
        forecast_df["temp_accel_3d"] = temp.diff(3)
        forecast_df["temp_accel_7d"] = temp.diff(7)
        
        # Setup phase: warming trend
        forecast_df["setup_warming"] = (
            ((forecast_df["temp_accel_7d"] > 1.0) & 
             (forecast_df["temp_accel_3d"] > 0.5)).astype(int)
        )
    
    # === STRONG ANOMALY + HW PROBABILITY ===
    has_anom = "predicted_anomaly" in forecast_df.columns
    has_hw = "hw_probability" in forecast_df.columns
    
    if has_anom and has_hw:
        strong_combo = (
            (forecast_df["predicted_anomaly"] > 2.5) &
            (forecast_df["hw_probability"] > 0.6)
        )
        forecast_df["danger_combination"] = strong_combo.astype(int)
    
    # === PERSISTENCE INDEX ===
    # How many days in a row are predicted to be hot/anomalous?
    if "forecast_temp" in forecast_df.columns:
        is_hot = (forecast_df["forecast_temp"] > 35).astype(int)
        blocks = (is_hot != is_hot.shift()).cumsum()
        forecast_df["consecutive_hot_days"] = is_hot.groupby(blocks).cumsum()
    
    if has_anom:
        is_positive_anom = (forecast_df["predicted_anomaly"] > 1.0).astype(int)
        blocks = (is_positive_anom != is_positive_anom.shift()).cumsum()
        forecast_df["consecutive_anomaly_days"] = is_positive_anom.groupby(blocks).cumsum()
    
    # === RAPID ONSET WARNING ===
    # Sudden jump from normal to heatwave conditions
    if has_hw:
        forecast_df["hw_onset_warning"] = 0
        # Look for transitions from low to high probability
        forecast_df.loc[
            (forecast_df["hw_probability"].shift(3) < 0.3) &
            (forecast_df["hw_probability"] > 0.6),
            "hw_onset_warning"
        ] = 1
    
    return forecast_df


def _calculate_composite_risk(forecast_df) -> pd.DataFrame:
    """
    Calculate composite risk score combining multiple factors.
    
    Risk Score (0-100) incorporates:
    - Heatwave probability (40% weight)
    - Anomaly magnitude (30% weight)
    - Temperature level (20% weight)
    - Persistence (10% weight)
    
    Returns:
        forecast_df with 'composite_risk' and 'risk_level' columns
    """
    forecast_df = forecast_df.copy()
    
    # Component 1: Heatwave probability (0-1 → 0-40 points)
    hw_score = forecast_df.get("hw_probability", 0) * 40 if "hw_probability" in forecast_df else 0
    
    # Component 2: Anomaly magnitude (0-1 → 0-30 points)
    if "predicted_anomaly" in forecast_df.columns:
        # Normalize anomaly: 0°C → 0 points, 5°C → 30 points (logistic)
        anom_score = (
            30.0 / (1.0 + np.exp(-0.5 * (forecast_df["predicted_anomaly"] - 2.5)))
        )
    else:
        anom_score = 0
    
    # Component 3: Absolute temperature (32°C → 0, 42°C → 20 points)
    if "forecast_temp" in forecast_df.columns:
        temp_score = (
            20.0 * ((forecast_df["forecast_temp"] - 32) / 10.0).clip(0, 1)
        )
    else:
        temp_score = 0
    
    # Component 4: Persistence (consecutive hot days - up to 10 points)
    if "consecutive_hot_days" in forecast_df.columns:
        persist_score = (
            10.0 * (forecast_df["consecutive_hot_days"] / 
                   forecast_df["consecutive_hot_days"].max().clip(lower=1))
        )
    else:
        persist_score = 0
    
    # Composite score
    forecast_df["composite_risk"] = (hw_score + anom_score + temp_score + persist_score).clip(0, 100)
    
    # Risk level classification
    forecast_df["risk_level"] = "LOW"
    forecast_df.loc[forecast_df["composite_risk"] > 25, "risk_level"] = "MODERATE"
    forecast_df.loc[forecast_df["composite_risk"] > 50, "risk_level"] = "HIGH"
    forecast_df.loc[forecast_df["composite_risk"] > 75, "risk_level"] = "CRITICAL"
    
    return forecast_df


def generate_forecast_summary(forecast_df, city_name: str) -> Dict:
    """
    Generate a human-readable summary of the forecast.
    
    Args:
        forecast_df: Forecast DataFrame with predictions
        city_name: City name for reporting
    
    Returns:
        Dictionary with summary statistics and alerts
    """
    
    summary = {
        "city": city_name,
        "forecast_days": len(forecast_df),
        "forecast_period": f"{forecast_df.index.min().date()} to {forecast_df.index.max().date()}",
    }
    
    # Temperature statistics
    if "forecast_temp" in forecast_df.columns:
        summary["temp_stats"] = {
            "min": round(forecast_df["forecast_temp"].min(), 1),
            "max": round(forecast_df["forecast_temp"].max(), 1),
            "mean": round(forecast_df["forecast_temp"].mean(), 1),
            "std": round(forecast_df["forecast_temp"].std(), 1),
        }
    
    # Heatwave outlook
    if "heatwave_event" in forecast_df.columns:
        hw_days = forecast_df["heatwave_event"].sum()
        summary["heatwave_days_predicted"] = int(hw_days)
        summary["heatwave_percentage"] = round(100 * hw_days / len(forecast_df), 1)
    else:
        summary["heatwave_days_predicted"] = 0
        summary["heatwave_percentage"] = 0
    
    # Heatwave probability
    if "hw_probability" in forecast_df.columns:
        summary["avg_hw_probability"] = round(forecast_df["hw_probability"].mean(), 3)
        summary["max_hw_probability"] = round(forecast_df["hw_probability"].max(), 3)
        summary["critical_hw_days"] = int((forecast_df["hw_probability"] > 0.8).sum())
    
    # Anomaly outlook
    if "predicted_anomaly" in forecast_df.columns:
        summary["anomaly_stats"] = {
            "min": round(forecast_df["predicted_anomaly"].min(), 2),
            "max": round(forecast_df["predicted_anomaly"].max(), 2),
            "mean": round(forecast_df["predicted_anomaly"].mean(), 2),
        }
        summary["extreme_hot_days"] = int(
            (forecast_df["predicted_anomaly"] > 3.0).sum()
        )
    
    # Risk assessment
    if "composite_risk" in forecast_df.columns:
        summary["avg_risk_score"] = round(forecast_df["composite_risk"].mean(), 1)
        summary["max_risk_score"] = round(forecast_df["composite_risk"].max(), 1)
        
        risk_dist = forecast_df["risk_level"].value_counts().to_dict()
        summary["risk_distribution"] = risk_dist
    
    # Early warnings
    early_warnings = []
    
    if "setup_warming" in forecast_df.columns and forecast_df["setup_warming"].any():
        setup_start = forecast_df[forecast_df["setup_warming"]].index[0]
        early_warnings.append(
            f"Setup phase detected: Rapid warming trend starting {setup_start.date()}"
        )
    
    if "danger_combination" in forecast_df.columns and forecast_df["danger_combination"].any():
        danger_days = forecast_df[forecast_df["danger_combination"]].index
        early_warnings.append(
            f"Dangerous combination of high heatwave probability + strong anomaly: "
            f"{len(danger_days)} days affected"
        )
    
    if "hw_onset_warning" in forecast_df.columns and forecast_df["hw_onset_warning"].any():
        onset_date = forecast_df[forecast_df["hw_onset_warning"]].index[0]
        early_warnings.append(
            f"Rapid heatwave onset alert: Conditions transition to high risk around {onset_date.date()}"
        )
    
    summary["early_warnings"] = early_warnings if early_warnings else ["No major warnings detected"]
    
    return summary


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "./")
    from utils import load_data
    from anomaly import detect_anomalies
    from model_training import train_xgb_model, train_sarima_model
    from forecast import predict_future
    
    # Load data
    city = "Bangalore"
    df = load_data(city)
    df = df.loc["2015":"2022"]
    
    # Detect anomalies
    df = detect_anomalies(df, "Temperature")
    
    # Train models
    xgb, metrics = train_xgb_model(df, "Temperature", city, force_retrain=True)
    sarima = train_sarima_model(df, "Temperature", city, force_retrain=True)
    
    # Forecast
    forecast_df, _, _ = predict_future(df, "Temperature", days=90, city_name=city,
                                       xgb_model=xgb, model=sarima)
    
    # Comprehensive prediction
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE FORECAST FOR {city.upper()}")
    print(f"{'='*60}\n")
    
    forecast_df = comprehensive_prediction(df, forecast_df, city)
    
    summary = generate_forecast_summary(forecast_df, city)
    
    import json
    print(json.dumps(summary, indent=2))
    
    # Show sample forecast
    print(f"\n{'='*60}")
    print("SAMPLE 10-DAY FORECAST")
    print(f"{'='*60}\n")
    
    sample = forecast_df[["forecast_temp", "hw_probability", "predicted_anomaly", 
                          "composite_risk", "risk_level"]].head(10)
    print(sample.to_string())
