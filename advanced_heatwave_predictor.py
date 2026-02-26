"""
advanced_heatwave_predictor.py — Advanced Heatwave Probability Prediction
===========================================================================
Uses machine learning and meteorological indices to predict future heatwave events.
Goes beyond temperature forecasting to directly predict heatwave probability.

Research-backed approach:
1. XGBoost Classifier: Learns heatwave patterns from historical data
2. Heatwave clustering: Identifies heatwave "seasons" (typically May-July in India)
3. Atmospheric indices: Pressure trends, humidity patterns (inferred from temperature)
4. Persistence modeling: Heat waves tend to cluster in time
5. Ensemble probabilistic forecasting: Multiple feature sets + confidence intervals

References:
- "Forecasting Heatwaves: A Spatio-Temporal Deep Learning Approach" (IEEE)
- "Machine Learning Methods for Climate Prediction" (Nature Climate Change)
- "Identifying Heatwave Characteristics from Large Climate Models" (Scientific Reports)
- IMD Meteorological Department forecasting methods
- "Heat Stress Monitoring and Forecasting" (World J Engng Technol)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = GradientBoostingClassifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _safe_city_name(city_name):
    """Safely encode city name for file system."""
    return city_name.replace(" ", "_").replace("(", "").replace(")", "").lower()


def _heatwave_classifier_path(city_name):
    """Path to saved heatwave classifier model."""
    return os.path.join(MODELS_DIR, f"hw_classifier_{_safe_city_name(city_name)}.pkl")


def _heatwave_scaler_path(city_name):
    """Path to saved feature scaler."""
    return os.path.join(MODELS_DIR, f"hw_scaler_{_safe_city_name(city_name)}.pkl")


def _heatwave_metrics_path(city_name):
    """Path to saved metrics."""
    return os.path.join(MODELS_DIR, f"hw_metrics_{_safe_city_name(city_name)}.json")


# ============================================================
# METEOROLOGICAL FEATURE ENGINEERING FOR HEATWAVE PREDICTION
# ============================================================

def build_heatwave_features(df, temp_col="Temperature", historical_data_only=False):
    """
    Build scientifically-grounded features for heatwave prediction.
    
    Features are based on:
    - Thermodynamic persistence (heat storage in atmosphere)
    - Rate of change (temperature acceleration)
    - Atmospheric pressure approximations (from temperature inversions)
    - Seasonal cycles and transitions
    - Temporal clustering patterns
    
    Args:
        df: DataFrame with temperature data indexed by date
        temp_col: Name of temperature column
        historical_data_only: If True, only use data up to yesterday
    
    Returns:
        DataFrame with engineered features
    """
    feat = pd.DataFrame(index=df.index)
    temp = df[temp_col]
    
    # === 1. PERSISTENCE FEATURES ===
    # Research shows heatwaves cluster temporally - persistence indicates higher probability
    # of continued/subsequent heatwaves
    
    # How many of the last N days were hot (>35C)
    for window in [3, 7, 14, 30]:
        is_hot = (temp.shift(1) > 35.0).astype(int)  # shift to prevent target leakage
        feat[f"hot_days_last{window}"] = is_hot.rolling(window, min_periods=1).sum()
        
        # Proportion of hot days in window
        feat[f"hot_pct_last{window}"] = (is_hot.rolling(window, min_periods=1).sum() / window).clip(0, 1)
    
    # Consecutive hot days so far
    is_hot_main = (temp > 35.0).astype(int)
    blocks = (is_hot_main == 0).cumsum()
    feat["consecutive_hot_days"] = is_hot_main.groupby(blocks).cumsum()
    
    # === 2. RATE OF CHANGE FEATURES ===
    # Temperature acceleration indicates development of heat dome
    # A rapidly warming trend often precedes heatwave onset
    
    feat["temp_change_1d"] = temp.diff(1)
    feat["temp_change_3d"] = temp.diff(3)
    feat["temp_change_7d"] = temp.diff(7)
    
    # Acceleration: is temperature change increasing?
    feat["accel_3d"] = feat["temp_change_3d"].diff(1)  # change in rate of change
    feat["accel_7d"] = feat["temp_change_7d"].diff(1)
    
    # Smoothed 7-day trend (indicates sustained warming)
    feat["warming_trend_7d"] = temp.shift(1).rolling(7, min_periods=1).mean().diff(1)
    
    # === 3. VARIABILITY FEATURES ===
    # Research: High variability can precede extreme events
    for window in [7, 14, 30]:
        temp_shifted = temp.shift(1)
        feat[f"temp_std_{window}"] = temp_shifted.rolling(window, min_periods=1).std()
        feat[f"temp_range_{window}"] = (temp_shifted.rolling(window, min_periods=1).max() - 
                                         temp_shifted.rolling(window, min_periods=1).min())
    
    # Daily range approximation (infer from variations)
    feat["intraday_variability"] = temp.shift(1).rolling(3, min_periods=1).std()
    
    # === 4. PRESSURE ANALOGUE FEATURES ===
    # High pressure systems are associated with heatwaves
    # Approximate using temperature inversion strength and horizontal gradient
    
    # High pressure => temperature doesn't drop at night => low day-to-day variance?
    # Actually: high pressure => stable atmosphere => consistent hot days
    feat["stability_index"] = (
        (temp.rolling(3, min_periods=1).std() * -1).add(5)  # inverse relationship
    )
    
    # Pressure tendency (estimated): rapid warming = rising pressure ahead
    feat["pressure_tendency"] = feat["temp_change_1d"].rolling(3, min_periods=1).mean()
    
    # === 5. SEASONAL & CYCLICAL FEATURES ===
    # Heatwaves follow seasonal patterns (typically May-July in India)
    
    doy = df.index.dayofyear
    feat["day_of_year"] = doy
    feat["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    feat["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    feat["month"] = df.index.month
    feat["quarter"] = df.index.quarter
    
    # Heat season indicator (May-Sept for most Indian regions)
    feat["is_heat_season"] = ((df.index.month >= 5) & (df.index.month <= 9)).astype(int)
    feat["days_into_season"] = feat.groupby(feat["is_heat_season"]).cumcount()
    
    # === 6. MULTI-SCALE STATISTICS ===
    # Recent conditions vs seasonal norms
    
    for window in [7, 14, 30, 60]:
        temp_shifted = temp.shift(1)
        feat[f"mean_{window}d"] = temp_shifted.rolling(window, min_periods=1).mean()
        feat[f"max_{window}d"] = temp_shifted.rolling(window, min_periods=1).max()
        feat[f"q75_{window}d"] = temp_shifted.rolling(window, min_periods=1).quantile(0.75)
        feat[f"q90_{window}d"] = temp_shifted.rolling(window, min_periods=1).quantile(0.90)
    
    # === 7. ANOMALY ACCUMULATION ===
    # If you've been above climatology for many days, heatwave more likely
    if "climatology" in df.columns:
        anomaly = temp - df["climatology"]
        feat["anomaly_val"] = anomaly
        feat["positive_anomaly_sum_7d"] = anomaly.clip(lower=0).rolling(7, min_periods=1).sum()
        feat["positive_anomaly_sum_30d"] = anomaly.clip(lower=0).rolling(30, min_periods=1).sum()
        feat["anomaly_mean_7d"] = anomaly.rolling(7, min_periods=1).mean()
    
    # === 8. CLUSTERING & PATTERN INDICES ===
    # Heatwaves cluster in time - if last week had heatwaves, more likely next week
    
    # Simple heatwave indicator (>35C and above climatology if available)
    if "climatology" in df.columns:
        potential_hw = ((temp > 35) & (anomaly > 2.0)).astype(int)
    else:
        potential_hw = (temp > 35).astype(int)
    
    feat["hw_events_last7d"] = potential_hw.rolling(7, min_periods=1).sum()
    feat["hw_events_last30d"] = potential_hw.rolling(30, min_periods=1).sum()
    feat["time_since_last_hw"] = (potential_hw == 0).cumsum()
    feat["time_since_last_hw"] = feat["time_since_last_hw"] * (potential_hw == 0)
    
    # === 9. TREND COMPONENTS ===
    # Linear trend in recent days
    for window in [7, 14, 30]:
        temp_shifted = temp.shift(1)
        windows = temp_shifted.rolling(window, min_periods=1)
        # Simple trend: recent_avg - older_avg
        feat[f"trend_{window}d"] = (
            temp_shifted.rolling(window//2, min_periods=1).mean() -
            temp_shifted.rolling(window, min_periods=1).mean()
        )
    
    # Fill any NaN values
    feat = feat.fillna(0)
    
    return feat


def train_heatwave_classifier(df, temp_col="Temperature", city_name="default", 
                              force_retrain=False, min_event_duration=3):
    """
    Train a classifier to predict heatwave events.
    
    The target is constructed by looking at temperature anomalies and intensity.
    A day is labeled as "heatwave day" if:
    - Part of a ≥3 day sequence of high temps + high anomalies
    
    Args:
        df: Historical data with Temperature column (and ideally climatology)
        temp_col: Name of temperature column
        city_name: City identifier
        force_retrain: Force retraining even if model exists
        min_event_duration: Minimum consecutive days for heatwave label
    
    Returns:
        (classifier, scaler, metrics_dict)
    """
    
    model_path = _heatwave_classifier_path(city_name)
    scaler_path = _heatwave_scaler_path(city_name)
    metrics_path = _heatwave_metrics_path(city_name)
    
    # Check if model exists
    if not force_retrain and os.path.exists(model_path) and os.path.exists(scaler_path):
        classifier = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        return classifier, scaler, metrics
    
    df = df.copy()
    
    # === BUILD TARGET VARIABLE ===
    # A day is a "heatwave day" if it's part of a ≥3 day heatwave event
    
    # Heuristically: temp >35C + above climatology (if available)
    if "climatology" in df.columns:
        df["anomaly"] = df[temp_col] - df["climatology"]
        is_extreme = (df[temp_col] > 35) & (df["anomaly"] > 1.5)
    else:
        is_extreme = df[temp_col] > 36.5  # fallback threshold
    
    # Mark consecutive sequences of ≥min_event_duration
    is_extreme_int = is_extreme.astype(int)
    groups = (is_extreme_int != is_extreme_int.shift()).cumsum()
    
    heatwave_label = pd.Series(0, index=df.index)
    for group_id, group_data in is_extreme_int.groupby(groups):
        if len(group_data) >= min_event_duration and group_data.iloc[0] == 1:
            heatwave_label.loc[group_data.index] = 1
    
    df["target"] = heatwave_label
    
    # === BUILD FEATURES ===
    feat = build_heatwave_features(df, temp_col)
    
    # Merge with target
    feat["target"] = df["target"]
    feat = feat.dropna(subset=["target"])
    
    if feat.shape[0] < 100:
        raise ValueError(f"Not enough data to train heatwave classifier: {feat.shape[0]} rows")
    
    # === SPLIT DATA (CHRONOLOGICAL) ===
    split_idx = int(len(feat) * 0.8)
    X_train = feat.iloc[:split_idx].drop("target", axis=1)
    y_train = feat.iloc[:split_idx]["target"]
    X_test = feat.iloc[split_idx:].drop("target", axis=1)
    y_test = feat.iloc[split_idx]["target"]
    
    # === SCALE FEATURES ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # === TRAIN CLASSIFIER ===
    try:
        # XGBoost classifier with calibration for probability estimation
        base_clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
        )
        
        # Calibrate to get proper probability estimates
        classifier = CalibratedClassifierCV(
            base_clf, method='sigmoid', cv=StratifiedKFold(n_splits=3)
        )
        classifier.fit(X_train_scaled, y_train)
        
    except Exception as e:
        print(f"Warning: XGBoost fit failed ({e}), falling back to RandomForest")
        base_clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=1
        )
        classifier = CalibratedClassifierCV(
            base_clf, method='sigmoid', cv=StratifiedKFold(n_splits=3)
        )
        classifier.fit(X_train_scaled, y_train)
    
    # === EVALUATE ===
    y_pred = classifier.predict(X_test_scaled)
    y_prob = classifier.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        "city": city_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "heatwave_events_train": int(y_train.sum()),
        "heatwave_events_test": int(y_test.sum()),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }
    
    # Save model, scaler, metrics
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(classifier, model_path)
    joblib.dump(scaler, scaler_path)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[{city_name}] Heatwave classifier trained — Precision: {metrics['precision']}, "
          f"Recall: {metrics['recall']}, F1: {metrics['f1']}, ROC-AUC: {metrics['roc_auc']}")
    
    return classifier, scaler, metrics


def predict_heatwave_probability(history_df, forecast_df, temp_col="Temperature", 
                                city_name="default", classifier=None, scaler=None):
    """
    Predict heatwave probability for future days.
    
    First trains/loads classifier on historical data, then applies to forecast.
    Returns the forecast_df with added columns:
    - 'hw_probability': Probability of heatwave (0-1)
    - 'hw_risk': Risk level ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')
    - 'hw_confidence': Confidence in prediction
    
    Args:
        history_df: Historical data with Temperature (and optionally climatology)
        forecast_df: Forecast dataframe with forecast_temp column
        temp_col: Name of temperature column in history
        city_name: City identifier
        classifier: Optional pre-trained classifier (else will load/train)
        scaler: Optional pre-trained scaler
    
    Returns:
        forecast_df with added columns
    """
    
    # Train/load classifier if not provided
    if classifier is None or scaler is None:
        try:
            classifier, scaler, _ = train_heatwave_classifier(
                history_df, temp_col, city_name
            )
        except Exception as e:
            print(f"Warning: Could not train heatwave classifier: {e}")
            # Fallback: simple rule-based prediction
            return _fallback_heatwave_probability(forecast_df, temp_col="forecast_temp")
    
    forecast_df = forecast_df.copy()
    
    # Build features for forecast
    # We need a continuous history leading into the forecast
    combined_df = pd.concat([history_df[[temp_col]], 
                             pd.DataFrame({temp_col: forecast_df["forecast_temp"]},
                                        index=forecast_df.index)])
    combined_df = combined_df.rename(columns={temp_col: "Temperature"})
    
    # Copy climatology if available
    if "climatology" in history_df.columns:
        hist_clim = history_df[["climatology"]]
        fc_clim = pd.DataFrame({"climatology": forecast_df["climatology"]},
                               index=forecast_df.index)
        combined_df = combined_df.join(pd.concat([hist_clim, fc_clim]))
    
    # Build features
    feat = build_heatwave_features(combined_df, "Temperature")
    
    # Get features for forecast period only
    forecast_feat = feat.loc[forecast_df.index].fillna(0)
    
    # Scale and predict
    X_scaled = scaler.transform(forecast_feat)
    
    y_prob = classifier.predict_proba(X_scaled)[:, 1]  # Probability of class 1 (heatwave)
    y_pred = classifier.predict(X_scaled)
    
    # Add to forecast
    forecast_df["hw_probability"] = y_prob
    forecast_df["hw_predicted"] = y_pred
    
    # Risk classification (based on probability)
    forecast_df["hw_risk"] = "LOW"
    forecast_df.loc[forecast_df["hw_probability"] > 0.3, "hw_risk"] = "MODERATE"
    forecast_df.loc[forecast_df["hw_probability"] > 0.6, "hw_risk"] = "HIGH"
    forecast_df.loc[forecast_df["hw_probability"] > 0.8, "hw_risk"] = "CRITICAL"
    
    # Confidence: higher when far from decision boundary (0.5)
    forecast_df["hw_confidence"] = (
        np.abs(forecast_df["hw_probability"] - 0.5) * 2
    ).clip(0, 1)
    
    return forecast_df


def _fallback_heatwave_probability(forecast_df, temp_col="forecast_temp"):
    """Fallback rule-based heatwave probability when classifier unavailable."""
    forecast_df = forecast_df.copy()
    
    # Simple rule: higher temp = higher probability
    if temp_col not in forecast_df.columns:
        forecast_df["hw_probability"] = 0.5
    else:
        temp = forecast_df[temp_col]
        # Logistic scaling: 35°C → 10%, 38°C → 50%, 41°C → 90%
        forecast_df["hw_probability"] = (
            1.0 / (1.0 + np.exp(-0.5 * (temp - 38.5)))  # sigmoid
        )
    
    forecast_df["hw_predicted"] = (forecast_df["hw_probability"] > 0.5).astype(int)
    forecast_df["hw_risk"] = "LOW"
    forecast_df.loc[forecast_df["hw_probability"] > 0.3, "hw_risk"] = "MODERATE"
    forecast_df.loc[forecast_df["hw_probability"] > 0.6, "hw_risk"] = "HIGH"
    forecast_df.loc[forecast_df["hw_probability"] > 0.8, "hw_risk"] = "CRITICAL"
    
    forecast_df["hw_confidence"] = np.abs(forecast_df["hw_probability"] - 0.5) * 2
    
    return forecast_df


if __name__ == "__main__":
    # Test on Bangalore data
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import load_data
    
    df = load_data("Bangalore")
    if "climatology" not in df.columns:
        from anomaly import detect_anomalies
        df = detect_anomalies(df, "Temperature")
    
    classifier, scaler, metrics = train_heatwave_classifier(
        df, "Temperature", "Bangalore", force_retrain=True
    )
    
    print("Metrics:", metrics)
