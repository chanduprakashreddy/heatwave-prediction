"""
advanced_anomaly_predictor.py — Advanced Temperature Anomaly Prediction
========================================================================
Direct prediction of future temperature anomalies using machine learning.
Detects not just historical anomalies, but predicts where future anomalies will occur.

Research-backed approach:
1. Anomaly as continuous variable: Predict actual deviation from climatology
2. Regression + Uncertainty quantification: How much above/below normal?
3. Extreme event probability: P(T > 95th percentile)?
4. Seasonal normalization: Account for changing anomaly thresholds by season
5. Autocorrelation structures: Anomalies persist due to atmospheric patterns

References:
- "Deep Learning for Climate Extremes" (Nature Climate Change)
- "Predicting Extreme Weather and Climate Events" (Springer)
- "Quantile Regression for Weather Forecasting" (Journal of Climate)
- "Machine Learning for Climate Downscaling" (PNAS)
- IMD operational anomaly forecasting methods
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _safe_city_name(city_name):
    """Safely encode city name for file system."""
    return city_name.replace(" ", "_").replace("(", "").replace(")", "").lower()


def _anomaly_regressor_path(city_name):
    """Path to saved anomaly amount regressor."""
    return os.path.join(MODELS_DIR, f"anomaly_reg_{_safe_city_name(city_name)}.pkl")


def _anomaly_scaler_path(city_name):
    """Path to saved scaler for anomaly features."""
    return os.path.join(MODELS_DIR, f"anomaly_scaler_{_safe_city_name(city_name)}.pkl")


def _anomaly_metrics_path(city_name):
    """Path to saved anomaly model metrics."""
    return os.path.join(MODELS_DIR, f"anomaly_metrics_{_safe_city_name(city_name)}.json")


# ============================================================
# ADVANCED ANOMALY FEATURE ENGINEERING
# ============================================================

def build_anomaly_prediction_features(df, temp_col="Temperature", 
                                      clim_mean=None, clim_std=None):
    """
    Build features specifically for predicting temperature anomalies.
    
    Unlike classification (is it anomalous?), we predict the magnitude and direction
    of the anomaly (how many degrees above/below normal?).
    
    Features capture:
    - Current state of anomaly
    - Trends and persistence
    - Seasonal modulation
    - Non-linear interactions
    
    Args:
        df: DataFrame with Temperature indexed by date
        temp_col: Temperature column name
        clim_mean: Climatological mean (day-of-year map) or None to compute
        clim_std: Climatological std (day-of-year map) or None to compute
    
    Returns:
        Feature DataFrame
    """
    feat = pd.DataFrame(index=df.index)
    temp = df[temp_col]
    
    # === 1. COMPUTE CLIMATOLOGY IF NOT PROVIDED ===
    if clim_mean is None:
        doy = df.index.dayofyear
        clim_mean = temp.groupby(doy).mean()
        clim_mean_series = pd.Series(doy, index=df.index).map(clim_mean)
    else:
        clim_mean_series = pd.Series(df.index.dayofyear, index=df.index).map(clim_mean)
    
    if clim_std is None:
        doy = df.index.dayofyear
        clim_std_map = temp.groupby(doy).std()
        clim_std = pd.Series(doy, index=df.index).map(clim_std_map)
    else:
        clim_std = pd.Series(df.index.dayofyear, index=df.index).map(clim_std)
    
    # Current anomaly value
    current_anomaly = temp - clim_mean_series
    feat["anomaly_current"] = current_anomaly
    
    # Standardized anomaly (in sigma units)
    clim_std = clim_std.fillna(clim_std.mean())
    clim_std = clim_std.replace(0, clim_std.mean())
    feat["zscore"] = current_anomaly / clim_std
    
    # === 2. ANOMALY PERSISTENCE ===
    # Anomalies persist due to dominant atmospheric patterns
    # Past anomaly is strong predictor of future anomaly
    
    for lag in [1, 2, 3, 7, 14, 30]:
        feat[f"anomaly_lag{lag}"] = current_anomaly.shift(lag)
        feat[f"zscore_lag{lag}"] = feat["zscore"].shift(lag)
    
    # --- Anomaly trend ---
    feat["anomaly_trend_3d"] = current_anomaly.diff(3)
    feat["anomaly_trend_7d"] = current_anomaly.diff(7)
    feat["anomaly_accel"] = feat["anomaly_trend_3d"].diff(1)
    
    # === 3. MULTI-SCALE ANOMALY INDICATORS ===
    # Average anomaly over different windows indicates "anomaly regime"
    
    for window in [7, 14, 30]:
        anomaly_shifted = current_anomaly.shift(1)
        feat[f"mean_anom_{window}d"] = anomaly_shifted.rolling(window, min_periods=1).mean()
        feat[f"std_anom_{window}d"] = anomaly_shifted.rolling(window, min_periods=1).std()
        feat[f"max_anom_{window}d"] = anomaly_shifted.rolling(window, min_periods=1).max()
        feat[f"min_anom_{window}d"] = anomaly_shifted.rolling(window, min_periods=1).min()
    
    # === 4. REGIME INDICATORS ===
    # Probability of being in "hot anomaly regime" vs "cold anomaly regime"
    
    positive_anom = (current_anomaly > 0).astype(int)
    strong_positive = (current_anomaly > 2.0).astype(int)
    
    feat["hot_anom_streak"] = (positive_anom == 1).groupby(
        (positive_anom != positive_anom.shift()).cumsum()
    ).cumsum() * positive_anom
    
    feat["strong_hot_streak"] = (strong_positive == 1).groupby(
        (strong_positive != strong_positive.shift()).cumsum()
    ).cumsum() * strong_positive
    
    # === 5. SEASONAL/CYCLICAL MODULATION ===
    # Anomaly thresholds vary by season
    
    doy = df.index.dayofyear
    feat["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    feat["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    feat["month"] = df.index.month
    feat["is_summer"] = ((df.index.month >= 4) & (df.index.month <= 8)).astype(int)
    feat["is_winter"] = ((df.index.month >= 11) | (df.index.month <= 2)).astype(int)
    
    # Seasonal climatology variations
    monthly_mean = temp.groupby(df.index.month).mean()
    feat["seasonal_baseline"] = df.index.month.map(monthly_mean)
    
    # === 6. EXTREME ANOMALY INDICATORS ===
    # Interaction features for extreme anomalies
    
    feat["temp_abs"] = np.abs(temp)
    feat["anom_abs"] = np.abs(current_anomaly)
    feat["high_temp_high_anom"] = ((temp > 35) & (current_anomaly > 2.0)).astype(int)
    
    # === 7. AUTOCORRELATION STRUCTURE ===
    # Anomalies have autocorrelation (red noise)
    # Capture this with exponential decay features
    
    # Exponential moving average of anomaly (with different decay rates)
    for alpha in [0.2, 0.5, 0.9]:
        feat[f"ema_anom_{alpha}"] = current_anomaly.ewm(alpha=alpha).mean()
    
    # === 8. ACCELERATION & CURVATURE ===
    # Second-order derivatives capture turning points
    
    feat["temp_accel"] = temp.diff(1).diff(1)
    feat["anomaly_curvature"] = current_anomaly.diff(1).diff(1)
    
    # === 9. COMPOSITE ANOMALY INDICES ===
    
    # Heat stress index (approximation from temp + humidity via persistence)
    # Humidity persistsapproximated by slow changes in temperature
    feat["heat_index_approx"] = (
        1.8 * temp + 32 - 0.55 * (1 - positive_anom * 0.3) * (temp - 58)
    )
    
    # Anomaly magnitude class
    feat["anomaly_class_0"] = ((np.abs(current_anomaly) <= 1.0)).astype(int)  # Normal
    feat["anomaly_class_1"] = ((np.abs(current_anomaly) > 1.0) & 
                               (np.abs(current_anomaly) <= 2.5)).astype(int)  # Moderate
    feat["anomaly_class_2"] = ((np.abs(current_anomaly) > 2.5)).astype(int)    # Extreme
    
    # === 10. PHASE INFORMATION ===
    # Time-of-day phase (approximated cyclically by day-of-year phase space)
    
    year_phase = (df.index.dayofyear / 365.25) * 2 * np.pi
    feat["annual_phase_sin"] = np.sin(year_phase)
    feat["annual_phase_cos"] = np.cos(year_phase)
    
    # Fill NaNs
    feat = feat.fillna(0)
    feat = feat.replace([np.inf, -np.inf], 0)
    
    return feat


def train_anomaly_regressor(df, temp_col="Temperature", city_name="default",
                           force_retrain=False):
    """
    Train a regressor to predict the magnitude of temperature anomalies.
    
    Unlike classification, this predicts HOW MUCH the temperature will deviate
    from climatology (±°C), capturing the continuous nature of anomalies.
    
    Args:
        df: Historical data with Temperature column
        temp_col: Name of temperature column
        city_name: City identifier
        force_retrain: Force retraining
    
    Returns:
        (regressor, scaler, metrics_dict)
    """
    
    regressor_path = _anomaly_regressor_path(city_name)
    scaler_path = _anomaly_scaler_path(city_name)
    metrics_path = _anomaly_metrics_path(city_name)
    
    # Check if models exist
    if not force_retrain and (
        os.path.exists(regressor_path) and os.path.exists(scaler_path)
    ):
        regressor = joblib.load(regressor_path)
        scaler = joblib.load(scaler_path)
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        return regressor, scaler, metrics
    
    df = df.copy()
    
    # === COMPUTE CLIMATOLOGY ===
    doy = df.index.dayofyear
    clim_mean = df.groupby(doy)[temp_col].mean()
    clim_std = df.groupby(doy)[temp_col].std()
    
    # === BUILD FEATURES & TARGET ===
    feat = build_anomaly_prediction_features(df, temp_col, clim_mean, clim_std)
    
    # Target: magnitude of anomaly (continuous)
    anomaly_target = df[temp_col] - doy.map(clim_mean)
    feat["target"] = anomaly_target
    
    feat = feat.dropna(subset=["target"])
    
    if len(feat) < 100:
        raise ValueError(f"Insufficient data for anomaly regressor: {len(feat)} samples")
    
    # === CHRONOLOGICAL SPLIT ===
    split_idx = int(len(feat) * 0.85)
    X_train = feat.iloc[:split_idx].drop("target", axis=1)
    y_train = feat.iloc[:split_idx]["target"]
    X_test = feat.iloc[split_idx:].drop("target", axis=1)
    y_test = feat.iloc[split_idx:]["target"]
    
    # === FEATURE SCALING ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # === TRAIN REGRESSOR ===
    try:
        regressor = XGBRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
        )
        regressor.fit(X_train_scaled, y_train)
    except Exception as e:
        print(f"Warning: XGBoost fitting failed ({e}), using RandomForest")
        regressor = RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=42, n_jobs=1
        )
        regressor.fit(X_train_scaled, y_train)
    
    # === EVALUATE ===
    y_pred_train = regressor.predict(X_train_scaled)
    y_pred_test = regressor.predict(X_test_scaled)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    metrics = {
        "city": city_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "mae_train": round(mae_train, 4),
        "rmse_train": round(rmse_train, 4),
        "r2_train": round(r2_train, 4),
        "mae_test": round(mae_test, 4),
        "rmse_test": round(rmse_test, 4),
        "r2_test": round(r2_test, 4),
    }
    
    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(regressor, regressor_path)
    joblib.dump(scaler, scaler_path)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[{city_name}] Anomaly regressor trained — "
          f"Test MAE: {mae_test:.3f}°C, RMSE: {rmse_test:.3f}°C, R²: {r2_test:.4f}")
    
    return regressor, scaler, metrics


def predict_anomalies_advanced(history_df, forecast_df, temp_col="Temperature",
                               city_name="default", regressor=None, scaler=None):
    """
    Predict the magnitude of temperature anomalies in the forecast period.
    
    Returns forecast_df with added columns:
    - 'predicted_anomaly': Predicted deviation from climatology (°C)
    - 'anomaly_confidence': Confidence in the prediction (0-1)
    - 'extreme_anomaly_prob': Probability of extreme anomaly (>95th percentile)
    - 'anomaly_direction': 'HOT', 'COLD', or 'NORMAL'
    
    Args:
        history_df: Historical data
        forecast_df: Forecast with forecast_temp column
        temp_col: Historical temp column name
        city_name: City ID
        regressor: Optional pre-trained regressor
        scaler: Optional pre-trained scaler
    
    Returns:
        forecast_df with anomaly predictions
    """
    
    # Train/load regressor if not provided
    if regressor is None or scaler is None:
        try:
            regressor, scaler, _ = train_anomaly_regressor(
                history_df, temp_col, city_name
            )
        except Exception as e:
            print(f"Warning: Could not train anomaly regressor: {e}")
            return _fallback_anomaly_prediction(forecast_df)
    
    forecast_df = forecast_df.copy()
    
    # === BUILD FEATURES FOR FORECAST ===
    
    # Compute climatology from history
    doy = history_df.index.dayofyear
    clim_mean = history_df.groupby(doy)[temp_col].mean()
    clim_std = history_df.groupby(doy)[temp_col].std()
    
    # Combine history + forecast for feature engineering
    combined_df = pd.concat([
        history_df[[temp_col]],
        pd.DataFrame({temp_col: forecast_df["forecast_temp"]},
                    index=forecast_df.index)
    ])
    combined_df = combined_df.rename(columns={temp_col: "Temperature"})
    
    # Build features on combined data
    feat = build_anomaly_prediction_features(combined_df, "Temperature", clim_mean, clim_std)
    
    # Extract features for forecast period
    forecast_feat = feat.loc[forecast_df.index].fillna(0)
    forecast_feat = forecast_feat.replace([np.inf, -np.inf], 0)
    
    # Scale
    X_scaled = scaler.transform(forecast_feat)
    
    # Predict anomaly magnitude
    predicted_anomaly = regressor.predict(X_scaled)
    
    # Estimate prediction uncertainty (residual std from training)
    # For simplicity, we'll use a heuristic based on feature magnitude
    feature_magnitude = np.sqrt((X_scaled ** 2).sum(axis=1))
    sigma = 0.5 + 0.1 * feature_magnitude / feature_magnitude.max()
    
    forecast_df["predicted_anomaly"] = predicted_anomaly
    forecast_df["anomaly_sigma"] = sigma
    
    # Confidence: inverse of uncertainty
    forecast_df["anomaly_confidence"] = 1.0 / (1.0 + forecast_df["anomaly_sigma"])
    
    # Extreme anomaly probability (tail probability > historical 95th percentile)
    hist_anom = history_df[temp_col] - doy.map(clim_mean)
    anom_95 = hist_anom.quantile(0.95)
    anom_5 = hist_anom.quantile(0.05)
    
    # P(X > 95th percentile) approximation via normal tail probability
    from scipy import stats
    forecast_df["extreme_hot_prob"] = 1.0 - stats.norm.cdf(
        anom_95, loc=predicted_anomaly, scale=forecast_df["anomaly_sigma"]
    )
    forecast_df["extreme_cold_prob"] = stats.norm.cdf(
        anom_5, loc=predicted_anomaly, scale=forecast_df["anomaly_sigma"]
    )
    
    # Direction classification
    forecast_df["anomaly_direction"] = "NORMAL"
    forecast_df.loc[predicted_anomaly > 1.5, "anomaly_direction"] = "HOT"
    forecast_df.loc[predicted_anomaly > 3.0, "anomaly_direction"] = "EXTREME_HOT"
    forecast_df.loc[predicted_anomaly < -1.5, "anomaly_direction"] = "COLD"
    forecast_df.loc[predicted_anomaly < -3.0, "anomaly_direction"] = "EXTREME_COLD"
    
    return forecast_df


def _fallback_anomaly_prediction(forecast_df):
    """Fallback when regressor unavailable."""
    forecast_df = forecast_df.copy()
    
    # Simple exponential smoothing on forecast temps
    if "forecast_temp" in forecast_df.columns:
        temp_mean = forecast_df["forecast_temp"].mean()
        forecast_df["predicted_anomaly"] = forecast_df["forecast_temp"] - temp_mean
    else:
        forecast_df["predicted_anomaly"] = 0
    
    forecast_df["anomaly_sigma"] = 1.5
    forecast_df["anomaly_confidence"] = 0.5
    forecast_df["extreme_hot_prob"] = forecast_df["predicted_anomaly"].clip(lower=0) / 5.0
    forecast_df["extreme_cold_prob"] = (-forecast_df["predicted_anomaly"]).clip(lower=0) / 5.0
    
    forecast_df["anomaly_direction"] = "NORMAL"
    forecast_df.loc[forecast_df["predicted_anomaly"] > 1.5, "anomaly_direction"] = "HOT"
    forecast_df.loc[forecast_df["predicted_anomaly"] > 3.0, "anomaly_direction"] = "EXTREME_HOT"
    forecast_df.loc[forecast_df["predicted_anomaly"] < -1.5, "anomaly_direction"] = "COLD"
    
    return forecast_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import load_data
    
    df = load_data("Bangalore")
    
    regressor, scaler, metrics = train_anomaly_regressor(
        df, "Temperature", "Bangalore", force_retrain=True
    )
    
    print("Metrics:", metrics)

