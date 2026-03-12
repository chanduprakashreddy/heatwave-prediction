"""
forecast.py — Hybrid temperature forecasting system
=====================================================
Short-term (≤90 days): XGBoost recursive + SARIMA ensemble (ML precision)
Long-term (>90 days): Climatology + historical variability + warming trend
                      (how real climate/meteorological agencies forecast)

Applies IMD heatwave criteria per station type:
- Plains: ≥40°C actual + ≥4.5°C departure (severe: >6.4°C) for ≥3 days
- Coastal: ≥37°C actual + ≥4.5°C departure for ≥3 days

References:
- IMD heatwave criteria (imd.gov.in)
- XGBoost recursive forecasting (IEEE, WorldScientific)
- IPCC AR6 WG1 Annex I — Regional warming trends for South Asia
- IMD Technical Report: "Observed Trends in Maximum Temperature Over India"
- WMO Guidelines on Seasonal Climate Prediction (climatological approach)
"""

import numpy as np
import pandas as pd
import joblib
import os
from model_training import build_features, _model_path, _sarima_model_path, MODELS_DIR
from anomaly import compute_climatology, compute_climatology_std

# ============================================================
# CLIMATE WARMING CONSTANTS
# ============================================================
# IPCC AR6 + IMD observed Tmax trend for Indian subcontinent: ~0.025°C/year
WARMING_RATE_PER_YEAR = 0.025  # °C/year (conservative estimate)

# Short-term forecast limit — beyond this, switch to climatological approach
SHORT_TERM_DAYS = 90

# IMD classification by station type
STATION_TYPE = {
    "Bangalore": "plains",
    "Chennai": "coastal",
    "Delhi NCR": "plains",
    "Lucknow": "plains",
    "Mumbai": "coastal",
    "Rajasthan (Jodhpur)": "plains",
    "Bhubaneswar": "plains",
    "Rourkela": "plains",
}

IMD_THRESHOLDS = {
    "plains": {"min_actual": 40.0, "heatwave_departure": 4.5, "severe_departure": 6.4},
    "coastal": {"min_actual": 37.0, "heatwave_departure": 4.5, "severe_departure": 6.4},
}


def predict_future(df, temp_col="Temperature", days=365, city_name="default",
                   xgb_model=None, model=None, percentile=0.95):
    """
    Forecast future temperatures using a HYBRID approach:
    
    1. Short-term (≤90 days): XGBoost + SARIMA ensemble with time-decay weights
    2. Long-term (>90 days): Climatology + historical anomaly sampling + warming trend
       This mimics how meteorological agencies make seasonal/decadal forecasts.
    
    The transition between short-term and long-term is smooth (blended).
    """
    # If models weren't explicitly passed, load them from disk
    if xgb_model is None:
        xgb_path = _model_path(city_name)
        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
            
    if model is None:
        sarima_path = _sarima_model_path(city_name)
        if os.path.exists(sarima_path):
            model = joblib.load(sarima_path)

    # Climatology from historical data
    clim_mean = compute_climatology(df, temp_col)
    clim_std = compute_climatology_std(df, temp_col)

    # Historical anomaly threshold (percentile-based)
    doy_hist = df.index.dayofyear
    hist_clim = doy_hist.map(clim_mean)
    hist_anomaly = df[temp_col] - hist_clim
    anomaly_threshold = float(hist_anomaly.quantile(percentile))

    # Get IMD thresholds for this station
    station_type = STATION_TYPE.get(city_name, "plains")
    imd_thresh = IMD_THRESHOLDS[station_type]

    # Max historical year
    max_hist_year = df.index.year.max()

    # === GENERATE FORECASTS ===
    # Step 1: ML-based short-term forecast (up to SHORT_TERM_DAYS)
    ml_days = min(days, SHORT_TERM_DAYS)
    
    xgb_forecast = None
    if xgb_model is not None:
        xgb_forecast = _xgb_recursive_forecast(
            df, xgb_model, temp_col, ml_days, clim_mean, clim_std
        )
    
    sarima_forecast = None
    if model is not None:
        sarima_forecast = _sarima_forecast(df, model, temp_col, ml_days, clim_mean)
    
    # Ensemble short-term ML forecasts
    if xgb_forecast is not None and sarima_forecast is not None:
        ml_forecast = _ensemble_forecasts(xgb_forecast, sarima_forecast, clim_mean)
    elif xgb_forecast is not None:
        ml_forecast = xgb_forecast
    elif sarima_forecast is not None:
        ml_forecast = sarima_forecast
    else:
        ml_forecast = None
    
    # Step 2: Climatological long-term forecast (for all days)
    clim_forecast = _climatological_forecast(df, temp_col, days, clim_mean, clim_std)
    
    # Step 3: Blend ML and climatological forecasts
    if ml_forecast is not None and days > SHORT_TERM_DAYS:
        # Hybrid: ML for first N days, climatology for rest, with smooth blend
        forecast_df = _blend_forecasts(ml_forecast, clim_forecast, SHORT_TERM_DAYS)
    elif ml_forecast is not None:
        # Pure ML for short forecasts
        forecast_df = ml_forecast
    else:
        # Pure climatology fallback
        forecast_df = clim_forecast

    # === APPLY CLIMATE WARMING TREND ===
    years_ahead = np.maximum(forecast_df.index.year - max_hist_year, 0)
    warming_offset = years_ahead * WARMING_RATE_PER_YEAR
    forecast_df["forecast_temp"] = forecast_df["forecast_temp"] + warming_offset
    forecast_df["forecast_upper"] = forecast_df["forecast_upper"] + warming_offset
    forecast_df["forecast_lower"] = forecast_df["forecast_lower"] + warming_offset
    # Recompute anomaly after warming adjustment
    forecast_df["anomaly"] = forecast_df["forecast_temp"] - forecast_df["climatology"]

    # --- Classify heatwave events using IMD criteria ---
    forecast_df = _apply_imd_heatwave_criteria(
        forecast_df, anomaly_threshold, imd_thresh
    )

    return forecast_df, anomaly_threshold, imd_thresh


def _ensemble_forecasts(xgb_forecast, sarima_forecast, clim_mean):
    """Time-decay weighted ensemble of XGBoost and SARIMA."""
    forecast_df = pd.DataFrame(index=xgb_forecast.index)
    n_days = len(xgb_forecast)
    
    # SARIMA weight decays from 0.5 → 0.1 over the forecast period
    sarima_weights = np.clip(0.5 - 0.4 * np.arange(n_days) / max(n_days, 1), 0.1, 0.5)
    xgb_weights = 1.0 - sarima_weights
    
    forecast_df["forecast_temp"] = (
        xgb_weights * xgb_forecast["forecast_temp"].values + 
        sarima_weights * sarima_forecast["forecast_temp"].values
    )
    upper = np.maximum(xgb_forecast["forecast_upper"].values, sarima_forecast["forecast_upper"].values)
    lower = np.minimum(xgb_forecast["forecast_lower"].values, sarima_forecast["forecast_lower"].values)
    forecast_df["forecast_upper"] = upper
    forecast_df["forecast_lower"] = lower
    forecast_df["doy"] = forecast_df.index.dayofyear
    forecast_df["climatology"] = forecast_df["doy"].map(clim_mean).ffill().bfill()
    forecast_df["anomaly"] = forecast_df["forecast_temp"] - forecast_df["climatology"]
    
    return forecast_df


def _climatological_forecast(df, temp_col, days, clim_mean, clim_std):
    """
    Generate a climatology-based forecast with realistic historical variability.
    
    This is how meteorological agencies make seasonal/decadal forecasts:
    - Base temperature = smoothed climatological mean for each day-of-year
    - Variability = sampled from real historical anomaly distribution per day-of-year
    - Trend = warming offset applied later in predict_future()
    
    The variability is sampled using a consistent random seed indexed by day-of-year,
    so the same query returns the same results (reproducibility).
    """
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days)
    
    # Compute historical anomaly distribution per day-of-year
    doy_hist = df.index.dayofyear
    hist_clim = doy_hist.map(clim_mean)
    hist_anomaly = df[temp_col] - hist_clim
    
    # Group anomalies by day-of-year
    doy_anomalies = {}
    for d in range(1, 367):
        mask = (doy_hist == d)
        if mask.sum() > 0:
            doy_anomalies[d] = hist_anomaly[mask].values
    
    # Sample variability from historical anomaly distribution
    np.random.seed(42)  # Reproducible results
    forecast_anomalies = []
    for date in future_dates:
        doy = date.dayofyear
        if doy in doy_anomalies and len(doy_anomalies[doy]) > 0:
            # Sample from historical anomaly distribution for this day
            anom = np.random.choice(doy_anomalies[doy])
        else:
            # Fallback: use nearest available day
            nearest_doy = min(doy_anomalies.keys(), key=lambda x: abs(x - doy))
            anom = np.random.choice(doy_anomalies[nearest_doy])
        forecast_anomalies.append(float(anom))
    
    forecast_anomalies = np.array(forecast_anomalies)
    
    # Build forecast: climatology + sampled anomaly
    forecast_temps = []
    climatologies = []
    for i, date in enumerate(future_dates):
        doy = date.dayofyear
        if doy in clim_mean.index:
            clim_val = clim_mean[doy]
        else:
            clim_val = clim_mean.mean()
        climatologies.append(float(clim_val))
        forecast_temps.append(float(clim_val + forecast_anomalies[i]))
    
    forecast_temps = np.array(forecast_temps)
    climatologies = np.array(climatologies)
    
    # Confidence intervals based on historical variability std
    ci_vals = []
    for date in future_dates:
        doy = date.dayofyear
        if doy in clim_std.index:
            ci_vals.append(float(clim_std[doy]))
        else:
            ci_vals.append(float(clim_std.mean()))
    ci_vals = np.array(ci_vals)
    
    forecast_df = pd.DataFrame({
        "forecast_temp": forecast_temps,
        "forecast_upper": forecast_temps + 1.96 * ci_vals,
        "forecast_lower": forecast_temps - 1.96 * ci_vals,
        "doy": future_dates.dayofyear,
        "climatology": climatologies,
        "anomaly": forecast_anomalies,
    }, index=future_dates)
    
    return forecast_df


def _blend_forecasts(ml_forecast, clim_forecast, blend_days):
    """
    Smoothly blend ML forecast (short-term) with climatological forecast (long-term).
    
    Uses a sigmoid transition over a 30-day window centered at blend_days.
    ML weight: 1.0 at day 0, 0.5 at blend_days, 0.0 at blend_days+30
    """
    # Full forecast uses climatological dates
    forecast_df = clim_forecast.copy()
    
    n_total = len(forecast_df)
    n_ml = len(ml_forecast)
    
    # Create blending weights: sigmoid transition
    transition_width = 30  # days over which to blend
    day_indices = np.arange(n_total)
    
    # Sigmoid: 1.0 at start, transitions to 0.0 around blend_days
    ml_weight = 1.0 / (1.0 + np.exp((day_indices - blend_days) / (transition_width / 4)))
    clim_weight = 1.0 - ml_weight
    
    # Apply blending for the ML forecast period
    for i in range(min(n_ml, n_total)):
        w_ml = ml_weight[i]
        w_clim = clim_weight[i]
        
        forecast_df.iloc[i, forecast_df.columns.get_loc("forecast_temp")] = (
            w_ml * ml_forecast["forecast_temp"].iloc[i] +
            w_clim * clim_forecast["forecast_temp"].iloc[i]
        )
        forecast_df.iloc[i, forecast_df.columns.get_loc("forecast_upper")] = (
            w_ml * ml_forecast["forecast_upper"].iloc[i] +
            w_clim * clim_forecast["forecast_upper"].iloc[i]
        )
        forecast_df.iloc[i, forecast_df.columns.get_loc("forecast_lower")] = (
            w_ml * ml_forecast["forecast_lower"].iloc[i] +
            w_clim * clim_forecast["forecast_lower"].iloc[i]
        )
    
    # Recompute anomaly
    forecast_df["anomaly"] = forecast_df["forecast_temp"] - forecast_df["climatology"]
    
    return forecast_df


def _xgb_recursive_forecast(df, model, temp_col, days, clim_mean, clim_std):
    """
    Recursive multi-step forecast: predict day t+1, append, repeat.
    Limited to SHORT_TERM_DAYS to avoid mean-reversion degradation.
    """
    buffer_days = 60
    history = df[[temp_col]].iloc[-buffer_days:].copy()

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days)
    predictions = []
    pred_upper = []
    pred_lower = []

    # Compute historical residual std for CI
    hist_feat = build_features(df, temp_col, train_clim_mean=clim_mean).dropna()
    feature_cols = [c for c in hist_feat.columns if c != "target"]
    if len(hist_feat) > 100:
        hist_pred = model.predict(hist_feat[feature_cols])
        residuals = hist_feat["target"].values - hist_pred
        residual_std = float(np.std(residuals))
    else:
        residual_std = 1.5

    year_min = df.index.year.min()
    year_max_hist = df.index.year.max()

    for i, date in enumerate(future_dates):
        feat = build_features(history, temp_col, train_clim_mean=clim_mean)
        last_row = feat.iloc[-1:].copy()
        feature_cols = [c for c in feat.columns if c != "target"]
        
        if last_row[feature_cols].isna().any(axis=1).iloc[0]:
            last_row[feature_cols] = last_row[feature_cols].fillna(method="ffill").fillna(0)
        
        # ALWAYS update seasonal/year features for the target date
        target_doy = date.dayofyear
        last_row["sin_doy"] = np.sin(2 * np.pi * target_doy / 365.25)
        last_row["cos_doy"] = np.cos(2 * np.pi * target_doy / 365.25)
        last_row["month"] = date.month
        if "year" in last_row.columns:
            last_row["year"] = date.year
        if "year_norm" in last_row.columns:
            year_max = max(year_max_hist, date.year)
            last_row["year_norm"] = (date.year - year_min) / max((year_max - year_min), 1)
        if "years_since_2000" in last_row.columns:
            last_row["years_since_2000"] = date.year - 2000

        # Update climatology for target day
        if target_doy in clim_mean.index:
            clim_val = clim_mean[target_doy]
        else:
            clim_val = clim_mean.mean()
        if "clim_mean" in last_row.columns:
            last_row["clim_mean"] = clim_val
        if "deviation_from_clim" in last_row.columns:
            last_row["deviation_from_clim"] = float(history[temp_col].iloc[-1]) - clim_val

        pred = float(model.predict(last_row[feature_cols])[0])
        predictions.append(pred)

        # Confidence intervals (widening over time)
        days_ratio = (i + 1) / max(days, 1)
        ci_factor = 1.0 + 0.5 * days_ratio + 0.3 * (days_ratio ** 2)
        pred_upper.append(pred + 1.96 * residual_std * ci_factor)
        pred_lower.append(pred - 1.96 * residual_std * ci_factor)

        # Append prediction to history
        new_row = pd.DataFrame({temp_col: [pred]}, index=[date])
        history = pd.concat([history, new_row])
        if len(history) > 100:
            history = history.iloc[-100:]

    forecast_df = pd.DataFrame({
        "forecast_temp": predictions,
        "forecast_upper": pred_upper,
        "forecast_lower": pred_lower,
    }, index=future_dates)

    forecast_df["doy"] = forecast_df.index.dayofyear
    forecast_df["climatology"] = forecast_df["doy"].map(clim_mean).ffill().bfill()
    forecast_df["anomaly"] = forecast_df["forecast_temp"] - forecast_df["climatology"]

    return forecast_df


def _sarima_forecast(df, model, temp_col, days, clim_mean):
    """SARIMA forecast."""
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days)
    
    forecast = model.get_forecast(steps=days)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)

    forecast_df = pd.DataFrame({
        "forecast_temp": forecast_mean.values,
        "forecast_upper": conf_int.iloc[:, 1].values,
        "forecast_lower": conf_int.iloc[:, 0].values,
    }, index=future_dates)
    forecast_df["doy"] = forecast_df.index.dayofyear
    forecast_df["climatology"] = forecast_df["doy"].map(clim_mean).ffill().bfill()
    forecast_df["anomaly"] = forecast_df["forecast_temp"] - forecast_df["climatology"]

    return forecast_df


def _apply_imd_heatwave_criteria(forecast_df, anomaly_threshold, imd_thresh):
    """
    Apply IMD heatwave classification:
    - HEATWAVE: actual ≥ min_threshold AND departure ≥ heatwave_departure, for ≥3 days
    - SEVERE: actual ≥ min_threshold AND departure > severe_departure, for ≥3 days
    """
    min_actual = imd_thresh["min_actual"]
    hw_dep = imd_thresh["heatwave_departure"]
    severe_dep = imd_thresh["severe_departure"]

    forecast_df["imd_heatwave_cond"] = (
        (forecast_df["forecast_temp"] >= min_actual) &
        (forecast_df["anomaly"] >= hw_dep)
    )
    forecast_df["imd_severe_cond"] = (
        (forecast_df["forecast_temp"] >= min_actual) &
        (forecast_df["anomaly"] > severe_dep)
    )
    forecast_df["pct_heatwave_cond"] = forecast_df["anomaly"] > anomaly_threshold

    forecast_df["heatwave_condition"] = (
        forecast_df["imd_heatwave_cond"] | forecast_df["pct_heatwave_cond"]
    )

    s = forecast_df["heatwave_condition"]
    groups = (s != s.shift()).cumsum()
    forecast_df["heatwave_event"] = s & (s.groupby(groups).transform("size") >= 3)

    forecast_df["severity"] = "NORMAL"
    forecast_df.loc[forecast_df["heatwave_event"], "severity"] = "HEATWAVE"
    forecast_df.loc[
        forecast_df["heatwave_event"] & forecast_df["imd_severe_cond"],
        "severity"
    ] = "SEVERE"

    return forecast_df
