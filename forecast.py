"""
forecast.py — XGBoost-based recursive multi-step forecasting
=============================================================
Uses trained XGBoost model from model_training.py for recursive
day-by-day forecasting with confidence intervals.

Applies IMD heatwave criteria per station type:
- Plains: ≥40°C actual + ≥4.5°C departure (severe: >6.4°C) for ≥3 days
- Coastal: ≥37°C actual + ≥4.5°C departure for ≥3 days

Falls back to SARIMA if XGBoost model is not available.

References:
- IMD heatwave criteria (imd.gov.in)
- XGBoost recursive forecasting (IEEE, WorldScientific)
"""

import numpy as np
import pandas as pd
import joblib
import os
from model_training import build_features, _model_path, _sarima_model_path, MODELS_DIR
from anomaly import compute_climatology, compute_climatology_std

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
    Forecast future temperatures using XGBoost recursive prediction
    and SARIMA ensemble, depending on which models are provided.
    Returns (forecast_df, anomaly_threshold, imd_thresholds).
    """
    # If models weren't explicitly passed, load them from disk
    if xgb_model is None:
        xgb_path = _model_path(city_name)
        if os.path.exists(xgb_path):
            xgb_model = joblib.load(xgb_path)
            
    if model is None: # This points to the SARIMA model 
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

    # Process forecasts
    xgb_forecast = None
    if xgb_model is not None:
        xgb_forecast = _xgb_recursive_forecast(
            df, xgb_model, temp_col, days, clim_mean, clim_std
        )
    
    sarima_forecast = None
    if model is not None:
        sarima_forecast = _sarima_forecast(df, model, temp_col, days, clim_mean)
        
    if xgb_forecast is not None and sarima_forecast is not None:
        # Ensemble: Average the two DataFrames
        forecast_df = pd.DataFrame(index=xgb_forecast.index)
        forecast_df["forecast_temp"] = (xgb_forecast["forecast_temp"] + sarima_forecast["forecast_temp"]) / 2
        # Use simple max deviation from the average for CI
        upper = np.maximum(xgb_forecast["forecast_upper"], sarima_forecast["forecast_upper"])
        lower = np.minimum(xgb_forecast["forecast_lower"], sarima_forecast["forecast_lower"])
        forecast_df["forecast_upper"] = upper
        forecast_df["forecast_lower"] = lower
        forecast_df["doy"] = forecast_df.index.dayofyear
        forecast_df["climatology"] = forecast_df["doy"].map(clim_mean).ffill().bfill()
        forecast_df["anomaly"] = forecast_df["forecast_temp"] - forecast_df["climatology"]
    elif xgb_forecast is not None:
        # Fallback to pure XGBoost
        forecast_df = xgb_forecast
    elif sarima_forecast is not None:
        # Fallback to pure SARIMA
        forecast_df = sarima_forecast
    else:
        raise ValueError("No trained models available for prediction.")

    # --- Classify heatwave events using IMD criteria ---
    forecast_df = _apply_imd_heatwave_criteria(
        forecast_df, anomaly_threshold, imd_thresh
    )

    return forecast_df, anomaly_threshold, imd_thresh


def _xgb_recursive_forecast(df, model, temp_col, days, clim_mean, clim_std):
    """
    Recursive multi-step forecast: predict day t+1, append, repeat.
    Also computes quantile-based confidence intervals.
    """
    # Start with the tail of historical data (need enough for lag features)
    buffer_days = 60
    history = df[[temp_col]].iloc[-buffer_days:].copy()

    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days)
    predictions = []
    pred_upper = []
    pred_lower = []

    # Compute historical prediction residual std for confidence intervals
    # Must use train_clim_mean to match the required feature length structure
    hist_feat = build_features(df, temp_col, train_clim_mean=clim_mean).dropna()
    feature_cols = [c for c in hist_feat.columns if c != "target"]
    if len(hist_feat) > 100:
        hist_pred = model.predict(hist_feat[feature_cols])
        residuals = hist_feat["target"].values - hist_pred
        residual_std = float(np.std(residuals))
    else:
        residual_std = 1.5  # fallback

    for i, date in enumerate(future_dates):
        # We need a proper trailing history to calculate the 30-day quantiles 
        # and 3-day streaks of the recursive predictions. 
        # So we must build features on the entire 'history' dataframe, 
        # then take the LAST row (which represents the features exactly before 'date')
        feat = build_features(history, temp_col, train_clim_mean=clim_mean)
        
        # We only dropna for the specific target date row 
        # because the 30-day rolling might cause NaNs early in the buffer list
        last_row = feat.iloc[-1:].copy()
        feature_cols = [c for c in feat.columns if c != "target"]
        
        if last_row[feature_cols].isna().any(axis=1).iloc[0]:
            # Edge case: fallback or fillna if buffer is still too small
            last_row[feature_cols] = last_row[feature_cols].fillna(method="ffill").fillna(0)
        
        # Update seasonal features for the target date
            target_doy = date.dayofyear
            last_row["sin_doy"] = np.sin(2 * np.pi * target_doy / 365.25)
            last_row["cos_doy"] = np.cos(2 * np.pi * target_doy / 365.25)
            last_row["month"] = date.month
            if "year" in last_row.columns:
                last_row["year"] = date.year
            if "year_norm" in last_row.columns:
                year_min = df.index.year.min()
                year_max = max(df.index.year.max(), date.year)
                last_row["year_norm"] = (date.year - year_min) / max((year_max - year_min), 1)

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
        ci_factor = 1.0 + 0.3 * np.sqrt(i + 1) / np.sqrt(days)
        pred_upper.append(pred + 1.96 * residual_std * ci_factor)
        pred_lower.append(pred - 1.96 * residual_std * ci_factor)

        # Append prediction to history for next step
        new_row = pd.DataFrame(
            {temp_col: [pred]},
            index=[date],
        )
        history = pd.concat([history, new_row])

    # Build forecast DataFrame
    forecast_df = pd.DataFrame(
        {
            "forecast_temp": predictions,
            "forecast_upper": pred_upper,
            "forecast_lower": pred_lower,
        },
        index=future_dates,
    )

    # Add climatology
    forecast_df["doy"] = forecast_df.index.dayofyear
    forecast_df["climatology"] = forecast_df["doy"].map(clim_mean).ffill().bfill()
    forecast_df["anomaly"] = forecast_df["forecast_temp"] - forecast_df["climatology"]

    return forecast_df


def _sarima_forecast(df, model, temp_col, days, clim_mean):
    """SARIMA forecast."""
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days)
    
    # We must properly forecast steps using the fit statespace model.
    # It will start the forecast from the end of the fitted training series.
    forecast = model.get_forecast(steps=days)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05) # 95% Confidence Interval

    forecast_df = pd.DataFrame({
        "forecast_temp": forecast_mean.values,
        "forecast_upper": conf_int.iloc[:, 1].values,  # upper bound
        "forecast_lower": conf_int.iloc[:, 0].values,  # lower bound
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
    - Also flags percentile-based (anomaly > 95th percentile threshold)
    """
    min_actual = imd_thresh["min_actual"]
    hw_dep = imd_thresh["heatwave_departure"]
    severe_dep = imd_thresh["severe_departure"]

    # IMD criteria
    forecast_df["imd_heatwave_cond"] = (
        (forecast_df["forecast_temp"] >= min_actual) &
        (forecast_df["anomaly"] >= hw_dep)
    )
    forecast_df["imd_severe_cond"] = (
        (forecast_df["forecast_temp"] >= min_actual) &
        (forecast_df["anomaly"] > severe_dep)
    )

    # Percentile-based criteria (backup for cities that rarely hit 40°C)
    forecast_df["pct_heatwave_cond"] = forecast_df["anomaly"] > anomaly_threshold

    # Combined: either IMD or percentile-based
    forecast_df["heatwave_condition"] = (
        forecast_df["imd_heatwave_cond"] | forecast_df["pct_heatwave_cond"]
    )

    # Require ≥3 consecutive days
    s = forecast_df["heatwave_condition"]
    groups = (s != s.shift()).cumsum()
    forecast_df["heatwave_event"] = s & (s.groupby(groups).transform("size") >= 3)

    # Severity classification
    forecast_df["severity"] = "NORMAL"
    forecast_df.loc[forecast_df["heatwave_event"], "severity"] = "HEATWAVE"
    forecast_df.loc[
        forecast_df["heatwave_event"] & forecast_df["imd_severe_cond"],
        "severity"
    ] = "SEVERE"

    return forecast_df
