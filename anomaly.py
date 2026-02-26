"""
anomaly.py — Improved anomaly detection
=========================================
Combines two research-backed methods:
1. Z-score deviation from 30-year climatological normal (with 15-day rolling window)
2. Isolation Forest on multi-feature input (temp, rolling stats, rate of change)

Final anomaly = union of both methods for maximum recall.

References:
- IMD climatological normals (1991-2020 baseline)
- Isolation Forest + feature-based approach (NIH/PMC heatwave papers)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def compute_climatology(df, temp_col="Temperature", window=15):
    """
    Compute 30-year climatological normal for each day-of-year
    using a rolling window for smooth seasonal baseline.
    Returns Series indexed by day-of-year (1-366).
    """
    doy = df.index.dayofyear
    daily_mean = df.groupby(doy)[temp_col].mean()

    # Smooth with rolling window to avoid day-to-day noise
    # Wrap-around for Jan/Dec transition
    extended = pd.concat([daily_mean.iloc[-window:], daily_mean, daily_mean.iloc[:window]])
    smoothed = extended.rolling(window, center=True, min_periods=1).mean()
    smoothed = smoothed.iloc[window: window + len(daily_mean)]
    smoothed.index = daily_mean.index

    return smoothed


def compute_climatology_std(df, temp_col="Temperature", window=15):
    """
    Compute day-of-year standard deviation with rolling window smoothing.
    """
    doy = df.index.dayofyear
    daily_std = df.groupby(doy)[temp_col].std()

    extended = pd.concat([daily_std.iloc[-window:], daily_std, daily_std.iloc[:window]])
    smoothed = extended.rolling(window, center=True, min_periods=1).mean()
    smoothed = smoothed.iloc[window: window + len(daily_std)]
    smoothed.index = daily_std.index

    return smoothed


def detect_anomalies(df, temp_col="Temperature", zscore_threshold=2.0):
    """
    Detect temperature anomalies using two methods:
    1. Z-score: deviation from climatological normal > threshold σ
    2. Isolation Forest: statistical outlier on multi-feature input

    Adds columns: 'climatology', 'clim_std', 'anomaly_val', 'zscore',
                  'anomaly_zscore', 'anomaly_iforest', 'anomaly' (combined)
    """
    df = df.copy()

    # === 1. Climatological baseline ===
    clim_mean = compute_climatology(df, temp_col)
    clim_std = compute_climatology_std(df, temp_col)

    doy = df.index.dayofyear
    df["climatology"] = doy.map(clim_mean)
    df["clim_std"] = doy.map(clim_std)

    # Fill any missing (leap year day 366)
    df["climatology"] = df["climatology"].ffill().bfill()
    df["clim_std"] = df["clim_std"].ffill().bfill()
    df["clim_std"] = df["clim_std"].replace(0, df["clim_std"].mean())

    # Anomaly value = deviation from climatology
    df["anomaly_val"] = df[temp_col] - df["climatology"]

    # Z-score
    df["zscore"] = df["anomaly_val"] / df["clim_std"]

    # Z-score anomaly flag (both hot and cold extremes)
    df["anomaly_zscore"] = (df["zscore"].abs() > zscore_threshold).astype(int)

    # === 2. Isolation Forest on multi-feature input ===
    # Build features for Isolation Forest
    iforest_features = pd.DataFrame(index=df.index)
    iforest_features["temp"] = df[temp_col]
    iforest_features["anomaly_val"] = df["anomaly_val"]
    iforest_features["zscore"] = df["zscore"]

    # Rolling features for context MUST BE SHIFTED otherwise the model learns
    # the anomaly simply from today's rate of change compared to today!
    temp_shifted = df[temp_col].shift(1)
    iforest_features["roll_mean_7"] = temp_shifted.rolling(7, min_periods=1).mean()
    iforest_features["roll_std_7"] = temp_shifted.rolling(7, min_periods=1).std().fillna(0)
    iforest_features["diff_1"] = df[temp_col].diff(1).shift(1).fillna(0)

    # Drop any remaining NaN
    iforest_features = iforest_features.fillna(0)

    iso_model = IsolationForest(
        contamination=0.03,  # ~3% anomalies (aligned with research)
        n_estimators=200,
        max_features=0.8,
        random_state=42,
    )

    iso_preds = iso_model.fit_predict(iforest_features)
    df["anomaly_iforest"] = pd.Series(iso_preds, index=df.index).map({1: 0, -1: 1})

    # === 3. Combined anomaly (union of both methods) ===
    df["anomaly"] = ((df["anomaly_zscore"] == 1) | (df["anomaly_iforest"] == 1)).astype(int)

    return df
