"""
model_training.py — XGBoost-based temperature forecasting model
================================================================
Research-backed approach using:
- Feature engineering: lag features, rolling statistics, seasonal encoding
- XGBoost Regressor for next-day temperature prediction
- Walk-forward chronological train/test split (80/20)
- Model evaluation: MAE, RMSE, R²
- Per-city model caching in models/ directory

References:
- AdaBoost + MLP for Indian Tmax anomalies (heathealth.info)
- XGBoost + RF ensemble for heatwave classification (NIH/PMC)
- Hybrid ARIMA + Gradient Boosting for temperature anomaly (SSRN)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, recall_score, precision_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from statsmodels.tsa.statespace.sarimax import SARIMAX
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _safe_city_name(city_name):
    return city_name.replace(" ", "_").replace("(", "").replace(")", "").lower()


def _model_path(city_name):
    return os.path.join(MODELS_DIR, f"xgb_{_safe_city_name(city_name)}.pkl")

def _sarima_model_path(city_name):
    return os.path.join(MODELS_DIR, f"sarima_{_safe_city_name(city_name)}.pkl")

def _metrics_path(city_name):
    return os.path.join(MODELS_DIR, f"metrics_{_safe_city_name(city_name)}.json")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df, temp_col="Temperature", train_clim_mean=None):
    """
    Build ML features from temperature time series, strictly avoiding data leakage.
    - All rolling/lag features are shifted so day 't' features only use data up to 't-1'.
    - Climatology deviation uses ONLY the training data's climatological mean.
    """
    feat = pd.DataFrame(index=df.index)
    temp = df[temp_col]

    # --- Lag features ---
    for lag in [1, 2, 3, 7, 14, 30]:
        feat[f"lag_{lag}"] = temp.shift(lag)

    # --- Shifted Rolling statistics ---
    # Shift by 1 first to prevent target leakage
    temp_shifted = temp.shift(1)
    for window in [7, 14, 30]:
        roll = temp_shifted.rolling(window, min_periods=1)
        feat[f"roll_mean_{window}"] = roll.mean()
        feat[f"roll_std_{window}"] = roll.std()
        feat[f"roll_min_{window}"] = roll.min()
        feat[f"roll_max_{window}"] = roll.max()
        
    # --- Rolling Quantile Features ---
    roll_30 = temp_shifted.rolling(30, min_periods=1)
    feat["roll_q90_30"] = roll_30.quantile(0.9)
    feat["roll_q10_30"] = roll_30.quantile(0.1)

    # --- Heatwave Persistence Features ---
    # Hot day definition (e.g., > 35C) based only on historic info
    is_hot = (temp_shifted > 35.0).astype(int)
    # Consecutive hot days (simple cumulative sum reset by 0)
    blocks = (is_hot == 0).cumsum()
    feat["consecutive_hot_days"] = is_hot.groupby(blocks).cumsum()
    feat["3_day_hot_streak"] = (feat["consecutive_hot_days"] >= 3).astype(int)

    # --- Rate of change (Shifted) ---
    feat["diff_1"] = temp.diff(1).shift(1)
    feat["diff_3"] = temp.diff(3).shift(1)

    # --- Seasonal encoding (sin/cos) ---
    doy = df.index.dayofyear
    feat["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    feat["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    feat["month"] = df.index.month

    # --- Climatological deviation ---
    # Must use ONLY the provided train_clim_mean to avoid leakage
    doy_series = df.index.dayofyear
    if train_clim_mean is not None:
        clim = doy_series.map(train_clim_mean)
        feat["clim_mean"] = clim
        # We also must use the shifted temp to compute deviation as a feature, 
        # otherwise we leak the target temperature into the feature set!
        feat["deviation_from_clim"] = temp_shifted - clim

    # --- Year trend ---
    feat["year"] = df.index.year
    feat["year_norm"] = (feat["year"] - feat["year"].min()) / max((feat["year"].max() - feat["year"].min()), 1)

    # Target
    feat["target"] = temp

    return feat


# ============================================================
# MODEL TRAINING
# ============================================================

def _compute_climatology_for_train(train_series, window=15):
    """Compute 15-day rolling window climatology strictly on training data."""
    doy = train_series.index.dayofyear
    daily_mean = train_series.groupby(doy).mean()

    # Smooth with rolling window
    extended = pd.concat([daily_mean.iloc[-window:], daily_mean, daily_mean.iloc[:window]])
    smoothed = extended.rolling(window, center=True, min_periods=1).mean()
    smoothed = smoothed.iloc[window: window + len(daily_mean)]
    smoothed.index = daily_mean.index
    return smoothed


def train_xgb_model(df, temp_col="Temperature", city_name="default", force_retrain=False):
    """
    Train XGBoost model using strict chronological split & Hyperparameter Tuning.
    """
    model_p = _model_path(city_name)
    metrics_p = _metrics_path(city_name)

    if not force_retrain and os.path.exists(model_p) and os.path.exists(metrics_p):
        model = joblib.load(model_p)
        with open(metrics_p, "r") as f:
            metrics = json.load(f)
        return model, metrics
    
    # SAFETY: On a live server, we should NOT train on the fly as it will time out.
    # Instead, we fail fast so the user knows the models are missing.
    print(f"❌ ERROR: XGBoost model not found at {model_p}. Training is disabled on live server.")
    print("Please run 'download_models.py' or train models locally before deploying.")
    return None, {"error": f"XGBoost model for {city_name} not found."}
    # 1. Strict chronological split FIRST (Train = first 90%, Test = last 10%)
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # 2. Compute train Climatology
    train_clim_mean = _compute_climatology_for_train(train_df[temp_col])

    # 3. Build features using strictly training climatology
    feat = build_features(df, temp_col, train_clim_mean=train_clim_mean)
    
    # Re-apply the initial date index division on the dataframe
    train_feat = feat.iloc[:split_idx].dropna()
    test_feat = feat.iloc[split_idx:].dropna()

    feature_cols = [c for c in feat.columns if c != "target"]
    X_train, y_train = train_feat[feature_cols], train_feat["target"]
    X_test, y_test = test_feat[feature_cols], test_feat["target"]

    try:
        # TimeSeriesSplit for CV
        tscv = TimeSeriesSplit(n_splits=3)
        
        base_model = XGBRegressor(random_state=42, n_jobs=1)
        
        # Simplified param_grid for free-tier server safety
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        
        search = RandomizedSearchCV(
            base_model, param_distributions=param_grid, n_iter=2, # Reduced to 2 for maximum speed
            scoring='neg_mean_absolute_error', cv=tscv, random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        
    except TypeError:
        # Fallback for simple testing
        model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Evaluate Baseline (Naive Persistence = predict yesterday's temperature)
    baseline_pred = test_feat["lag_1"]
    mae_baseline = mean_absolute_error(y_test, baseline_pred)

    metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "mae_baseline": round(mae_baseline, 3),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": len(feature_cols),
        "city": city_name,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, model_p)
    with open(metrics_p, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{city_name}] XGBoost trained — Model MAE: {mae:.2f}°C vs Baseline MAE: {mae_baseline:.2f}°C")
    return model, metrics


def train_sarima_model(df, temp_col="Temperature", city_name="default", force_retrain=False):
    """
    Train SARIMA model for temperature prediction to be used in ensemble.
    Returns the trained model.
    """
    model_p = _sarima_model_path(city_name)
    
    if not force_retrain and os.path.exists(model_p):
        return joblib.load(model_p)
        
    # SAFETY: On the free server, training SARIMA from scratch causes a timeout/crash.
    # If the model is missing (download failed), we return None. 
    # The system will fall back to using only XGBoost, which is safer.
    print(f"⚠️ WARNING: SARIMA Model not found at {model_p}. Skipping training to prevent server timeout.")
    return None



def get_feature_columns(df, temp_col="Temperature"):
    """Return the list of feature column names (for prediction)."""
    feat = build_features(df, temp_col)
    return [c for c in feat.columns if c != "target"]


def load_model_and_metrics(city_name):
    """Load a previously trained model and its metrics. Returns (xgb_model, sarima_model, metrics)"""
    model_p = _model_path(city_name)
    sarima_p = _sarima_model_path(city_name)
    metrics_p = _metrics_path(city_name)

    xgb_model = None
    if os.path.exists(model_p):
        xgb_model = joblib.load(model_p)

    sarima_model = None
    if os.path.exists(sarima_p):
        sarima_model = joblib.load(sarima_p)

    metrics = {}
    if os.path.exists(metrics_p):
        with open(metrics_p, "r") as f:
            metrics = json.load(f)

    return xgb_model, sarima_model, metrics
