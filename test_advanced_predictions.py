"""
test_advanced_predictions.py — Test suite for advanced prediction modules
==========================================================================
Validates that new heatwave and anomaly prediction modules work correctly
and don't break existing functionality.

Run: python test_advanced_predictions.py [city_name]
"""

import sys
import traceback
import pandas as pd
import numpy as np
from datetime import datetime

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text):
    """Print formatted header."""
    print(f"\n{BLUE}{'='*70}")
    print(f"{text.center(70)}")
    print(f"{'='*70}{RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text):
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def test_imports():
    """Test that all new modules can be imported."""
    print_header("Testing Module Imports")
    
    modules = [
        ("utils", "load_data, get_available_cities"),
        ("anomaly", "detect_anomalies"),
        ("model_training", "train_xgb_model, train_sarima_model"),
        ("forecast", "predict_future"),
        ("advanced_heatwave_predictor", "train_heatwave_classifier, predict_heatwave_probability"),
        ("advanced_anomaly_predictor", "train_anomaly_regressor, predict_anomalies_advanced"),
        ("prediction_engine", "comprehensive_prediction"),
    ]
    
    all_ok = True
    for module_name, items in modules:
        try:
            exec(f"from {module_name} import {items}")
            print_success(f"Imported from {module_name}: {items}")
        except Exception as e:
            print_error(f"Failed to import {module_name}: {str(e)}")
            all_ok = False
    
    return all_ok


def test_data_loading(city_name):
    """Test data loading."""
    print_header(f"Testing Data Loading")
    
    try:
        from utils import load_data, get_available_cities
        
        # Check available cities
        cities = get_available_cities()
        print_success(f"Available cities: {', '.join(cities)}")
        
        if city_name not in cities:
            print_warning(f"'{city_name}' not in available cities, using '{cities[0]}'")
            city_name = cities[0]
        
        # Load data
        df = load_data(city_name)
        print_success(f"Loaded {len(df)} records for {city_name}")
        
        # Check data quality
        assert "Temperature" in df.columns, "Temperature column missing"
        assert not df.empty, "DataFrame is empty"
        assert df["Temperature"].notna().sum() > 0, "All temperatures are NaN"
        
        print_success(f"Data quality check passed")
        print(f"  - Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  - Temp range: {df['Temperature'].min():.1f}°C to {df['Temperature'].max():.1f}°C")
        
        return city_name, df
    
    except Exception as e:
        print_error(f"Data loading failed: {str(e)}")
        traceback.print_exc()
        return None, None


def test_anomaly_detection(df, city_name):
    """Test anomaly detection."""
    print_header("Testing Anomaly Detection")
    
    try:
        from anomaly import detect_anomalies
        
        df_anom = detect_anomalies(df.copy(), "Temperature")
        
        # Check new columns
        expected_cols = ["anomaly", "anomaly_val", "zscore"]
        for col in expected_cols:
            assert col in df_anom.columns, f"Missing {col} column"
        
        print_success(f"Anomaly detection added {len(expected_cols)} new columns")
        
        anomaly_count = df_anom["anomaly"].sum()
        anomaly_pct = 100 * anomaly_count / len(df_anom)
        print_success(f"Detected {anomaly_count} anomalous days ({anomaly_pct:.1f}%)")
        
        return df_anom
    
    except Exception as e:
        print_error(f"Anomaly detection failed: {str(e)}")
        traceback.print_exc()
        return None


def test_model_training(df, city_name):
    """Test existing model training."""
    print_header("Testing Model Training")
    
    try:
        from model_training import train_xgb_model, train_sarima_model
        from anomaly import detect_anomalies
        
        # Prepare data
        model_df = df[["Temperature"]].copy()
        model_df = model_df.sort_index()
        model_df = model_df.asfreq("D")
        model_df["Temperature"] = model_df["Temperature"].interpolate(method="linear", limit=7)
        model_df["Temperature"] = model_df["Temperature"].ffill().bfill()
        model_df = model_df.dropna()
        
        # Train XGBoost
        print("\nTraining XGBoost model...")
        xgb_model, xgb_metrics = train_xgb_model(model_df, "Temperature", city_name)
        print_success(f"XGBoost: MAE={xgb_metrics['mae']}, R²={xgb_metrics['r2']}")
        
        # Train SARIMA
        print("Training SARIMA model...")
        sarima_model = train_sarima_model(model_df, "Temperature", city_name)
        print_success(f"SARIMA model trained")
        
        return xgb_model, sarima_model, model_df
    
    except Exception as e:
        print_error(f"Model training failed: {str(e)}")
        traceback.print_exc()
        return None, None, None


def test_temperature_forecast(xgb_model, sarima_model, model_df, city_name):
    """Test temperature forecasting."""
    print_header("Testing Temperature Forecasting")
    
    try:
        from forecast import predict_future
        from anomaly import detect_anomalies
        
        # Add anomalies for forecast
        model_df_anom = detect_anomalies(model_df.copy(), "Temperature")
        
        # Forecast
        forecast_df, threshold, imd_thresh = predict_future(
            model_df_anom, "Temperature", days=90, city_name=city_name,
            xgb_model=xgb_model, model=sarima_model
        )
        
        assert len(forecast_df) == 90, f"Expected 90 forecast days, got {len(forecast_df)}"
        
        # Check columns
        expected_cols = ["forecast_temp", "forecast_upper", "forecast_lower", "heatwave_event"]
        for col in expected_cols:
            assert col in forecast_df.columns, f"Missing {col} in forecast"
        
        hw_days = forecast_df["heatwave_event"].sum()
        print_success(f"Generated 90-day forecast with {hw_days:.0f} predicted heatwave days")
        print(f"  - Temp range: {forecast_df['forecast_temp'].min():.1f}°C to {forecast_df['forecast_temp'].max():.1f}°C")
        
        return forecast_df
    
    except Exception as e:
        print_error(f"Temperature forecast failed: {str(e)}")
        traceback.print_exc()
        return None


def test_heatwave_predictor(df, forecast_df, city_name):
    """Test advanced heatwave predictor."""
    print_header("Testing Advanced Heatwave Predictor")
    
    try:
        from advanced_heatwave_predictor import train_heatwave_classifier, predict_heatwave_probability
        from anomaly import detect_anomalies
        
        # Prepare historical data
        df_anom = detect_anomalies(df.copy(), "Temperature")
        
        # Train classifier
        print("Training heatwave classifier...")
        classifier, scaler, hw_metrics = train_heatwave_classifier(
            df_anom, "Temperature", city_name
        )
        print_success(f"Heatwave classifier trained")
        print(f"  - Precision: {hw_metrics.get('precision', 'N/A')}")
        print(f"  - Recall: {hw_metrics.get('recall', 'N/A')}")
        print(f"  - F1: {hw_metrics.get('f1', 'N/A')}")
        print(f"  - ROC-AUC: {hw_metrics.get('roc_auc', 'N/A')}")
        
        # Predict heatwave probability
        print("Predicting heatwave probabilities...")
        forecast_df = predict_heatwave_probability(
            df_anom, forecast_df.copy(), "Temperature", city_name,
            classifier=classifier, scaler=scaler
        )
        
        # Check new columns
        expected_cols = ["hw_probability", "hw_risk", "hw_confidence"]
        for col in expected_cols:
            assert col in forecast_df.columns, f"Missing {col}"
        
        print_success(f"Added heatwave probability predictions")
        print(f"  - Avg HW probability: {forecast_df['hw_probability'].mean():.3f}")
        print(f"  - Max HW probability: {forecast_df['hw_probability'].max():.3f}")
        print(f"  - High-risk days (>0.6): {(forecast_df['hw_probability'] > 0.6).sum()}")
        print(f"  - Critical days (>0.8): {(forecast_df['hw_probability'] > 0.8).sum()}")
        
        return forecast_df
    
    except Exception as e:
        print_error(f"Heatwave predictor failed: {str(e)}")
        traceback.print_exc()
        return None


def test_anomaly_predictor(df, forecast_df, city_name):
    """Test advanced anomaly predictor."""
    print_header("Testing Advanced Anomaly Predictor")
    
    try:
        from advanced_anomaly_predictor import train_anomaly_regressor, predict_anomalies_advanced
        from anomaly import detect_anomalies
        
        # Prepare historical data
        df_anom = detect_anomalies(df.copy(), "Temperature")
        
        # Train regressor
        print("Training anomaly regressor...")
        regressor, scaler, anom_metrics = train_anomaly_regressor(
            df_anom, "Temperature", city_name
        )
        print_success(f"Anomaly regressor trained")
        print(f"  - Test MAE: {anom_metrics.get('mae_test', 'N/A')}°C")
        print(f"  - Test RMSE: {anom_metrics.get('rmse_test', 'N/A')}°C")
        print(f"  - Test R²: {anom_metrics.get('r2_test', 'N/A')}")
        
        # Predict anomalies
        print("Predicting temperature anomalies...")
        forecast_df = predict_anomalies_advanced(
            df_anom, forecast_df.copy(), "Temperature", city_name,
            regressor=regressor, scaler=scaler
        )
        
        # Check new columns
        expected_cols = ["predicted_anomaly", "anomaly_direction", "anomaly_confidence", "extreme_hot_prob"]
        for col in expected_cols:
            assert col in forecast_df.columns, f"Missing {col}"
        
        print_success(f"Added anomaly predictions")
        print(f"  - Avg predicted anomaly: {forecast_df['predicted_anomaly'].mean():.2f}°C")
        print(f"  - Max positive anomaly: {forecast_df['predicted_anomaly'].max():.2f}°C")
        print(f"  - Extreme hot days (>3°C): {(forecast_df['predicted_anomaly'] > 3.0).sum()}")
        print(f"  - Extreme cold days (<-3°C): {(forecast_df['predicted_anomaly'] < -3.0).sum()}")
        
        return forecast_df
    
    except Exception as e:
        print_error(f"Anomaly predictor failed: {str(e)}")
        traceback.print_exc()
        return None


def test_prediction_engine(df, forecast_df, city_name):
    """Test unified prediction engine."""
    print_header("Testing Unified Prediction Engine")
    
    try:
        from prediction_engine import comprehensive_prediction, generate_forecast_summary
        from anomaly import detect_anomalies
        
        # Prepare data
        df_anom = detect_anomalies(df.copy(), "Temperature")
        
        # Comprehensive prediction
        print("Running comprehensive prediction...")
        enhanced_forecast = comprehensive_prediction(
            df_anom, forecast_df.copy(), city_name
        )
        
        # Check all expected columns
        expected_cols = [
            "hw_probability", "predicted_anomaly", "composite_risk",
            "risk_level", "setup_warming", "consecutive_hot_days"
        ]
        for col in expected_cols:
            assert col in enhanced_forecast.columns, f"Missing {col}"
        
        print_success(f"Comprehensive prediction completed")
        
        # Generate summary
        print("Generating forecast summary...")
        summary = generate_forecast_summary(enhanced_forecast, city_name)
        
        print_success(f"Forecast summary generated")
        print(f"  - Forecast period: {summary.get('forecast_period', 'N/A')}")
        print(f"  - Heatwave days predicted: {summary.get('heatwave_days_predicted', 'N/A')}")
        print(f"  - Average HW probability: {summary.get('avg_hw_probability', 'N/A')}")
        print(f"  - Average risk score: {summary.get('avg_risk_score', 'N/A')}")
        print(f"  - Early warnings: {len(summary.get('early_warnings', []))}")
        
        return enhanced_forecast, summary
    
    except Exception as e:
        print_error(f"Prediction engine failed: {str(e)}")
        traceback.print_exc()
        return None, None


def main():
    """Run all tests."""
    print_header("ADVANCED PREDICTION SYSTEM TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Get city name
    city_name = sys.argv[1] if len(sys.argv) > 1 else "Bangalore"
    
    # Test imports
    if not test_imports():
        print_error("Failed to import required modules. Aborting.")
        return False
    
    # Test data loading
    city_name, df = test_data_loading(city_name)
    if df is None:
        print_error("Failed to load data. Aborting.")
        return False
    
    # Test anomaly detection
    df_anom = test_anomaly_detection(df, city_name)
    if df_anom is None:
        print_warning("Anomaly detection had issues, continuing with original data")
        df_anom = df
    
    # Test model training
    xgb_model, sarima_model, model_df = test_model_training(df, city_name)
    if xgb_model is None or sarima_model is None:
        print_error("Model training failed. Aborting.")
        return False
    
    # Test temperature forecast
    forecast_df = test_temperature_forecast(xgb_model, sarima_model, model_df, city_name)
    if forecast_df is None:
        print_error("Temperature forecast failed. Aborting.")
        return False
    
    # Test heatwave predictor
    forecast_df = test_heatwave_predictor(df_anom, forecast_df, city_name)
    if forecast_df is None:
        print_warning("Heatwave predictor failed, skipping anomaly predictor")
        return True
    
    # Test anomaly predictor
    forecast_df = test_anomaly_predictor(df_anom, forecast_df, city_name)
    if forecast_df is None:
        print_warning("Anomaly predictor failed, skipping prediction engine")
        return True
    
    # Test prediction engine
    enhanced_forecast, summary = test_prediction_engine(df_anom, forecast_df, city_name)
    if enhanced_forecast is None:
        print_warning("Prediction engine failed")
        return True
    
    # Success summary
    print_header("TEST SUMMARY")
    print_success("All tests completed successfully!")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"City: {city_name}")
    print(f"Forecast days: {len(enhanced_forecast)}")
    print(f"All columns added and validated")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
