"""Quick verification: test that future year predictions show warming trend and heatwave detection."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_data
from anomaly import detect_anomalies
from model_training import train_xgb_model, train_sarima_model
from forecast import predict_future
from prediction_engine import comprehensive_prediction

# Test Delhi NCR — a plains station with strong heatwaves
city = "Delhi NCR"
df = load_data(city)

# Prepare model data
model_df = df[["Temperature"]].copy().sort_index().asfreq("D")
model_df["Temperature"] = model_df["Temperature"].interpolate(method="linear", limit=7).ffill().bfill()
model_df = model_df.dropna(subset=["Temperature"])
model_df = detect_anomalies(model_df, "Temperature")

# Load models
xgb_model, metrics = train_xgb_model(model_df, "Temperature", city)
sarima_model = train_sarima_model(model_df, "Temperature", city)

print(f"\n{'='*60}")
print(f"VERIFICATION: {city}")
print(f"Historical range: {model_df.index.min().date()} to {model_df.index.max().date()}")
print(f"{'='*60}")

# Test 1: Short forecast (90 days)
forecast_90, thresh, imd = predict_future(
    model_df, "Temperature", days=90, city_name=city,
    model=sarima_model, xgb_model=xgb_model
)
print(f"\n  90-day forecast:")
print(f"    Mean temp: {forecast_90['forecast_temp'].mean():.1f}°C")
print(f"    Max temp:  {forecast_90['forecast_temp'].max():.1f}°C")
print(f"    HW days:   {forecast_90['heatwave_event'].sum()}")

# Test 2: Multi-year forecast (2025-2030)
import pandas as pd
max_date = model_df.index.max()
target_date = pd.Timestamp(year=2030, month=12, day=31)
forecast_days = min((target_date - max_date).days, 3650)

forecast_long, thresh, imd = predict_future(
    model_df, "Temperature", days=forecast_days, city_name=city,
    model=sarima_model, xgb_model=xgb_model
)

# Check warming trend: group by year and show mean temps
yearly_mean = forecast_long.groupby(forecast_long.index.year)["forecast_temp"].mean()
yearly_max = forecast_long.groupby(forecast_long.index.year)["forecast_temp"].max()
yearly_hw = forecast_long.groupby(forecast_long.index.year)["heatwave_event"].sum()

print(f"\n  Long-term forecast (to 2030):")
print(f"    {'Year':>6}  {'Mean':>8}  {'Max':>8}  {'HW Days':>8}")
print(f"    {'----':>6}  {'----':>8}  {'----':>8}  {'-------':>8}")
for year in sorted(yearly_mean.index):
    print(f"    {year:>6}  {yearly_mean[year]:>8.1f}  {yearly_max[year]:>8.1f}  {int(yearly_hw[year]):>8}")

# Check warming signal
first_year = sorted(yearly_mean.index)[0]
last_year = sorted(yearly_mean.index)[-1]
warming_per_year = (yearly_mean[last_year] - yearly_mean[first_year]) / (last_year - first_year)
print(f"\n  Warming trend: {warming_per_year:.3f}°C/year")
print(f"  Expected: ~0.025°C/year (IPCC/IMD)")

# Check that heatwaves exist in forecast
total_hw = int(forecast_long['heatwave_event'].sum())
print(f"  Total heatwave days forecast: {total_hw}")

# Run comprehensive prediction for enhanced stats
forecast_enhanced = comprehensive_prediction(model_df, forecast_long, city)
if "hw_probability" in forecast_enhanced.columns:
    yearly_hw_prob = forecast_enhanced.groupby(forecast_enhanced.index.year)["hw_probability"].mean()
    print(f"\n  Year-wise heatwave probability:")
    for year in sorted(yearly_hw_prob.index):
        print(f"    {year}: {yearly_hw_prob[year]:.3f}")

print(f"\n{'='*60}")
print("VERIFICATION COMPLETE")
print(f"{'='*60}")
