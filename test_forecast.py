import pandas as pd
from utils import load_data
from forecast import predict_future
from model_training import train_xgb_model, train_sarima_model
from prediction_engine import comprehensive_prediction
import json

df = load_data("Bangalore")
xgb, _ = train_xgb_model(df, "Temperature", "Bangalore")
sarima = train_sarima_model(df, "Temperature", "Bangalore")

forecast_df, _, _ = predict_future(df, "Temperature", days=1000, city_name="Bangalore", xgb_model=xgb, model=sarima)
print("Max forecasted temp before advanced:", forecast_df["forecast_temp"].max())

forecast_df = comprehensive_prediction(df, forecast_df, "Bangalore")
print("Max hw_probability:", forecast_df["hw_probability"].max())
print("Max predicted_anomaly:", forecast_df.get("predicted_anomaly", pd.Series([0])).max())
print("Sum hw_predicted:", forecast_df.get("hw_predicted", pd.Series([0])).sum())
