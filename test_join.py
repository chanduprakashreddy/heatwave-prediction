import pandas as pd
from utils import load_data
from forecast import predict_future
from model_training import train_xgb_model, train_sarima_model

df = load_data("Bangalore")
from anomaly import detect_anomalies
df = detect_anomalies(df, "Temperature")

xgb, _ = train_xgb_model(df, "Temperature", "Bangalore")
sarima = train_sarima_model(df, "Temperature", "Bangalore")

forecast_df, _, _ = predict_future(df, "Temperature", days=1000, city_name="Bangalore", xgb_model=xgb, model=sarima)

history_df = df
temp_col = "Temperature"

combined_df = pd.concat([history_df[[temp_col]], 
                         pd.DataFrame({temp_col: forecast_df["forecast_temp"]},
                                    index=forecast_df.index)])
combined_df = combined_df.rename(columns={temp_col: "Temperature"})

if "climatology" in history_df.columns:
    hist_clim = history_df[["climatology"]]
    fc_clim = pd.DataFrame({"climatology": forecast_df["climatology"]},
                           index=forecast_df.index)
    concat_clim = pd.concat([hist_clim, fc_clim])
    combined_df = combined_df.join(concat_clim)

print(combined_df.columns)
print(combined_df["climatology"].isna().sum())
