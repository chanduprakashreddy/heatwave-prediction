from utils import load_data
from anomaly import detect_anomalies
from forecast import predict_future
import numpy as np

TEMP_COL = "Temperature"

df = load_data("Delhi NCR")
df = df.loc["1990":"2022"]

print("Max Temp after load:", df[TEMP_COL].max())

df = detect_anomalies(df, TEMP_COL)
print("Max Temp after anomalies:", df[TEMP_COL].max())

hist_temps = np.round(df[TEMP_COL].values, 2).tolist()
hist_dates = df.index.strftime("%Y-%m-%d").tolist()

max_t = max(hist_temps)
idx = hist_temps.index(max_t)
print("Max Temp in payload array:", max_t)
print("Date of max temp in payload array:", hist_dates[idx])
