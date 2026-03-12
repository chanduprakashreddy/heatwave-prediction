import requests
import json
import time

url = "http://127.0.0.1:5000/api/analyze"
payload = {"city": "Bangalore", "startYear": "2015", "endYear": "2025"}

print(f"Testing {url} with {payload}")
try:
    start_time = time.time()
    response = requests.post(url, json=payload, timeout=120)
    end_time = time.time()
    print(f"Request took {end_time - start_time:.2f} seconds")
    
    if response.status_code == 200:
        data = response.json()
        print("Success:", data["success"])
        print("Forecast stats:", data["stats"])
        if "forecast" in data:
            print("Forecast max date:", data["forecast"]["dates"][-1])
            print("Number of forecast days:", len(data["forecast"]["dates"]))
            
            # Check if advanced anomalies are applied
            hw_count = sum(data["forecast"]["heatwave"])
            print(f"Forecast heatwave events: {hw_count}")
        else:
            print("No forecast key in response")
    else:
        print("Error:", response.status_code)
        print(response.text)
except Exception as e:
    print(f"Failed: {e}")
