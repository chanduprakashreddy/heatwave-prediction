import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Map filename patterns to friendly city names
CITY_MAP = {
    "Bangalore": "Bangalore_1990_2022_BangaloreCity.csv",
    "Chennai": "Chennai_1990_2022_Madras.csv",
    "Delhi NCR": "Delhi_NCR_1990_2022_Safdarjung.csv",
    "Lucknow": "Lucknow_1990_2022.csv",
    "Mumbai": "Mumbai_1990_2022_Santacruz.csv",
    "Rajasthan (Jodhpur)": "Rajasthan_1990_2022_Jodhpur.csv",
    "Bhubaneswar": "weather_Bhubhneshwar_1990_2022.csv",
    "Rourkela": "weather_Rourkela_2021_2022.csv",
}


def get_available_cities():
    """Return list of city names that have data files."""
    cities = []
    for city, fname in CITY_MAP.items():
        if os.path.exists(os.path.join(DATA_DIR, fname)):
            cities.append(city)
    return sorted(cities)


def load_data(city_name):
    """
    Load weather CSV for a given city.
    CSV columns: time, tavg, tmin, tmax, prcp
    Returns a DataFrame indexed by date with a 'Temperature' column (tmax).
    """
    fname = CITY_MAP.get(city_name)
    if fname is None:
        raise ValueError(f"Unknown city: {city_name}")

    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)

    # Parse dates — format is DD-MM-YYYY
    df["date"] = pd.to_datetime(df["time"], format="mixed", dayfirst=True)
    df = df.sort_values("date")
    df.set_index("date", inplace=True)

    # Preserve exact dataset values for visualization and auditability.
    df["Temperature_raw"] = pd.to_numeric(df["tmax"], errors="coerce")

    # Modeling column (can be interpolated downstream in model prep).
    df["Temperature"] = df["Temperature_raw"].copy()

    # Keep only real dataset rows for plotting/analysis on original observations.
    df = df.dropna(subset=["Temperature_raw"])
    df = df[~df.index.duplicated(keep="first")]

    return df
