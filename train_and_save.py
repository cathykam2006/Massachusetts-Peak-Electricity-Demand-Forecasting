"""
train_and_save.py

Run this ONCE, locally, before deploying to Streamlit. It rebuilds the dataset
(using Notebook 1's pipeline), trains XGBoost and LightGBM, and saves everything
the Streamlit app needs to disk: the combined dataset, trained models, feature
list, and test-set predictions.

Usage:
    python train_and_save.py

Requires an EIA API key — set it as an environment variable before running:
    export EIA_API_KEY="your_key_here"      (Mac/Linux)
    set EIA_API_KEY=your_key_here           (Windows)
"""

import os
import joblib
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import shap
import holidays
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

API_KEY = os.environ.get("EIA_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the EIA_API_KEY environment variable before running this script.")

LAT, LON = 42.3601, -71.0589  # Boston, MA
START_DATE, END_DATE = "2021-01-01", "2023-12-31"


# ---------------------------------------------------------------------------
# 1. Pull data
# ---------------------------------------------------------------------------
def get_eia_data(start_date, end_date, api_key, offset=0, length=5000):
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    params = {
        "api_key": api_key, "frequency": "hourly", "data[0]": "value",
        "facets[respondent][]": "ISNE", "facets[type][]": "D",
        "start": start_date, "end": end_date,
        "sort[0][column]": "period", "sort[0][direction]": "asc",
        "offset": offset, "length": length,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()


def pull_demand(start_date, end_date, api_key):
    all_records, offset = [], 0
    while True:
        data = get_eia_data(start_date, end_date, api_key, offset=offset)
        records = data["response"]["data"]
        if not records:
            break
        all_records.extend(records)
        offset += 5000
        if len(records) < 5000:
            break
    df = pd.DataFrame(all_records)
    df["period"] = pd.to_datetime(df["period"])
    df = df.sort_values("period").set_index("period")
    df = df.rename(columns={"value": "Demand (MW)"})
    df["Demand (MW)"] = pd.to_numeric(df["Demand (MW)"], errors="coerce")
    return df.dropna(subset=["Demand (MW)"])


def pull_weather(start_date, end_date, lat=LAT, lon=LON):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    weather_df = pd.DataFrame({
        "period": data["hourly"]["time"],
        "Temperature (°C)": data["hourly"]["temperature_2m"],
        "Humidity (%)": data["hourly"]["relative_humidity_2m"],
        "Wind Speed (m/s)": data["hourly"]["wind_speed_10m"],
        "Solar Radiation (W/m²)": data["hourly"]["shortwave_radiation"],
    })
    weather_df["period"] = pd.to_datetime(weather_df["period"])
    return weather_df.set_index("period")


def get_eia_fuel_data(start_date, end_date, api_key, fueltype, offset=0, length=5000):
    url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
    params = {
        "api_key": api_key, "frequency": "hourly", "data[0]": "value",
        "facets[respondent][]": "ISNE", "facets[fueltype][]": fueltype,
        "start": start_date, "end": end_date,
        "sort[0][column]": "period", "sort[0][direction]": "asc",
        "offset": offset, "length": length,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()


def pull_fuel_series(fueltype, start_date, end_date, api_key, colname):
    all_records, offset = [], 0
    while True:
        data = get_eia_fuel_data(start_date, end_date, api_key, fueltype, offset=offset)
        records = data["response"]["data"]
        if not records:
            break
        all_records.extend(records)
        offset += 5000
        if len(records) < 5000:
            break
    fuel_df = pd.DataFrame(all_records)
    fuel_df["period"] = pd.to_datetime(fuel_df["period"])
    fuel_df["value"] = pd.to_numeric(fuel_df["value"], errors="coerce")
    return fuel_df.set_index("period")[["value"]].rename(columns={"value": colname})


print("Pulling demand data...")
demand_start = f"{START_DATE}T00"
demand_end = f"{END_DATE}T23"
df = pull_demand(demand_start, demand_end, API_KEY)

print("Pulling weather data...")
weather_df = pull_weather(START_DATE, END_DATE)

print("Pulling solar/wind generation data...")
solar_clean = pull_fuel_series("SUN", demand_start, demand_end, API_KEY, "Solar Generation (MWh)")
wind_clean = pull_fuel_series("WND", demand_start, demand_end, API_KEY, "Wind Generation (MWh)")

# ---------------------------------------------------------------------------
# 2. Combine + fix timezone ONCE (see Notebook 1 for why this ordering matters)
# ---------------------------------------------------------------------------
print("Combining and cleaning...")
df_combined = df.join(weather_df, how="inner")
df_combined = df_combined.join(solar_clean, how="left").join(wind_clean, how="left")

df_combined.index = df_combined.index.tz_localize("UTC").tz_convert("America/New_York")
df_combined["hour"] = df_combined.index.hour
df_combined["dayofweek"] = df_combined.index.dayofweek
df_combined["month"] = df_combined.index.month
df_combined["is_weekend"] = df_combined["dayofweek"].isin([5, 6]).astype(int)

us_holidays = holidays.US(years=[2020, 2021, 2022, 2023])
df_combined["is_holiday"] = df_combined.index.date
df_combined["is_holiday"] = df_combined["is_holiday"].apply(lambda d: d in us_holidays).astype(int)

target = "Demand (MW)"
df_combined["load_lag_1h"] = df_combined[target].shift(1)
df_combined["load_lag_24h"] = df_combined[target].shift(24)
df_combined["load_lag_168h"] = df_combined[target].shift(168)
df_combined["load_roll_mean_24h"] = df_combined[target].rolling(24).mean()
df_combined["cooling_degree"] = (df_combined["Temperature (°C)"] - 18).clip(lower=0)
df_combined["heating_degree"] = (18 - df_combined["Temperature (°C)"]).clip(lower=0)

df_combined = df_combined.dropna()
df_combined.to_csv("data/isone_full_dataset.csv")
print(f"Saved combined dataset: {df_combined.shape}")

# ---------------------------------------------------------------------------
# 3. Train/test split + models
# ---------------------------------------------------------------------------
feature_cols = [
    "hour", "dayofweek", "month", "is_weekend", "is_holiday",
    "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Solar Radiation (W/m²)",
    "Solar Generation (MWh)", "Wind Generation (MWh)",
    "load_lag_1h", "load_lag_24h", "load_lag_168h", "load_roll_mean_24h",
    "cooling_degree", "heating_degree",
]

df_combined = df_combined.sort_index()
split_date = df_combined.index.max() - pd.Timedelta(days=90)
train = df_combined[df_combined.index <= split_date]
test = df_combined[df_combined.index > split_date]

print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
xgb_model.fit(train[feature_cols], train[target])
xgb_pred = xgb_model.predict(test[feature_cols])

print("Training LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1,
)
lgb_model.fit(train[feature_cols], train[target])
lgb_pred = lgb_model.predict(test[feature_cols])

ensemble_pred = (xgb_pred + lgb_pred) / 2

# Baseline, for the comparison table
seasonal_avg = train.groupby(["dayofweek", "hour"])[target].mean()
baseline_pred = test.apply(lambda row: seasonal_avg.loc[row["dayofweek"], row["hour"]], axis=1)


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": round(mae, 1), "RMSE": round(rmse, 1), "MAPE": round(mape, 2)}


comparison = {
    "Baseline (seasonal avg)": metrics(test[target], baseline_pred),
    "XGBoost": metrics(test[target], xgb_pred),
    "LightGBM": metrics(test[target], lgb_pred),
    "Ensemble": metrics(test[target], ensemble_pred),
}
comparison_df = pd.DataFrame(comparison).T
comparison_df.to_csv("data/model_comparison.csv")
print(comparison_df)

# ---------------------------------------------------------------------------
# 4. SHAP values (precomputed, so the app doesn't need the `shap` explainer live)
# ---------------------------------------------------------------------------
print("Computing SHAP values...")
explainer = shap.TreeExplainer(xgb_model)
shap_values_test = explainer.shap_values(test[feature_cols])

# ---------------------------------------------------------------------------
# 5. Save everything the app needs
# ---------------------------------------------------------------------------
print("Saving artifacts...")
joblib.dump(xgb_model, "models/xgb_model.pkl")
joblib.dump(lgb_model, "models/lgb_model.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

test_out = test.copy()
test_out["xgb_pred"] = xgb_pred
test_out["lgb_pred"] = lgb_pred
test_out["ensemble_pred"] = ensemble_pred
test_out["baseline_pred"] = baseline_pred
test_out.to_csv("data/test_predictions.csv")

np.save("data/shap_values_test.npy", shap_values_test)

print("Done. Artifacts saved to data/ and models/ — ready for the Streamlit app.")
