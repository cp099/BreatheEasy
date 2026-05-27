# File: scripts/rebuild_data_and_models.py
"""
Rebuilds the training and historical summary datasets using clean historical
data from Open-Meteo (2022 to present) and retrains all LightGBM models.
"""
import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta

# --- Setup Project Root Path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.health_rules.calculator import calculate_aqi_from_pollutants
from src.modeling.train import train_and_save_models

TARGET_STATIONS = {
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777}
}

START_DATE = "2022-10-01"
END_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Post-Processing", "CSV_Files")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def download_and_rebuild():
    print(f"--- Rebuilding Data from {START_DATE} to {END_DATE} ---")
    all_cities_dfs = []
    
    for city, coords in TARGET_STATIONS.items():
        lat = coords["lat"]
        lon = coords["lon"]
        print(f"\nFetching data for {city} (Lat: {lat}, Lon: {lon})...")
        
        # 1. Fetch historical weather
        weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={START_DATE}&end_date={END_DATE}&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&timezone=GMT"
        weather_resp = requests.get(weather_url)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
        
        hourly_w = pd.DataFrame(weather_data["hourly"])
        hourly_w["time"] = pd.to_datetime(hourly_w["time"])
        hourly_w.set_index("time", inplace=True)
        
        daily_w = hourly_w.resample('D').agg({
            'temperature_2m': ['mean', 'min', 'max'],
            'relative_humidity_2m': 'mean',
            'precipitation': 'sum',
            'wind_speed_10m': 'mean'
        })
        daily_w.columns = [
            'temperature_2m_mean', 'temperature_2m_min', 'temperature_2m_max',
            'relative_humidity_2m_mean', 'precipitation_sum', 'wind_speed_10m_mean'
        ]
        
        # 2. Fetch historical air quality
        aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&start_date={START_DATE}&end_date={END_DATE}&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&timezone=GMT"
        aq_resp = requests.get(aq_url)
        aq_resp.raise_for_status()
        aq_data = aq_resp.json()
        
        hourly_aq = pd.DataFrame(aq_data["hourly"])
        hourly_aq["time"] = pd.to_datetime(hourly_aq["time"])
        hourly_aq.set_index("time", inplace=True)
        
        daily_aq = hourly_aq.resample('D').mean()
        
        # Format columns for calculate_aqi_from_pollutants
        daily_aq['PM2.5'] = daily_aq['pm2_5']
        daily_aq['PM10'] = daily_aq['pm10']
        daily_aq['NO2'] = daily_aq['nitrogen_dioxide']
        daily_aq['O3'] = daily_aq['ozone']
        daily_aq['CO'] = daily_aq['carbon_monoxide'] / 1000.0  # Convert ug/m3 to mg/m3
        daily_aq['SO2'] = daily_aq['sulphur_dioxide']
        
        # Calculate daily CPCB AQI
        print(f"Calculating daily AQI for {city}...")
        daily_aq['AQI'] = daily_aq.apply(calculate_aqi_from_pollutants, axis=1)
        
        # Merge weather and AQI
        merged = pd.merge(daily_w, daily_aq[['AQI']], left_index=True, right_index=True, how='inner')
        merged['City'] = city
        merged['latitude'] = lat
        merged['longitude'] = lon
        
        all_cities_dfs.append(merged)
        
    # Combine data
    master_df = pd.concat(all_cities_dfs)
    master_df.index.name = 'Date'
    master_df = master_df.reset_index()
    master_df = master_df.sort_values(by=['City', 'Date']).reset_index(drop=True)
    
    # Calculate date components and lag features
    master_df['day_of_week'] = master_df['Date'].dt.dayofweek
    master_df['month'] = master_df['Date'].dt.month
    master_df['year'] = master_df['Date'].dt.year
    
    master_df['AQI_lag_1_day'] = master_df.groupby('City')['AQI'].shift(1)
    master_df['AQI_lag_7_day'] = master_df.groupby('City')['AQI'].shift(7)
    
    # Drop rows with NaNs (created by shift lags)
    master_df.dropna(subset=['AQI_lag_1_day', 'AQI_lag_7_day', 'AQI'], inplace=True)
    
    # Save training dataset
    os.makedirs(DATA_DIR, exist_ok=True)
    features_path = os.path.join(DATA_DIR, "Master_Daily_Features.csv")
    master_df.to_csv(features_path, index=False)
    print(f"\nSaved clean daily features to: {features_path}")
    
    # Overwrite the historical summary dataset
    history_df = master_df[['Date', 'City', 'AQI']].copy()
    history_df.rename(columns={'Date': 'Datetime'}, inplace=True)
    # Datetime should be timezone naive format matching dashboard query expectations
    history_df['Datetime'] = history_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    history_path = os.path.join(DATA_DIR, "Master_Features_AQI_Data.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Saved clean historical trend lines to: {history_path}")
    
    # Retrain the LightGBM models
    print("\nStarting model training...")
    train_and_save_models(features_path, MODELS_DIR)
    print("\n--- Process Complete! ---")

if __name__ == "__main__":
    download_and_rebuild()
