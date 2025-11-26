# File: scripts/build_daily_features.py

"""
This script builds the final, feature-rich DAILY dataset for model training.
"""
import pandas as pd
import os
import sys

# --- Setup Project Root Path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Configuration ---
AQI_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Post-Processing", "CSV_Files", "city_day.csv")
WEATHER_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Post-Processing", "CSV_Files", "Master_Dataset_V2.csv")
OUTPUT_FILE_PATH = os.path.join(PROJECT_ROOT, "data", "Post-Processing", "CSV_Files", "Master_Daily_Features.csv")

TARGET_CITIES = ['Bangalore', 'Chennai', 'Kolkata', 'Mumbai']
TARGET_STATIONS = {
    "Bangalore": {"lat": 12.9152, "lon": 77.6103},
    "Chennai":   {"lat": 13.16,   "lon": 80.26},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
    "Mumbai":    {"lat": 19.08,   "lon": 72.88}
}

def create_daily_features(aqi_path, weather_path, output_path):
    print("--- Starting Daily Feature Engineering ---")
    
    # --- Step 1: Load and Prepare Daily AQI Data ---
    print(f"Loading daily AQI data from: {aqi_path}")
    try:
        aqi_df = pd.read_csv(aqi_path, parse_dates=['Datetime'])
        aqi_df.rename(columns={'Datetime': 'Date'}, inplace=True)
        
        aqi_df = aqi_df[aqi_df['City'].isin(TARGET_CITIES)].copy()
        aqi_df = aqi_df[['Date', 'City', 'AQI']].dropna(subset=['AQI'])
        print(f"  -> Loaded {len(aqi_df)} daily AQI records for target cities.")
    except Exception as e:
        print(f"ERROR: Failed to load or process AQI data. Reason: {e}")
        return

    # --- Step 2: Load and Summarize Hourly Weather Data ---
    print(f"\nLoading and summarizing hourly weather data from: {weather_path}")
    try:
        weather_df = pd.read_csv(weather_path)
        weather_df['Datetime'] = pd.to_datetime(weather_df['Datetime'])
        city_cols = [f'City_{city}' for city in TARGET_CITIES]
        weather_df['City'] = weather_df[[col for col in city_cols if col in weather_df.columns]].idxmax(axis=1).str.replace('City_', '')
        
        weather_df['Date'] = weather_df['Datetime'].dt.normalize()
        aggregations = {
            'temperature_2m': ['mean', 'min', 'max'],
            'relative_humidity_2m': ['mean'], 'precipitation': ['sum'], 'wind_speed_10m': ['mean']
        }
        daily_weather_df = weather_df.groupby(['City', 'Date']).agg(aggregations).reset_index()
        daily_weather_df.columns = ['_'.join(col).strip() for col in daily_weather_df.columns.values]
        daily_weather_df.rename(columns={'City_': 'City', 'Date_': 'Date'}, inplace=True)
        print(f"  -> Created {len(daily_weather_df)} daily weather summary records.")
    except Exception as e:
        print(f"ERROR: Failed during weather processing. Reason: {e}")
        return

    # --- Step 3: Merge, Add Coordinates, and Engineer Features ---
    print("\nMerging data...")
    master_df = pd.merge(aqi_df, daily_weather_df, on=['Date', 'City'], how='inner')
    print(f"  -> Merged dataset shape: {master_df.shape}")
    
    print("Adding station coordinates...")
    lat_map = {city: details['lat'] for city, details in TARGET_STATIONS.items()}
    lon_map = {city: details['lon'] for city, details in TARGET_STATIONS.items()}
    master_df['latitude'] = master_df['City'].map(lat_map)
    master_df['longitude'] = master_df['City'].map(lon_map)
    
    print("Engineering time and lag features...")
    master_df = master_df.sort_values(by=['City', 'Date']).reset_index(drop=True)
    master_df['day_of_week'] = master_df['Date'].dt.dayofweek
    master_df['month'] = master_df['Date'].dt.month
    master_df['year'] = master_df['Date'].dt.year
    master_df['AQI_lag_1_day'] = master_df.groupby('City')['AQI'].shift(1)
    master_df['AQI_lag_7_day'] = master_df.groupby('City')['AQI'].shift(7)
    master_df.dropna(inplace=True)
    
    # --- Step 4: Save Final Dataset ---
    print("\n--- Feature Engineering Complete ---")
    print(f"Final dataset shape: {master_df.shape}")
    master_df.to_csv(output_path, index=False)
    print(f"Successfully saved final dataset to: {output_path}")

if __name__ == "__main__":
    create_daily_features(AQI_DATA_PATH, WEATHER_DATA_PATH, OUTPUT_FILE_PATH)