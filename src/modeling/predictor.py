# File: src/modeling/predictor.py
"""
Handles loading saved DAILY LightGBM models and generating a 3-day AQI forecast.
"""

import pandas as pd
import os
import sys
import logging
import joblib
import requests
from datetime import timedelta
import math

# --- Setup Project Root Path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import Project Modules & Dependencies ---
try:
    from src.config_loader import CONFIG
    from src.health_rules.info import get_aqi_info
    from src.exceptions import ModelFileNotFoundError, PredictionError
    from src.api_integration.client import get_current_aqi_for_city
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"Predictor: Could not import dependencies. Using dummy fallbacks. Error: {e}")
    class ModelFileNotFoundError(FileNotFoundError): pass
    class PredictionError(Exception): pass
    def get_aqi_info(aqi_value): return {'level': 'N/A', 'color': '#DDDDDD', 'implications': 'AQI info unavailable.'}
    def get_current_aqi_for_city(city_name):
        logging.error("Dummy get_current_aqi_for_city called due to import error.")
        # Return a dictionary with an 'error' key to signal failure
        return {'error': 'AQI client unavailable'}

log = logging.getLogger(__name__)

# --- Configuration ---
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Post-Processing", "CSV_Files", "Master_Daily_Features.csv")
WEATHER_FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
MAX_RESIDUAL_CAP = CONFIG.get('modeling', {}).get('max_residual_cap', 75)
_loaded_models_cache = {}

def load_lgbm_model(city_name: str):
    """Loads a trained daily LightGBM model for a specific city."""
    global _loaded_models_cache
    if city_name in _loaded_models_cache:
        log.info(f"Returning cached LightGBM model for {city_name}.")
        return _loaded_models_cache[city_name]

    model_filename = f"{city_name}_lgbm_daily_model.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    if not os.path.exists(model_path):
        raise ModelFileNotFoundError(f"Model file not found: {model_path}")
        
    model = joblib.load(model_path)
    _loaded_models_cache[city_name] = model
    log.info(f"Model for {city_name} loaded successfully.")
    return model


def get_daily_summary_forecast(city_name: str, days_ahead: int = 3):
    """
    High-level function to generate a daily AQI forecast for the next 3 days.
    """
    log.info(f"--- Starting {days_ahead}-day forecast for {city_name} ---")
    try:
        # --- Step 1: Get the Anchor - The Live AQI Value ---
        log.info(f"Fetching live AQI for {city_name} to anchor forecast...")
        live_aqi_data = get_current_aqi_for_city(f"{city_name}, India")
        if not (live_aqi_data and live_aqi_data.get('aqi') is not None):
            raise PredictionError(f"Could not retrieve live AQI for {city_name}.")
        live_aqi_value = live_aqi_data['aqi']
        log.info(f"Successfully fetched live AQI anchor: {live_aqi_value}")
        
        # --- Step 2: Generate a Raw Multi-Day Forecast ---
        # We will generate a forecast starting from today to get the model's expected trend.
        model = load_lgbm_model(city_name)
        full_df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
        city_history = full_df[full_df['City'] == city_name].tail(7)
        if len(city_history) < 7: raise PredictionError("Not enough history for lags.")
        
        last_known_row = city_history.iloc[-1]
        params = {
            "latitude": last_known_row['latitude'], "longitude": last_known_row['longitude'],
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_mean",
            "forecast_days": 7 
        }
        response = requests.get(WEATHER_FORECAST_API_URL, params=params)
        response.raise_for_status()
        weather_df = pd.DataFrame(response.json()['daily'])
        weather_df.rename(columns={'time': 'Date'}, inplace=True)
        weather_df['Date']
        
        # Iteratively generate a "raw" forecast for today and the next 3 days
        raw_predictions = []
        current_history = city_history.reset_index(drop=True)
        today_date = pd.Timestamp.now().normalize()
        
        for i in range(days_ahead + 1): # Predict today + future days
            target_date = today_date + timedelta(days=i)
            weather_df['Date'] = pd.to_datetime(weather_df['Date'])
            weather_for_day = weather_df[weather_df['Date'] == target_date]
            if weather_for_day.empty: raise PredictionError(f"Missing weather for {target_date.date()}")
            
            last_known_day = current_history.iloc[-1]
            seven_days_ago = current_history.iloc[-7]
            
            features = { 'temperature_2m_mean': weather_for_day['temperature_2m_mean'].iloc[0], 'temperature_2m_min': weather_for_day['temperature_2m_min'].iloc[0], 'temperature_2m_max': weather_for_day['temperature_2m_max'].iloc[0], 'relative_humidity_2m_mean': weather_for_day['relative_humidity_2m_mean'].iloc[0], 'precipitation_sum': weather_for_day['precipitation_sum'].iloc[0], 'wind_speed_10m_mean': weather_for_day['wind_speed_10m_mean'].iloc[0], 'day_of_week': target_date.dayofweek, 'month': target_date.month, 'year': target_date.year, 'AQI_lag_1_day': last_known_day['AQI'], 'AQI_lag_7_day': seven_days_ago['AQI'] }
            features_df = pd.DataFrame([features], columns=model.feature_name_)
            prediction = model.predict(features_df)[0]
            
            raw_predictions.append(prediction)
            
            new_history_row = pd.DataFrame([{'Date': target_date, 'AQI': prediction}])
            current_history = pd.concat([current_history.iloc[1:], new_history_row], ignore_index=True)
            
        raw_forecast_df = pd.DataFrame({
            'date': pd.date_range(start=today_date, periods=days_ahead + 1),
            'raw_pred': raw_predictions
        })
        log.info(f"Generated raw model forecast:\n{raw_forecast_df}")
        
        # --- Step 3: Apply the "Anchor and Trend" Correction ---
        prediction_for_today = raw_forecast_df['raw_pred'].iloc[0]
        live_residual = live_aqi_value - prediction_for_today
        
        # Apply the residual to the model's prediction for TODAY.
        # This becomes the starting point for our final forecast.
        final_forecast_values = [live_aqi_value]
        
        # Now, calculate the day-to-day CHANGES from the raw forecast.
        # Example: if raw forecast is [180, 190, 185], the changes are [+10, -5]
        day_to_day_changes = raw_forecast_df['raw_pred'].diff().dropna()
        
        # Apply these changes to our anchored value.
        for change in day_to_day_changes:
            
            # --- START: New Dynamic Smoothing Logic ---
            
            # Define thresholds for the residual.
            low_residual_threshold = 20  # If error is less than 20, trust the model more.
            high_residual_threshold = 100 # If error is over 100, trust the model very little.

            # Define smoothing factors based on trust.
            trust_factor_high = 0.8  # Trust the model's trend 80%
            trust_factor_low = 0.2   # Only trust the model's trend 20%
            
            abs_residual = abs(live_residual)

            if abs_residual <= low_residual_threshold:
                smoothing_factor = trust_factor_high
            elif abs_residual >= high_residual_threshold:
                smoothing_factor = trust_factor_low
            else:
                # Linearly interpolate the smoothing factor for residuals between the thresholds.
                # This creates a smooth transition from high trust to low trust.
                slope = (trust_factor_low - trust_factor_high) / (high_residual_threshold - low_residual_threshold)
                smoothing_factor = trust_factor_high + slope * (abs_residual - low_residual_threshold)

            log.info(f"Live residual is {live_residual:.1f}. Applying dynamic smoothing factor of {smoothing_factor:.2f}.")
            smoothed_change = change * smoothing_factor
            
            # --- END: New Dynamic Smoothing Logic ---
            
            next_day_value = final_forecast_values[-1] + smoothed_change
            final_forecast_values.append(next_day_value)
            
        # We only need the future days, so we skip the first value (today's anchor)
        final_forecast_for_ui = final_forecast_values[1:]

        # 4. Format for the UI
        ui_list = []
        future_dates = pd.date_range(start=today_date + timedelta(days=1), periods=days_ahead)
        for i, calibrated_aqi in enumerate(final_forecast_for_ui):
            calibrated_aqi = max(1, calibrated_aqi) # Ensure not negative
            aqi_info = get_aqi_info(calibrated_aqi)
            ui_list.append({
                "date": future_dates[i].strftime('%Y-%m-%d'),
                "predicted_aqi": round(calibrated_aqi),
                "level": aqi_info.get("level", "N/A"),
                "color": aqi_info.get("color", "#FFFFFF"),
                "implications": aqi_info.get("implications", "N/A")
            })
        # 6. Log the predictions for performance tracking
        try:
            log_file_path = os.path.join(PROJECT_ROOT, "predictions_log.csv")
            log_df = pd.DataFrame(ui_list)
            log_df['city'] = city_name # Add the city name for grouping

            if not os.path.exists(log_file_path):
                # If the log file doesn't exist, create it with a header.
                log_df.to_csv(log_file_path, index=False, header=True)
                log.info(f"Created new prediction log at: {log_file_path}")
            else:
                # If it exists, append the new predictions without the header.
                log_df.to_csv(log_file_path, mode='a', index=False, header=False)
                log.info(f"Appended {len(log_df)} predictions to log for {city_name}.")
                
        except Exception as log_e:
            # A logging failure should not crash the main forecast.
            log.error(f"Failed to write predictions to log file: {log_e}", exc_info=True)

        return ui_list
    
    except Exception as e:
        log.error(f"An unexpected error in get_daily_summary_forecast for {city_name}: {e}", exc_info=True)
        return []

# --- Main Execution ---
if __name__ == "__main__":
    test_city = "Mumbai"
    print(f"\nGenerating daily forecast for {test_city}...")
    daily_forecast = get_daily_summary_forecast(test_city)
    
    if daily_forecast:
        import pprint
        print("\n--- 3-Day Daily Summary Forecast ---")
        pprint.pprint(daily_forecast)
    else:
        print(f"\nFailed to generate forecast for {test_city}.")