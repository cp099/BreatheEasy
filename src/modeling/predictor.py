# File: src/modeling/predictor.py

import pandas as pd
import os
import sys
import logging
import json
from prophet import Prophet
from prophet.serialize import model_from_json

# --- Setup Project Root Path ---
# Ensure project modules can be imported
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    logging.info(f"Predictor: Project root determined using __file__: {PROJECT_ROOT}")
except NameError:
    PROJECT_ROOT = os.path.abspath('.')
    logging.warning(f"Predictor: __file__ not defined. Assuming CWD is project root: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)
     logging.info(f"Predictor: Added project root to sys.path: {PROJECT_ROOT}")

# Import necessary functions from other project modules
try:
    # Need the API client for the real-time adjustment part
    from src.api_integration.client import get_current_aqi_for_city
    logging.info("Predictor: Successfully imported project module 'src.api_integration.client'.")
except ModuleNotFoundError:
    logging.error("Predictor: Could not import API client module. Ensure structure is correct.")
    raise
except ImportError as e:
     logging.error(f"Predictor: Error importing API client module: {e}")
     raise

# --- Logging Configuration ---
# Configure logger specifically for this module if needed, or rely on root logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - predictor.py - %(message)s')
# Using root logger configured elsewhere is usually fine
log = logging.getLogger(__name__) # Get logger for this module

# --- Configuration ---
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_VERSION = "v2" # Should match the version used in train.py

# --- Model Loading Cache ---
# Cache loaded models in memory to avoid repeated file reads/deserialization
_loaded_models_cache = {}

def load_prophet_model(city_name, version=MODEL_VERSION, models_dir=MODELS_DIR):
    """Loads a previously trained and saved Prophet model for a specific city."""
    global _loaded_models_cache

    model_key = f"{city_name}_{version}"
    if model_key in _loaded_models_cache:
        log.info(f"Returning cached model for {city_name} (version {version}).")
        return _loaded_models_cache[model_key]

    model_filename = f"{city_name}_prophet_model_{version}.json"
    model_path = os.path.join(models_dir, model_filename)

    log.info(f"Loading model for {city_name} (version {version}) from: {model_path}")
    if not os.path.exists(model_path):
        log.error(f"Model file not found: {model_path}")
        return None

    try:
        with open(model_path, 'r') as fin:
            model = model_from_json(json.load(fin)) # Deserialize model from JSON
        log.info(f"Model for {city_name} loaded successfully.")
        _loaded_models_cache[model_key] = model # Cache the loaded model
        return model
    except Exception as e:
        log.error(f"Error loading model for {city_name} from {model_path}: {e}", exc_info=True)
        return None

# --- Prediction Function (Adapted from Notebook Cell 4) ---

def generate_forecast(target_city, days_ahead=5, apply_residual_correction=True, last_known_aqi=None):
    """
    Generates AQI forecast for the next few days for a given city,
    loading the appropriate saved model and optionally applying residual correction.

    Args:
        target_city (str): The city name (e.g., 'Delhi').
        days_ahead (int): Number of days to forecast (default 5).
        apply_residual_correction (bool): Whether to fetch/use current AQI for correction.
        last_known_aqi (float, optional): The most recent actual AQI value to use for correction
                                          instead of calling the API. Assumed to be for the model's
                                          last training date.

    Returns:
        pd.DataFrame: DataFrame with 'ds' (date), 'yhat_adjusted' (final forecast),
                      'yhat' (original forecast), 'yhat_lower'/'yhat_upper' (uncertainty),
                      'residual' (applied correction). Returns None on failure.
    """
    log.info(f"Generating {days_ahead}-day forecast for {target_city}...")

    # 1. Load the saved model for the city
    trained_model = load_prophet_model(target_city)
    if trained_model is None:
        log.error(f"Failed to load model for {target_city}. Cannot generate forecast.")
        return None

    # 2. Create future dates dataframe
    try:
        future_dates_df = trained_model.make_future_dataframe(periods=days_ahead, include_history=False)
        log.info(f"Predicting for dates: {future_dates_df['ds'].min().date()} to {future_dates_df['ds'].max().date()}")
    except Exception as e:
         log.error(f"Error creating future dataframe: {e}")
         return None
    if future_dates_df.empty:
        log.error("make_future_dataframe returned empty dataframe.")
        return None

    # --- Placeholder for adding FUTURE regressors ---
    # If model expects regressors, add their future values to future_dates_df here
    # --- End Placeholder ---

    # 3. Generate initial forecast
    try:
        forecast = trained_model.predict(future_dates_df)
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        log.error(f"Failed to generate initial forecast for {target_city}: {e}")
        return None

    # Initialize adjusted columns
    forecast['yhat_adjusted'] = forecast['yhat'].copy()
    forecast['residual'] = 0.0

    # 4. Apply Residual Correction (Optional)
    actual_today = None
    if apply_residual_correction:
        log.info(f"Attempting residual correction for {target_city}...")
        try:
            # "Today" for correction is the last date the loaded model was trained on
            today_ds = trained_model.history_dates.max()
            log.info(f"Using model's last training date as 'today' for correction: {today_ds.date()}")

            today_df = pd.DataFrame({'ds': [today_ds]})
            # --- Placeholder for adding regressors for "today" ---
            # --- End Placeholder ---
            today_forecast = trained_model.predict(today_df)

            if not today_forecast.empty:
                predicted_today = today_forecast['yhat'].iloc[0]
                log.info(f"Model's prediction for {today_ds.date()}: {predicted_today:.2f}")

                # Get actual value
                if last_known_aqi is not None:
                     actual_today = float(last_known_aqi)
                     log.info(f"Using provided last_known_aqi for {today_ds.date()}: {actual_today}")
                else:
                    log.info(f"Fetching current AQI from API for {target_city}...")
                    current_aqi_info = get_current_aqi_for_city(target_city)
                    if current_aqi_info and 'aqi' in current_aqi_info:
                        actual_today = float(current_aqi_info['aqi'])
                        log.info(f"Actual current AQI from API: {actual_today}")
                    else:
                         log.warning(f"Could not get current AQI from API for {target_city}. Skipping residual correction.")

                # Apply correction if actual value was obtained
                if actual_today is not None:
                    residual = actual_today - predicted_today
                    forecast['residual'] = residual
                    log.info(f"Calculated residual: {residual:.2f}")
                    forecast['yhat_adjusted'] = (forecast['yhat'] + residual).clip(lower=0) # Add residual and clip >= 0
                    log.info(f"Applied residual correction.")
                else:
                     forecast['residual'] = 0.0
                     forecast['yhat_adjusted'] = forecast['yhat'].copy()
            else:
                 log.warning(f"Could not generate model prediction for {today_ds.date()}. Skipping residual correction.")
                 forecast['residual'] = 0.0
                 forecast['yhat_adjusted'] = forecast['yhat'].copy()
        except Exception as e:
            log.error(f"Error during residual correction: {e}. Proceeding without correction.", exc_info=True)
            forecast['residual'] = 0.0
            forecast['yhat_adjusted'] = forecast['yhat'].copy()
    else:
        log.info("Residual correction not applied as per request.")

    # 5. Return the forecast dataframe
    log.info(f"Forecast generation complete for {target_city}.")
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'residual', 'yhat_adjusted']]

# --- Helper Function for UI Formatting ---

def format_forecast_for_ui(forecast_df):
    """
    Formats the forecast DataFrame into a list of dictionaries suitable for UI (JSON).

    Args:
        forecast_df (pd.DataFrame): The DataFrame returned by generate_forecast.

    Returns:
        list: A list of dictionaries, each with 'date' (YYYY-MM-DD string)
              and 'predicted_aqi' (rounded integer). Returns empty list if input is None or empty.
              Example: [{'date': '2025-01-01', 'predicted_aqi': 156}, ...]
    """
    if forecast_df is None or forecast_df.empty:
        return []

    # Select relevant columns and create list
    ui_data = []
    for index, row in forecast_df.iterrows():
        try:
             # Ensure date is formatted correctly, handle potential NaT
             date_str = pd.to_datetime(row['ds']).strftime('%Y-%m-%d') if pd.notna(row['ds']) else None
             # Use the adjusted forecast, round to nearest integer, handle potential NaN/None
             predicted_aqi = int(round(row['yhat_adjusted'])) if pd.notna(row['yhat_adjusted']) else None

             if date_str is not None and predicted_aqi is not None:
                 ui_data.append({
                     'date': date_str,
                     'predicted_aqi': predicted_aqi
                 })
             else:
                  log.warning(f"Skipping row due to missing date or predicted_aqi: {row}")

        except Exception as e:
             log.error(f"Error formatting row {index} for UI: {row}. Error: {e}")
             continue # Skip row on error

    return ui_data


# --- Example Usage Block (for testing this module directly) ---
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s')

    print("\n" + "="*30)
    print(" Running predictor.py Tests ")
    print("="*30 + "\n")

    test_city = "Delhi" # City must have a saved model file
    days_to_forecast = 5

    # --- Test 1: Predict WITH API correction ---
    print(f"\n--- Test 1: Predicting {days_to_forecast} days for '{test_city}' WITH API Correction ---")
    forecast_api = generate_forecast(test_city, days_ahead=days_to_forecast, apply_residual_correction=True)
    if forecast_api is not None:
        print("Forecast (API Corrected):")
        print(forecast_api)
    else:
        print(f"Failed to generate forecast for {test_city}.")

    # --- Test 2: Predict WITH manual correction ---
    simulated_aqi = 150.0 # Example last known AQI
    print(f"\n--- Test 2: Predicting {days_to_forecast} days for '{test_city}' WITH Manual Correction (Last AQI={simulated_aqi}) ---")
    forecast_manual = generate_forecast(test_city, days_ahead=days_to_forecast, apply_residual_correction=True, last_known_aqi=simulated_aqi)
    if forecast_manual is not None:
        print("Forecast (Manual Corrected):")
        print(forecast_manual)
    else:
        print(f"Failed to generate forecast for {test_city}.")


    # --- Test 3: Predict WITHOUT correction ---
    print(f"\n--- Test 3: Predicting {days_to_forecast} days for '{test_city}' WITHOUT Correction ---")
    forecast_raw = generate_forecast(test_city, days_ahead=days_to_forecast, apply_residual_correction=False)
    if forecast_raw is not None:
        print("Forecast (Raw):")
        print(forecast_raw)
    else:
        print(f"Failed to generate forecast for {test_city}.")

    # --- Test 4: Predict for another city (e.g., Mumbai) ---
    test_city_2 = "Mumbai"
    if os.path.exists(os.path.join(MODELS_DIR, f"{test_city_2}_prophet_model_{MODEL_VERSION}.json")):
         print(f"\n--- Test 4: Predicting {days_to_forecast} days for '{test_city_2}' WITH API Correction ---")
         forecast_mumbai = generate_forecast(test_city_2, days_ahead=days_to_forecast, apply_residual_correction=True)
         if forecast_mumbai is not None:
             print(f"Forecast for {test_city_2} (API Corrected):")
             print(forecast_mumbai)
         else:
             print(f"Failed to generate forecast for {test_city_2}.")
    else:
         print(f"\nSkipping Test 4: Model file for {test_city_2} not found.")


    # --- Test 5: Predict for non-existent model ---
    print("\n--- Test 5: Predicting for 'Atlantis' (No Model) ---")
    forecast_none = generate_forecast("Atlantis", days_ahead=days_to_forecast)
    if forecast_none is None:
        print("Success! Correctly returned None for city with no model.")
    else:
        print("Failure! Expected None for city with no model.")


    # --- Test 6: Format Forecast for UI ---
    print("\n--- Test 6: Formatting Forecast for UI ---")
    if forecast_api is not None: # Use the result from the API corrected test
        ui_formatted_data = format_forecast_for_ui(forecast_api)
        print(f"UI Formatted Data for {test_city} (API Corrected Forecast):")
        import pprint # Use pprint for cleaner dictionary printing
        pprint.pprint(ui_formatted_data)
    else:
        print(f"Skipping UI format test as '{test_city}' forecast was not generated.")

    print("\n" + "="*30)
    print(" predictor.py Tests Finished ")
    print("="*30 + "\n")