# File: src/modeling/predictor.py

import pandas as pd
import os
import sys
import logging
import json
from prophet import Prophet
from prophet.serialize import model_from_json

# --- Setup Project Root Path ---
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)
     logging.info(f"Predictor: Added project root to sys.path: {PROJECT_ROOT}")

# --- Import necessary functions ---
try:
    from src.api_integration.client import get_current_aqi_for_city # For residual correction
    from src.health_rules.info import get_aqi_info # For interpreting predicted AQI levels
    logging.info("Predictor: Successfully imported dependent modules.")
except ModuleNotFoundError as e:
    logging.error(f"Predictor: Could not import dependent modules. Error: {e}")
    raise
except ImportError as e:
     logging.error(f"Predictor: Error importing dependent modules: {e}")
     raise

# --- Logging Configuration ---
log = logging.getLogger(__name__) # Get logger for this module

# --- Configuration ---
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_VERSION = "v2" # Should match the version used in train.py

# --- Model Loading Cache ---
_loaded_models_cache = {}

def load_prophet_model(city_name, version=MODEL_VERSION, models_dir=MODELS_DIR):
    """Loads a previously trained Prophet model, using cache."""
    global _loaded_models_cache
    model_key = f"{city_name}_{version}"
    if model_key in _loaded_models_cache:
        log.info(f"Returning cached model for {city_name} (v{version}).")
        return _loaded_models_cache[model_key]

    model_filename = f"{city_name}_prophet_model_{version}.json"
    model_path = os.path.join(models_dir, model_filename)
    log.info(f"Loading model for {city_name} (v{version}) from: {model_path}")
    if not os.path.exists(model_path):
        log.error(f"Model file not found: {model_path}")
        return None
    try:
        with open(model_path, 'r') as fin:
            model = model_from_json(json.load(fin))
        log.info(f"Model for {city_name} loaded successfully.")
        _loaded_models_cache[model_key] = model
        return model
    except Exception as e:
        log.error(f"Error loading model for {city_name}: {e}", exc_info=True)
        return None

# --- Prediction Function (Section 4 Core) ---
def generate_forecast(target_city, days_ahead=5, apply_residual_correction=True, last_known_aqi=None):
    """
    Generates AQI forecast DataFrame for the next few days for a given city.
    (Function definition remains exactly the same as the previous correct version)

    Args:
        target_city (str): City name.
        days_ahead (int): How many days to predict.
        apply_residual_correction (bool): Use API/last value for correction.
        last_known_aqi (float, optional): Manual override for last known value.

    Returns:
        pd.DataFrame: Forecast results or None on failure.
    """
    log.info(f"Generating {days_ahead}-day forecast for {target_city}...")
    trained_model = load_prophet_model(target_city)
    if trained_model is None:
        log.error(f"Failed to load model for {target_city}.")
        return None
    try:
        future_dates_df = trained_model.make_future_dataframe(periods=days_ahead, include_history=False)
        log.info(f"Predicting for dates: {future_dates_df['ds'].min().date()} to {future_dates_df['ds'].max().date()}")
    except Exception as e:
         log.error(f"Error creating future dataframe: {e}")
         return None
    if future_dates_df.empty:
        log.error("make_future_dataframe returned empty dataframe.")
        return None
    try:
        forecast = trained_model.predict(future_dates_df)
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        log.error(f"Failed to generate initial forecast for {target_city}: {e}")
        return None

    forecast['yhat_adjusted'] = forecast['yhat'].copy()
    forecast['residual'] = 0.0

    actual_today = None
    if apply_residual_correction:
        log.info(f"Attempting residual correction for {target_city}...")
        try:
            today_ds = trained_model.history_dates.max()
            log.info(f"Using model's last training date as 'today' for correction: {today_ds.date()}")
            today_df = pd.DataFrame({'ds': [today_ds]})
            today_forecast = trained_model.predict(today_df)

            if not today_forecast.empty:
                predicted_today = today_forecast['yhat'].iloc[0]
                log.info(f"Model's prediction for {today_ds.date()}: {predicted_today:.2f}")
                if last_known_aqi is not None:
                     actual_today = float(last_known_aqi)
                     log.info(f"Using provided last_known_aqi for {today_ds.date()}: {actual_today}")
                else:
                    log.info(f"Fetching current AQI from API for {target_city}...")
                    current_aqi_info = get_current_aqi_for_city(target_city) # From client.py
                    if current_aqi_info and 'aqi' in current_aqi_info:
                        actual_today = float(current_aqi_info['aqi'])
                        log.info(f"Actual current AQI from API: {actual_today}")
                    else:
                         log.warning(f"Could not get current AQI from API for {target_city}. Skipping residual correction.")

                if actual_today is not None:
                    residual = actual_today - predicted_today
                    forecast['residual'] = residual
                    log.info(f"Calculated residual: {residual:.2f}")
                    forecast['yhat_adjusted'] = (forecast['yhat'] + residual).clip(lower=0)
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
    log.info(f"Forecast generation complete for {target_city}.")
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'residual', 'yhat_adjusted']]

# --- Helper Function for UI Formatting (Section 4 UI) ---
def format_forecast_for_ui(forecast_df):
    """
    Formats the forecast DataFrame into a list of dictionaries suitable for UI (JSON).
    (Function definition remains exactly the same as the previous correct version)

    Args:
        forecast_df (pd.DataFrame): The DataFrame returned by generate_forecast.

    Returns:
        list: List of dicts [{'date': 'YYYY-MM-DD', 'predicted_aqi': INT}, ...]
    """
    if forecast_df is None or forecast_df.empty:
        return []
    ui_data = []
    for index, row in forecast_df.iterrows():
        try:
             date_str = pd.to_datetime(row['ds']).strftime('%Y-%m-%d') if pd.notna(row['ds']) else None
             predicted_aqi = int(round(row['yhat_adjusted'])) if pd.notna(row['yhat_adjusted']) else None
             if date_str is not None and predicted_aqi is not None:
                 ui_data.append({'date': date_str, 'predicted_aqi': predicted_aqi})
             else:
                  log.warning(f"Skipping row due to missing date or predicted_aqi: {row}")
        except Exception as e:
             log.error(f"Error formatting row {index} for UI: {row}. Error: {e}")
             continue
    return ui_data

# --- Function for Section 6 (Predicted Weekly Risks) - MOVED HERE ---
def get_predicted_weekly_risks(city_name, days_ahead=5):
    """
    Generates the forecast for a city and interprets the health implications
    for each predicted day based on CPCB AQI categories.

    Args:
        city_name (str): The name of the city (e.g., 'Delhi').
        days_ahead (int): Number of days to forecast (default 5).

    Returns:
        list: A list of dictionaries, where each dictionary represents a day
              in the forecast and contains 'date', 'predicted_aqi', 'level',
              'color', and 'implications'. Returns an empty list on failure.
    """
    log.info(f"Getting predicted weekly risks (Section 6) for {city_name}...")

    # 1. Generate the forecast (use the function defined above)
    forecast_df = generate_forecast(
        target_city=city_name,
        days_ahead=days_ahead,
        apply_residual_correction=True # Use adjusted forecast for risks
    )

    if forecast_df is None or forecast_df.empty:
        log.error(f"Failed to generate forecast for {city_name}. Cannot determine predicted risks.")
        return []

    predicted_risks_list = []
    log.info(f"Interpreting health implications for {len(forecast_df)} forecasted days...")

    # 2. Iterate through forecast and interpret using get_aqi_info
    for index, row in forecast_df.iterrows():
        try:
            predicted_aqi = row['yhat_adjusted'] # Use adjusted value
            forecast_date = pd.to_datetime(row['ds']).strftime('%Y-%m-%d')

            if pd.isna(predicted_aqi):
                 log.warning(f"Skipping date {forecast_date} for risk interpretation due to missing predicted AQI.")
                 continue

            aqi_value_int = int(round(predicted_aqi))

            # 3. Get AQI category info (using function imported from info.py)
            aqi_category_info = get_aqi_info(aqi_value_int)

            if aqi_category_info:
                predicted_risks_list.append({
                    "date": forecast_date,
                    "predicted_aqi": aqi_value_int,
                    "level": aqi_category_info.get("level", "N/A"),
                    "color": aqi_category_info.get("color", "#FFFFFF"),
                    "implications": aqi_category_info.get("implications", "N/A")
                })
            else:
                log.warning(f"Could not get AQI category info for predicted AQI {aqi_value_int} on {forecast_date}.")
                # Append with placeholder info
                predicted_risks_list.append({
                    "date": forecast_date, "predicted_aqi": aqi_value_int, "level": "Unknown",
                    "color": "#808080", "implications": "Category undefined for value."
                })
        except Exception as e:
             log.error(f"Error processing forecast row {index} for weekly risks: {row}. Error: {e}", exc_info=True)
             continue

    log.info(f"Finished interpreting predicted weekly risks for {city_name}.")
    return predicted_risks_list

# --- Example Usage Block (for testing this module directly) ---
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s')

    print("\n" + "="*30)
    print(" Running predictor.py Tests ")
    print("="*30 + "\n")

    test_city = "Delhi"
    days_to_forecast = 5

    # --- Test 1: Predict WITH API correction ---
    print(f"\n--- Test 1: Predicting {days_to_forecast} days for '{test_city}' WITH API Correction ---")
    forecast_api = generate_forecast(test_city, days_ahead=days_to_forecast, apply_residual_correction=True)
    if forecast_api is not None: print(forecast_api)
    else: print(f"Failed test 1 for {test_city}")

    # --- Test 2: Predict WITH manual correction ---
    simulated_aqi = 150.0
    print(f"\n--- Test 2: Predicting {days_to_forecast} days for '{test_city}' WITH Manual Correction (Last AQI={simulated_aqi}) ---")
    forecast_manual = generate_forecast(test_city, days_ahead=days_to_forecast, apply_residual_correction=True, last_known_aqi=simulated_aqi)
    if forecast_manual is not None: print(forecast_manual)
    else: print(f"Failed test 2 for {test_city}")

    # --- Test 3: Predict WITHOUT correction ---
    print(f"\n--- Test 3: Predicting {days_to_forecast} days for '{test_city}' WITHOUT Correction ---")
    forecast_raw = generate_forecast(test_city, days_ahead=days_to_forecast, apply_residual_correction=False)
    if forecast_raw is not None: print(forecast_raw)
    else: print(f"Failed test 3 for {test_city}")

    # --- Test 4: Format Forecast for UI ---
    print("\n--- Test 4: Formatting Forecast for UI ---")
    if forecast_api is not None:
        ui_formatted_data = format_forecast_for_ui(forecast_api)
        print(f"UI Formatted Data for {test_city} (API Corrected Forecast):")
        import pprint
        pprint.pprint(ui_formatted_data)
    else: print("Skipping test 4.")

    # --- Test 5: Get Predicted Weekly Risks (Section 6 logic) ---
    print("\n--- Test 5: Getting Predicted Weekly Risks (Section 6) ---")
    print(f"--- Getting predicted weekly risks for {test_city} ---")
    weekly_risks = get_predicted_weekly_risks(test_city, days_ahead=days_to_forecast) # Now calling func in this file
    if weekly_risks:
        import pprint
        print("Predicted Weekly Risks/Implications:")
        pprint.pprint(weekly_risks)
    else:
        print(f"Could not generate predicted weekly risks for {test_city}.")

    # --- Test 6: Predict for another city (e.g., Mumbai) ---
    test_city_2 = "Mumbai"
    if os.path.exists(os.path.join(MODELS_DIR, f"{test_city_2}_prophet_model_{MODEL_VERSION}.json")):
         print(f"\n--- Test 6: Predicting {days_to_forecast} days for '{test_city_2}' WITH API Correction ---")
         forecast_mumbai = generate_forecast(test_city_2, days_ahead=days_to_forecast, apply_residual_correction=True)
         if forecast_mumbai is not None:
             print(f"Forecast for {test_city_2} (API Corrected):")
             print(forecast_mumbai)
         else:
             print(f"Failed test 6 for {test_city_2}.")
    else:
         print(f"\nSkipping Test 6: Model file for {test_city_2} not found.")

    # --- Test 7: Predict for non-existent model ---
    print("\n--- Test 7: Predicting for 'Atlantis' (No Model) ---")
    forecast_none = generate_forecast("Atlantis", days_ahead=days_to_forecast)
    if forecast_none is None: print("Success! Correctly returned None.")
    else: print("Failure! Expected None.")


    print("\n" + "="*30)
    print(" predictor.py Tests Finished ")
    print("="*30 + "\n")