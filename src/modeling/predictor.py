# File: src/modeling/predictor.py

import pandas as pd
import os
import sys
import logging # Standard logging import
import json
from prophet import Prophet
from prophet.serialize import model_from_json

# --- Setup Project Root Path ---
# (Keep existing PROJECT_ROOT logic)
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')):
         PROJECT_ROOT = os.path.abspath('.')
    # Cannot log reliably here yet
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import Configuration and Other Modules ---
# This import also sets up logging via config_loader.py
try:
    from src.config_loader import CONFIG
    # Import dependent functions *after* path setup and config load attempt
    from src.api_integration.client import get_current_aqi_for_city
    from src.health_rules.info import get_aqi_info
    # Import custom exceptions
    from src.exceptions import ModelFileNotFoundError, ModelLoadError, PredictionError, APIKeyError, APITimeoutError, APINotFoundError, APIError
    # Logging is configured by config_loader
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"Predictor: Could not import dependencies. Error: {e}", exc_info=True)
    CONFIG = {}
    # Define dummy exceptions/functions
    class ModelFileNotFoundError(FileNotFoundError): pass
    class ModelLoadError(Exception): pass
    class PredictionError(Exception): pass
    class APIKeyError(Exception): pass
    class APITimeoutError(Exception): pass
    class APINotFoundError(Exception): pass
    class APIError(Exception): pass
    def get_current_aqi_for_city(city_name): log.error("API client unavailable."); return None
    def get_aqi_info(aqi_value): log.error("AQI info unavailable."); return None
except Exception as e:
     logging.basicConfig(level=logging.WARNING)
     logging.error(f"Predictor: Error importing dependencies: {e}", exc_info=True)
     CONFIG = {}
     class ModelFileNotFoundError(FileNotFoundError): pass
     class ModelLoadError(Exception): pass
     class PredictionError(Exception): pass
     class APIKeyError(Exception): pass
     class APITimeoutError(Exception): pass
     class APINotFoundError(Exception): pass
     class APIError(Exception): pass
     def get_current_aqi_for_city(city_name): log.error("API client unavailable."); return None
     def get_aqi_info(aqi_value): log.error("AQI info unavailable."); return None


# --- Get Logger ---
log = logging.getLogger(__name__)

# --- Configuration Values ---
# (Keep logic for getting config values: MODELS_DIR, MODEL_VERSION, DEFAULT_FORECAST_DAYS)
relative_models_dir = CONFIG.get('paths', {}).get('models_dir', 'models')
MODELS_DIR = os.path.join(PROJECT_ROOT, relative_models_dir)
MODEL_VERSION = CONFIG.get('modeling', {}).get('prophet_model_version', 'v2')
DEFAULT_FORECAST_DAYS = CONFIG.get('modeling', {}).get('forecast_days', 5)

# --- Model Loading Cache ---
_loaded_models_cache = {}

def load_prophet_model(city_name, version=MODEL_VERSION, models_dir=MODELS_DIR):
    """
    Loads a previously trained Prophet model, using cache.

    Raises:
        ModelFileNotFoundError: If the model file doesn't exist.
        ModelLoadError: If the model file exists but cannot be loaded/deserialized.
    """
    global _loaded_models_cache
    model_key = f"{city_name}_{version}"
    if model_key in _loaded_models_cache:
        log.info(f"Returning cached model for {city_name} (v{version}).")
        return _loaded_models_cache[model_key]

    model_filename = f"{city_name}_prophet_model_{version}.json"
    model_path = os.path.join(models_dir, model_filename)
    log.info(f"Loading model for {city_name} (v{version}) from: {model_path}")
    if not os.path.exists(model_path):
        msg = f"Model file not found: {model_path}"
        log.error(msg)
        raise ModelFileNotFoundError(msg) # Raise specific error

    try:
        with open(model_path, 'r') as fin:
            model = model_from_json(json.load(fin))
        log.info(f"Model for {city_name} loaded successfully.")
        _loaded_models_cache[model_key] = model
        return model
    except Exception as e:
        msg = f"Error loading model for {city_name} from {model_path}: {e}"
        log.error(msg, exc_info=True)
        raise ModelLoadError(msg) from e # Raise specific error


# --- Prediction Function (Section 4 Core) ---
def generate_forecast(target_city, days_ahead=DEFAULT_FORECAST_DAYS, apply_residual_correction=True, last_known_aqi=None):
    """
    Generates AQI forecast DataFrame. Handles model load and prediction errors.

    Returns:
        pd.DataFrame or None: Forecast results or None if a critical error occurs
                              (like model not found or prediction failure).
                              Note: API errors during correction will log warnings
                              but still return the uncorrected forecast.
    """
    log.info(f"Generating {days_ahead}-day forecast for {target_city}...")
    try:
        trained_model = load_prophet_model(target_city) # Can raise ModelFileNotFoundError, ModelLoadError
    except (ModelFileNotFoundError, ModelLoadError) as e:
        log.error(f"Cannot generate forecast: {e}")
        return None # Return None if model cannot be loaded
    except Exception as e: # Catch unexpected loading errors
        log.error(f"Unexpected error loading model for {target_city}: {e}", exc_info=True)
        return None

    try:
        future_dates_df = trained_model.make_future_dataframe(periods=days_ahead, include_history=False)
        log.info(f"Predicting for dates: {future_dates_df['ds'].min().date()} to {future_dates_df['ds'].max().date()}")
    except Exception as e:
         log.error(f"Error creating future dataframe for {target_city}: {e}")
         # Optional: raise PredictionError(f"Failed to create future dates: {e}") from e
         return None # Return None if dates can't be generated
    if future_dates_df.empty:
        log.error("make_future_dataframe returned empty dataframe for {target_city}.")
        return None

    # --- Placeholder for adding FUTURE regressors ---
    # ...
    # --- End Placeholder ---

    try:
        forecast = trained_model.predict(future_dates_df)
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        log.error(f"Failed to generate initial Prophet forecast for {target_city}: {e}", exc_info=True)
        # Optional: raise PredictionError(f"Prophet predict failed: {e}") from e
        return None # Return None if prediction fails

    # Initialize adjusted columns
    forecast['yhat_adjusted'] = forecast['yhat'].copy()
    forecast['residual'] = 0.0
    actual_today = None

    # Apply Residual Correction (best effort - logs errors but doesn't fail forecast)
    if apply_residual_correction:
        log.info(f"Attempting residual correction for {target_city}...")
        try:
            today_ds = trained_model.history_dates.max()
            log.info(f"Using model's last training date as 'today' for correction: {today_ds.date()}")
            today_df = pd.DataFrame({'ds': [today_ds]})
            # Add regressors to today_df if needed
            today_forecast = trained_model.predict(today_df)

            if not today_forecast.empty:
                predicted_today = today_forecast['yhat'].iloc[0]
                log.info(f"Model's prediction for {today_ds.date()}: {predicted_today:.2f}")
                if last_known_aqi is not None:
                     actual_today = float(last_known_aqi)
                     log.info(f"Using provided last_known_aqi for {today_ds.date()}: {actual_today}")
                else:
                    log.info(f"Fetching current AQI from API for {target_city}...")
                    # get_current_aqi_for_city handles its own API errors and returns None
                    current_aqi_info = get_current_aqi_for_city(target_city)
                    if current_aqi_info and 'aqi' in current_aqi_info:
                        actual_today = float(current_aqi_info['aqi'])
                        log.info(f"Actual current AQI from API: {actual_today}")
                    else:
                         log.warning(f"Could not get current AQI from API for {target_city}. Skipping correction.")

                if actual_today is not None:
                    residual = actual_today - predicted_today
                    forecast['residual'] = residual
                    log.info(f"Calculated residual: {residual:.2f}")
                    forecast['yhat_adjusted'] = (forecast['yhat'] + residual).clip(lower=0)
                    log.info(f"Applied residual correction.")
                else: # Reset if actual couldn't be obtained
                     forecast['residual'] = 0.0
                     forecast['yhat_adjusted'] = forecast['yhat'].copy()
            else:
                 log.warning(f"Could not generate prediction for {today_ds.date()}. Skipping correction.")
                 forecast['residual'] = 0.0
                 forecast['yhat_adjusted'] = forecast['yhat'].copy()
        except Exception as e: # Catch errors during the correction process itself
            log.error(f"Error during residual correction step: {e}. Proceeding without correction.", exc_info=True)
            forecast['residual'] = 0.0
            forecast['yhat_adjusted'] = forecast['yhat'].copy()
    else:
        log.info("Residual correction not applied.")

    log.info(f"Forecast generation complete for {target_city}.")
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'residual', 'yhat_adjusted']]


# --- Helper Function for UI Formatting (Section 4 UI) ---
def format_forecast_for_ui(forecast_df):
    """Formats forecast DataFrame into list of dicts for UI."""
    # (No changes needed in this function's logic)
    if forecast_df is None or forecast_df.empty: return []
    ui_data = []
    for index, row in forecast_df.iterrows():
        try:
             date_str = pd.to_datetime(row['ds']).strftime('%Y-%m-%d') if pd.notna(row['ds']) else None
             predicted_aqi = int(round(row['yhat_adjusted'])) if pd.notna(row['yhat_adjusted']) else None
             if date_str is not None and predicted_aqi is not None:
                 ui_data.append({'date': date_str, 'predicted_aqi': predicted_aqi})
             else: log.warning(f"Skipping row due to missing data: {row}")
        except Exception as e:
             log.error(f"Error formatting row {index} for UI: {row}. Error: {e}")
             continue
    return ui_data


# --- Function for Section 6 (Predicted Weekly Risks) ---
def get_predicted_weekly_risks(city_name, days_ahead=DEFAULT_FORECAST_DAYS):
    """Generates forecast and interprets health implications. Handles forecast errors."""
    log.info(f"Getting predicted weekly risks (Section 6) for {city_name} for {days_ahead} days...")
    try:
        # Call generate_forecast which might return None
        forecast_df = generate_forecast(
            target_city=city_name,
            days_ahead=days_ahead,
            apply_residual_correction=True
        )
    except Exception as e: # Catch unexpected errors from generate_forecast itself
        log.error(f"Unhandled error during forecast generation for weekly risks: {e}", exc_info=True)
        forecast_df = None

    if forecast_df is None or forecast_df.empty:
        log.error(f"Failed to generate forecast for {city_name}. Cannot determine predicted risks.")
        return [] # Return empty list on failure

    # (Keep the rest of the interpretation logic as is)
    predicted_risks_list = []
    log.info(f"Interpreting health implications for {len(forecast_df)} forecasted days...")
    for index, row in forecast_df.iterrows():
        try:
            predicted_aqi = row['yhat_adjusted']
            forecast_date = pd.to_datetime(row['ds']).strftime('%Y-%m-%d')
            if pd.isna(predicted_aqi):
                 log.warning(f"Skipping date {forecast_date} for risk interp: missing predicted AQI.")
                 continue
            aqi_value_int = int(round(predicted_aqi))
            aqi_category_info = get_aqi_info(aqi_value_int)
            if aqi_category_info:
                predicted_risks_list.append({
                    "date": forecast_date, "predicted_aqi": aqi_value_int,
                    "level": aqi_category_info.get("level", "N/A"),
                    "color": aqi_category_info.get("color", "#FFFFFF"),
                    "implications": aqi_category_info.get("implications", "N/A")})
            else:
                log.warning(f"Could not get AQI category info for predicted AQI {aqi_value_int} on {forecast_date}.")
                predicted_risks_list.append({"date": forecast_date, "predicted_aqi": aqi_value_int,
                                            "level": "Unknown", "color": "#808080",
                                            "implications": "Category undefined."})
        except Exception as e:
             log.error(f"Error processing row {index} for weekly risks: {row}. Error: {e}", exc_info=True)
             continue
    log.info(f"Finished interpreting predicted weekly risks for {city_name}.")
    return predicted_risks_list

# --- Example Usage Block (Adjusted to show error handling) ---
if __name__ == "__main__":
    # Logging configured by importing CONFIG above
    print("\n" + "="*30)
    print(" Running predictor.py Tests ")
    print("="*30 + "\n")

    test_city = "Delhi"
    days_to_forecast = DEFAULT_FORECAST_DAYS

    # Wrap calls in try/except to demonstrate handling
    try:
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
        if forecast_api is not None: # Use result from test 1
            ui_formatted_data = format_forecast_for_ui(forecast_api)
            print(f"UI Formatted Data for {test_city} (API Corrected Forecast):")
            import pprint
            pprint.pprint(ui_formatted_data)
        else: print("Skipping test 4.")

        # --- Test 5: Get Predicted Weekly Risks (Section 6 logic) ---
        print("\n--- Test 5: Getting Predicted Weekly Risks (Section 6) ---")
        print(f"--- Getting predicted weekly risks for {test_city} ---")
        weekly_risks = get_predicted_weekly_risks(test_city, days_ahead=days_to_forecast)
        if weekly_risks:
            import pprint
            print("Predicted Weekly Risks/Implications:")
            pprint.pprint(weekly_risks)
        else: print(f"Could not generate predicted weekly risks for {test_city}.")

        # --- Test 6: Predict for another city (e.g., Mumbai) ---
        test_city_2 = "Mumbai"
        # Check existence using configured paths/version
        model_exists = os.path.exists(os.path.join(MODELS_DIR, f"{test_city_2}_prophet_model_{MODEL_VERSION}.json"))
        if model_exists:
             print(f"\n--- Test 6: Predicting {days_to_forecast} days for '{test_city_2}' WITH API Correction ---")
             forecast_mumbai = generate_forecast(test_city_2, days_ahead=days_to_forecast, apply_residual_correction=True)
             if forecast_mumbai is not None: print(f"Forecast for {test_city_2}:\n{forecast_mumbai}")
             else: print(f"Failed test 6 for {test_city_2}.")
        else: print(f"\nSkipping Test 6: Model file for {test_city_2} not found.")

        # --- Test 7: Predict for non-existent model ---
        print("\n--- Test 7: Predicting for 'Atlantis' (No Model) ---")
        forecast_none = generate_forecast("Atlantis", days_ahead=days_to_forecast)
        if forecast_none is None: print("Success! Correctly returned None.")
        else: print("Failure! Expected None.")

    except Exception as e: # Catch unexpected errors during the test runs
         print(f"\n!!! An error occurred during the main test execution block: {e} !!!")
         log.error("Error in predictor.py __main__ block", exc_info=True)


    print("\n" + "="*30)
    print(" predictor.py Tests Finished ")
    print("="*30 + "\n")