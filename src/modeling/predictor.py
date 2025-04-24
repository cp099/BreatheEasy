# File: src/modeling/predictor.py

"""
Handles loading saved Prophet models and generating AQI forecasts.

Provides functions to:
- Load city-specific Prophet models (with caching).
- Generate raw forecasts for a specified number of days ahead.
- Apply a residual correction to the forecast using the latest live AQI data.
- Format the forecast output for easy UI consumption (JSON).
- Interpret the forecast to provide predicted weekly health risks based on AQI levels.

Relies on configuration for model paths/versions and external modules for
API access and AQI category interpretation.
"""

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
# (Keep existing import logic with fallbacks)
try:
    from src.config_loader import CONFIG
    from src.api_integration.client import get_current_aqi_for_city
    from src.health_rules.info import get_aqi_info
    from src.exceptions import ModelFileNotFoundError, ModelLoadError, PredictionError, APIKeyError, APITimeoutError, APINotFoundError, APIError
    # Logging configured by config_loader
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"Predictor: Could not import dependencies. Error: {e}", exc_info=True)
    CONFIG = {}
    class ModelFileNotFoundError(FileNotFoundError): pass
    class ModelLoadError(Exception): pass
    class PredictionError(Exception): pass
    class APIKeyError(Exception): pass
    class APITimeoutError(Exception): pass
    class APINotFoundError(Exception): pass
    class APIError(Exception): pass
    def get_current_aqi_for_city(city_name): logging.error("API client unavailable."); return None
    def get_aqi_info(aqi_value): logging.error("AQI info unavailable."); return None
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
     def get_current_aqi_for_city(city_name): logging.error("API client unavailable."); return None
     def get_aqi_info(aqi_value): logging.error("AQI info unavailable."); return None


# --- Get Logger ---
log = logging.getLogger(__name__)

# --- Configuration Values ---
# (Keep logic for getting config values)
relative_models_dir = CONFIG.get('paths', {}).get('models_dir', 'models')
MODELS_DIR = os.path.join(PROJECT_ROOT, relative_models_dir)
MODEL_VERSION = CONFIG.get('modeling', {}).get('prophet_model_version', 'v2')
DEFAULT_FORECAST_DAYS = CONFIG.get('modeling', {}).get('forecast_days', 5)

# --- Model Loading Cache ---
_loaded_models_cache = {}

def load_prophet_model(city_name, version=MODEL_VERSION, models_dir=MODELS_DIR):
    """Loads a specified trained Prophet model from a JSON file, using an in-memory cache.

    Constructs the model file path based on city name, version, and models directory
    specified in the configuration.

    Args:
        city_name (str): The name of the city for which to load the model.
        version (str, optional): The model version string (suffix).
                                 Defaults to MODEL_VERSION from config.
        models_dir (str, optional): The absolute path to the directory containing model files.
                                   Defaults to MODELS_DIR from config.

    Returns:
        Prophet: The loaded (deserialized) Prophet model instance.

    Raises:
        ModelFileNotFoundError: If the expected model file doesn't exist.
        ModelLoadError: If the model file exists but an error occurs during
                        deserialization (e.g., corrupted file, incompatible version).
    """
    # (Function code remains the same)
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
        raise ModelFileNotFoundError(msg)
    try:
        with open(model_path, 'r') as fin:
            model = model_from_json(json.load(fin))
        log.info(f"Model for {city_name} loaded successfully.")
        _loaded_models_cache[model_key] = model
        return model
    except Exception as e:
        msg = f"Error loading model for {city_name} from {model_path}: {e}"
        log.error(msg, exc_info=True)
        raise ModelLoadError(msg) from e

# --- Prediction Function (Section 4 Core) ---
def generate_forecast(target_city, days_ahead=DEFAULT_FORECAST_DAYS, apply_residual_correction=True, last_known_aqi=None):
    """Generates a multi-day AQI forecast DataFrame for a specific city.

    Loads the appropriate pre-trained Prophet model. Creates future dates
    and generates predictions using the model. Optionally fetches the latest
    live AQI via API (or uses a provided value) to calculate a residual
    (difference between latest actual and model's prediction for that day)
    and applies this residual to the forecast values as an adjustment.

    Args:
        target_city (str): The city for which to generate the forecast.
        days_ahead (int, optional): Number of days into the future to forecast.
                                    Defaults to DEFAULT_FORECAST_DAYS from config.
        apply_residual_correction (bool, optional): If True, attempts to fetch the
                                                   latest AQI and apply correction.
                                                   Defaults to True.
        last_known_aqi (float, optional): If provided, uses this value for residual
                                          correction instead of calling the API. Assumed
                                          to be for the model's last training date.
                                          Defaults to None.

    Returns:
        pandas.DataFrame or None: A DataFrame containing the forecast results with columns:
                                  'ds' (Timestamp): Forecast date.
                                  'yhat' (float): Original Prophet forecast value.
                                  'yhat_lower' (float): Lower uncertainty bound.
                                  'yhat_upper' (float): Upper uncertainty bound.
                                  'residual' (float): The calculated residual used for adjustment (0 if not applied).
                                  'yhat_adjusted' (float): The final forecast value after residual adjustment (clipped >= 0).
                                  Returns None if a critical error occurs (e.g., model not found,
                                  prediction generation fails). API errors during correction are logged
                                  but allow the function to return the unadjusted forecast.
    """
    # (Function code remains the same)
    log.info(f"Generating {days_ahead}-day forecast for {target_city}...")
    try:
        trained_model = load_prophet_model(target_city)
    except (ModelFileNotFoundError, ModelLoadError) as e:
        log.error(f"Cannot generate forecast: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error loading model for {target_city}: {e}", exc_info=True)
        return None

    try:
        future_dates_df = trained_model.make_future_dataframe(periods=days_ahead, include_history=False)
        log.info(f"Predicting for dates: {future_dates_df['ds'].min().date()} to {future_dates_df['ds'].max().date()}")
    except Exception as e:
         log.error(f"Error creating future dataframe for {target_city}: {e}")
         return None
    if future_dates_df.empty:
        log.error("make_future_dataframe returned empty dataframe for {target_city}.")
        return None

    try:
        forecast = trained_model.predict(future_dates_df)
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        log.error(f"Failed to generate initial Prophet forecast for {target_city}: {e}", exc_info=True)
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
                else:
                     forecast['residual'] = 0.0
                     forecast['yhat_adjusted'] = forecast['yhat'].copy()
            else:
                 log.warning(f"Could not generate prediction for {today_ds.date()}. Skipping correction.")
                 forecast['residual'] = 0.0
                 forecast['yhat_adjusted'] = forecast['yhat'].copy()
        except Exception as e:
            log.error(f"Error during residual correction step: {e}. Proceeding without correction.", exc_info=True)
            forecast['residual'] = 0.0
            forecast['yhat_adjusted'] = forecast['yhat'].copy()
    else:
        log.info("Residual correction not applied.")

    log.info(f"Forecast generation complete for {target_city}.")
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'residual', 'yhat_adjusted']]


# --- Helper Function for UI Formatting (Section 4 UI) ---
def format_forecast_for_ui(forecast_df):
    """Formats the forecast DataFrame into a list of dictionaries suitable for UI (JSON).

    Selects the date ('ds') and adjusted forecast ('yhat_adjusted'), formats the
    date as 'YYYY-MM-DD', rounds the AQI to an integer, and returns a list
    of dictionaries.

    Args:
        forecast_df (pd.DataFrame or None): The DataFrame returned by generate_forecast.

    Returns:
        list[dict]: A list where each dict has 'date' (str) and 'predicted_aqi' (int).
                    Returns an empty list if input is None or empty.
    """
    # (Function code remains the same)
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
    """Generates forecast and interprets health implications for each predicted day.

    Calls generate_forecast to get the N-day AQI predictions (using residual
    correction by default). Then, for each day's predicted AQI, it determines
    the corresponding CPCB AQI category and health implication using get_aqi_info.

    Args:
        city_name (str): The city for which to generate predicted risks.
        days_ahead (int, optional): Number of days to forecast. Defaults to
                                    DEFAULT_FORECAST_DAYS from config.

    Returns:
        list[dict]: A list of dictionaries, one for each forecast day, containing:
                    - 'date' (str): Date in 'YYYY-MM-DD' format.
                    - 'predicted_aqi' (int): The predicted AQI value (adjusted).
                    - 'level' (str): The corresponding CPCB AQI category level (e.g., 'Moderate').
                    - 'color' (str): Hex color code associated with the level.
                    - 'implications' (str): Health implications text for the level.
                    Returns an empty list if the forecast cannot be generated.
    """
    # (Function code remains the same)
    log.info(f"Getting predicted weekly risks (Section 6) for {city_name} for {days_ahead} days...")
    try:
        forecast_df = generate_forecast(target_city=city_name, days_ahead=days_ahead, apply_residual_correction=True)
    except Exception as e:
        log.error(f"Unhandled error during forecast generation for weekly risks: {e}", exc_info=True)
        forecast_df = None

    if forecast_df is None or forecast_df.empty:
        log.error(f"Failed to generate forecast for {city_name}. Cannot determine predicted risks.")
        return []

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

# --- Example Usage Block ---
# (Keep existing __main__ block as is)
if __name__ == "__main__":
    # ... (test code remains the same) ...
    pass # Added pass for valid syntax if test code removed/commented