# File: src/modeling/predictor.py

"""
Handles loading saved Prophet models and generating AQI forecasts.

Provides functions to:
- Load city-specific Prophet models (with caching).
- Generate raw forecasts for a specified number of days ahead.
- Apply a residual correction (with decay) to the forecast using the latest live AQI data.
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
relative_models_dir = CONFIG.get('paths', {}).get('models_dir', 'models')
MODELS_DIR = os.path.join(PROJECT_ROOT, relative_models_dir)
MODEL_VERSION = CONFIG.get('modeling', {}).get('prophet_model_version', 'v2')
DEFAULT_FORECAST_DAYS = CONFIG.get('modeling', {}).get('forecast_days', 5)
# --- NEW: Configuration for Decay Factor ---
RESIDUAL_DECAY_FACTOR = CONFIG.get('modeling', {}).get('residual_decay_factor', 0.85) # Example default

# --- Model Loading Cache ---
_loaded_models_cache = {}

def load_prophet_model(city_name, version=MODEL_VERSION, models_dir=MODELS_DIR):
    """Loads a specified trained Prophet model from a JSON file, using an in-memory cache."""
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

# --- Prediction Function (Section 4 Core - CORRECTED WITH DECAY) ---
def generate_forecast(target_city, days_ahead=DEFAULT_FORECAST_DAYS, apply_residual_correction=True, last_known_aqi=None):
    """Generates an AQI forecast DataFrame for the actual next N days from today,
    optionally adjusting based on the latest actual AQI using decay."""
    # (Keep initial part of function: load model, determine dates, initial forecast)
    log.info(f"Generating {days_ahead}-day forecast for {target_city} starting from tomorrow...")
    try:
        trained_model = load_prophet_model(target_city)
    except (ModelFileNotFoundError, ModelLoadError) as e:
        log.error(f"Cannot generate forecast: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error loading model for {target_city}: {e}", exc_info=True)
        return None

    try:
        last_train_date = trained_model.history_dates.max()
        today_actual_date = pd.Timestamp.now().normalize()
        log.info(f"Reference 'today' for forecast generation: {today_actual_date.date()}")
        if today_actual_date < last_train_date:
             log.warning(f"Current date {today_actual_date.date()} is before model's last training date {last_train_date.date()}. Using last train date as reference 'today'.")
             today_ds_ref = last_train_date # Use last train date for prediction lookup
        else:
             today_ds_ref = today_actual_date # Use actual today for prediction lookup
             offset_days = (today_actual_date - last_train_date).days

        total_periods_to_predict = (today_ds_ref - last_train_date).days + days_ahead
        log.info(f"Days between last train date and reference date: {(today_ds_ref - last_train_date).days}. Total periods to predict: {total_periods_to_predict}")

        if total_periods_to_predict <= 0:
            log.error(f"Total periods to predict is not positive ({total_periods_to_predict}). Cannot forecast.")
            return None

        future_dates_df_full = trained_model.make_future_dataframe(periods=total_periods_to_predict, include_history=False)
        future_dates_df_full = future_dates_df_full[future_dates_df_full['ds'] > last_train_date]
        if future_dates_df_full.empty:
             log.error("Failed to generate any future dates past training data.")
             return None

        log.info("Generating initial long-range forecast...")
        full_forecast = trained_model.predict(future_dates_df_full)

        start_forecast_date = today_actual_date + pd.Timedelta(days=1)
        end_forecast_date = today_actual_date + pd.Timedelta(days=days_ahead)
        log.info(f"Extracting target forecast slice from {start_forecast_date.date()} to {end_forecast_date.date()}")
        target_forecast_mask = (full_forecast['ds'] >= start_forecast_date) & (full_forecast['ds'] <= end_forecast_date)
        forecast = full_forecast.loc[target_forecast_mask, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

        if forecast.empty:
             log.warning(f"No forecast data generated for the target date range {start_forecast_date.date()} to {end_forecast_date.date()}.")
             return None

    except Exception as e:
        log.error(f"Failed during forecast generation or slicing for {target_city}: {e}", exc_info=True)
        return None

    # Apply Residual Correction (best effort)
    forecast['yhat_adjusted'] = forecast['yhat'].copy()
    forecast['residual'] = 0.0
    actual_today = None

    if apply_residual_correction:
        log.info(f"Attempting residual correction based on reference 'today': {today_ds_ref.date()}")
        try:
            # Find model's prediction for the reference date (today or last train date)
            predicted_today_row = full_forecast[full_forecast['ds'] == today_ds_ref]
            if not predicted_today_row.empty:
                predicted_today = predicted_today_row['yhat'].iloc[0]
                log.info(f"Model's prediction for reference date {today_ds_ref.date()}: {predicted_today:.2f}")

                # Get actual value (API or provided)
                if last_known_aqi is not None:
                     actual_today = float(last_known_aqi)
                     log.info(f"Using provided last_known_aqi for {today_ds_ref.date()}: {actual_today}")
                else:
                    log.info(f"Fetching current AQI from API for {target_city}...")
                    current_aqi_info = get_current_aqi_for_city(target_city) # Use original city name for API
                    if current_aqi_info and 'aqi' in current_aqi_info:
                        actual_today = float(current_aqi_info['aqi'])
                        log.info(f"Actual current AQI from API (proxy for today): {actual_today}")
                    else:
                         log.warning(f"Could not get current AQI from API. Skipping correction.")

                # --- Apply DECAYING residual correction IF actual value found ---
                if actual_today is not None:
                    base_residual = actual_today - predicted_today
                    log.info(f"Calculated base residual: {base_residual:.2f}")

                    applied_residuals = []
                    current_decay = 1.0
                    for i in range(len(forecast)):
                         correction = base_residual * current_decay
                         applied_residuals.append(correction)
                         current_decay *= RESIDUAL_DECAY_FACTOR # Apply decay

                    # Add the calculated decaying residual to yhat
                    # Ensure index alignment if forecast DataFrame was sliced
                    if len(applied_residuals) == len(forecast):
                         forecast['residual'] = applied_residuals
                         forecast['yhat_adjusted'] = (forecast['yhat'] + forecast['residual']).clip(lower=0)
                         log.info(f"Applied decaying residual correction (factor={RESIDUAL_DECAY_FACTOR}).")
                    else:
                         log.warning(f"Length mismatch between forecast ({len(forecast)}) and residuals ({len(applied_residuals)}). Skipping correction.")
                         forecast['residual'] = 0.0
                         forecast['yhat_adjusted'] = forecast['yhat'].copy()
                # --- End of decaying residual logic ---

            else:
                 log.warning(f"Could not find prediction for ref date {today_ds_ref.date()}. Skipping correction.")
                 # Ensure reset if prediction wasn't found
                 forecast['residual'] = 0.0
                 forecast['yhat_adjusted'] = forecast['yhat'].copy()

        except Exception as e:
            log.error(f"Error during residual correction: {e}. Proceeding without.", exc_info=True)
            forecast['residual'] = 0.0
            forecast['yhat_adjusted'] = forecast['yhat'].copy()
    else:
        log.info("Residual correction not applied as per request.")

    log.info(f"Forecast generation complete for {target_city} for target dates.")
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'residual', 'yhat_adjusted']]

# --- Helper Function for UI Formatting ---
def format_forecast_for_ui(forecast_df):
    """Formats forecast DataFrame into list of dicts for UI."""
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

# --- Function for Section 6 ---
# (Keep get_predicted_weekly_risks exactly as it was - it calls the updated generate_forecast)
def get_predicted_weekly_risks(city_name, days_ahead=DEFAULT_FORECAST_DAYS):
    """Generates forecast and interprets health implications for each predicted day."""
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
    pass