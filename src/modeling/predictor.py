# File: src/modeling/predictor.py
"""
Handles loading saved Prophet models and generating AQI forecasts.

This module is the core of the prediction engine. Its primary responsibilities are:
- Loading city-specific Prophet models and their associated metadata (e.g., historical residual).
- Fetching multi-day weather forecasts to use as model regressors.
- Generating a base AQI forecast using the model and weather data.
- Applying a historical residual correction to refine the forecast.
- Providing helper functions to format the forecast for UI consumption and to
  interpret the forecast into human-readable weekly health risks.
"""

import pandas as pd
import os
import sys
import logging 
import json
import numpy as np
from prophet import Prophet
from prophet.serialize import model_from_json
import math

# --- Setup Project Root Path ---
# This allows the script to be run from anywhere and still find the project root.
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    # Fallback for environments where __file__ is not defined.
    PROJECT_ROOT = os.path.abspath('.')
    if 'src' not in os.listdir(PROJECT_ROOT):
        alt_root = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
        if 'src' in os.listdir(alt_root): PROJECT_ROOT = alt_root
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

# --- Import Project Modules & Dependencies ---
# The config_loader import also triggers the centralized logging setup.
try:
    from src.config_loader import CONFIG
    from src.api_integration.client import get_current_aqi_for_city 
    from src.api_integration.weather_client import get_weather_forecast
    from src.health_rules.info import get_aqi_info
    from src.exceptions import ModelFileNotFoundError, ModelLoadError, PredictionError, APIError, APITimeoutError, ConfigError 
except ImportError as e:
    # Fallback if dependencies or project structure are not found.
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"Predictor: Could not import dependencies. Error: {e}", exc_info=True)
    CONFIG = {}
    class ModelFileNotFoundError(FileNotFoundError): pass
    class ModelLoadError(Exception): pass
    class PredictionError(Exception): pass
    class APIError(Exception): pass 
    class APITimeoutError(Exception): pass 
    class ConfigError(Exception): pass
    def get_weather_forecast(city_name, days): logging.error("Weather forecast client unavailable."); return None
    def get_current_aqi_for_city(city_name): logging.error("AQI client unavailable."); return None
    def get_aqi_info(aqi_value): logging.error("AQI info unavailable."); return None
except Exception as e:
     # Broad exception for other potential import errors.
     logging.basicConfig(level=logging.WARNING)
     logging.error(f"Predictor: Error importing dependencies: {e}", exc_info=True)
     CONFIG = {}
     class ModelFileNotFoundError(FileNotFoundError): pass
     class ModelLoadError(Exception): pass
     class PredictionError(Exception): pass
     class APIError(Exception): pass
     class APITimeoutError(Exception): pass
     class ConfigError(Exception): pass
     def get_weather_forecast(city_name, days): logging.error("Weather forecast client unavailable."); return None
     def get_aqi_info(aqi_value): logging.error("AQI info unavailable."); return None



log = logging.getLogger(__name__)

# --- Module Configuration ---
relative_models_dir = CONFIG.get('paths', {}).get('models_dir', 'models')
MODELS_DIR = os.path.join(PROJECT_ROOT, relative_models_dir)
MODEL_VERSION = CONFIG.get('modeling', {}).get('prophet_model_version', 'v3_weather')
DEFAULT_FORECAST_DAYS = CONFIG.get('modeling', {}).get('forecast_days', 3)
WEATHER_REGRESSORS_CONFIG = CONFIG.get('modeling', {}).get('weather_regressors', [])
RESIDUAL_DECAY_FACTOR = CONFIG.get('modeling', {}).get('residual_decay_factor', 0.85)
REALISTIC_MIN_AQI = CONFIG.get('modeling', {}).get('min_realistic_aqi', 1)
MAX_RESIDUAL_CAP = CONFIG.get('modeling', {}).get('max_residual_cap', 75)
LIVE_RESIDUAL_WEIGHT = CONFIG.get('modeling', {}).get('live_residual_weight', 0.6) # 60% weight to live, 40% to historical

# In-memory cache for loaded models and their metadata.
_loaded_models_cache = {}
_loaded_metadata_cache = {}

def load_prophet_model_and_metadata(city_name, version=MODEL_VERSION, models_dir=MODELS_DIR):
    """
    Loads a trained Prophet model and its associated metadata, using an in-memory cache.

    Args:
        city_name (str): The city for which to load the model.
        version (str): The model version suffix.
        models_dir (str): The directory containing the model files.

    Returns:
        tuple[Prophet, dict]: A tuple containing the loaded Prophet model instance
                              and the metadata dictionary (e.g., {'last_historical_residual': -5.2}).

    Raises:
        ModelFileNotFoundError: If the required model or metadata file doesn't exist.
        ModelLoadError: If a file exists but cannot be loaded or deserialized.
    """
    global _loaded_models_cache, _loaded_metadata_cache

    model_key = f"{city_name}_{version}"
    metadata_key = f"{city_name}_{version}_meta"

   
    cached_model = _loaded_models_cache.get(model_key)
    cached_metadata = _loaded_metadata_cache.get(metadata_key)
    if cached_model is not None and cached_metadata is not None:
        log.info(f"Returning cached model and metadata for {city_name} (v{version}).")
        return cached_model, cached_metadata

    # --- Load Model from JSON file ---
    model_filename = f"{city_name}_prophet_model_{version}.json"
    model_path = os.path.join(models_dir, model_filename)
    log.info(f"Loading model for {city_name} (v{version}) from: {model_path}")
    if not os.path.exists(model_path):
        msg = f"Model file not found: {model_path}"
        log.error(msg); raise ModelFileNotFoundError(msg)
    try:
        with open(model_path, 'r') as fin:
            model = model_from_json(json.load(fin))
        log.info(f"Model for {city_name} loaded successfully.")
        
        loaded_regressors = set(model.extra_regressors.keys())
        expected_regressors = set(WEATHER_REGRESSORS_CONFIG)
        if loaded_regressors != expected_regressors:
             log.warning(f"Model/Config regressor mismatch for {city_name} v{version}. Expected: {expected_regressors}, Got: {loaded_regressors}")
    except Exception as e:
        msg = f"Error loading model {model_path}: {e}"
        log.error(msg, exc_info=True); raise ModelLoadError(msg) from e

    # --- Load Metadata (for historical residual) ---
    metadata_filename = f"{city_name}_prophet_metadata_{version}.json"
    metadata_path = os.path.join(models_dir, metadata_filename)
    metadata = {"last_historical_residual": 0.0} # Default to no correction.
    log.info(f"Loading metadata for {city_name} (v{version}) from: {metadata_path}")
    if not os.path.exists(metadata_path):
        log.warning(f"Metadata file not found: {metadata_path}. Residual correction will use 0.")
    else:
        try:
            with open(metadata_path, 'r') as f:
                loaded_meta = json.load(f)
            if "last_historical_residual" in loaded_meta:
                 metadata["last_historical_residual"] = float(loaded_meta["last_historical_residual"])
                 log.info(f"Successfully loaded historical residual: {metadata['last_historical_residual']:.2f}")
            else:
                 log.warning(f"Key 'last_historical_residual' not found in {metadata_path}. Using 0.")
        except (json.JSONDecodeError, ValueError, TypeError, Exception) as e:
            log.error(f"Error loading or parsing metadata {metadata_path}: {e}. Using residual 0.", exc_info=True)

    _loaded_models_cache[model_key] = model
    _loaded_metadata_cache[metadata_key] = metadata
    return model, metadata



def generate_forecast(target_city, days_ahead=DEFAULT_FORECAST_DAYS, apply_residual_correction=True):
    """
    Generates an AQI forecast using a trained model, weather forecasts, and historical residual correction.

    This is the main prediction workflow:
    1.  Loads the appropriate model and its historical residual value.
    2.  Fetches a weather forecast for the required number of days.
    3.  Prepares the weather data to be used as regressors.
    4.  Generates a base forecast using the Prophet model.
    5.  Applies the historical residual with decay to create an adjusted forecast.
    """
    log.info(f"Generating {days_ahead}-day forecast WITH WEATHER for {target_city}...")

    # 1. Load Model and Associated Metadata
    try:
        trained_model, model_metadata = load_prophet_model_and_metadata(target_city)
        if trained_model is None or model_metadata is None: raise ModelLoadError("Model/Metadata load failed.")
        model_regressors = list(trained_model.extra_regressors.keys())
        historical_residual = model_metadata.get("last_historical_residual", 0.0)
        log.info(f"Model loaded. Regressors: {model_regressors}. Historical residual: {historical_residual:.2f}")
    except (ModelFileNotFoundError, ModelLoadError) as e: log.error(f"Cannot forecast: {e}"); return None
    except Exception as e: log.error(f"Unexpected error loading model/meta: {e}", exc_info=True); return None

    # 2. Fetch and Prepare Weather Forecast Data for Regressors
    try:
        last_train_date = trained_model.history_dates.max()
        today_actual_date = pd.Timestamp.now().normalize()
        start_target_forecast_date = today_actual_date + pd.Timedelta(days=1)
        end_target_forecast_date = today_actual_date + pd.Timedelta(days=days_ahead)

        total_periods_needed = (end_target_forecast_date - last_train_date).days
        if total_periods_needed <= 0: raise PredictionError("Forecast end date not after last train date.")

        periods_for_prophet = max(1, total_periods_needed)
        future_dates_df_full = trained_model.make_future_dataframe(periods=periods_for_prophet, include_history=False)
        future_dates_df_full = future_dates_df_full[future_dates_df_full['ds'] > last_train_date].reset_index(drop=True)
        if future_dates_df_full.empty: raise PredictionError("make_future_dataframe created empty df.")
        log.info(f"Model prediction range: {future_dates_df_full['ds'].min().date()} to {future_dates_df_full['ds'].max().date()}")

    except Exception as e: log.error(f"Error creating future dates: {e}"); raise PredictionError(f"Date range failed: {e}") from e

    # 3. Generate Initial Prophet Forecast
    if model_regressors:
        log.info("Fetching and preparing weather forecast regressors...")
        try:
             weather_api_query_city = target_city
             if target_city in CONFIG.get('modeling', {}).get('target_cities', []):
                  weather_api_query_city = f"{target_city}, India"
             weather_forecast_list = get_weather_forecast(weather_api_query_city, days=days_ahead)
             if not weather_forecast_list:
                  raise PredictionError(f"Weather forecast unavailable for {weather_api_query_city}.")

             weather_forecast_df = pd.DataFrame(weather_forecast_list)
             if 'date' not in weather_forecast_df.columns:
                  raise PredictionError("Weather forecast response missing 'date' column.")
             weather_forecast_df['ds'] = pd.to_datetime(weather_forecast_df['date'])
             weather_forecast_df.set_index('ds', inplace=True) # Index by date

             full_date_range_index = pd.date_range(start=future_dates_df_full['ds'].min(), end=future_dates_df_full['ds'].max(), freq='D')
             aligned_weather_df = pd.DataFrame(index=full_date_range_index)
             aligned_weather_df.index.name = 'ds'

             log.info("Mapping forecast data and filling NaNs...")
             for regressor in model_regressors:
                 source_col = None
                 if regressor == 'temperature_2m':           source_col = 'avgtemp_c'
                 elif regressor == 'relative_humidity_2m':   source_col = 'avghumidity'
                 elif regressor == 'wind_speed_10m':         source_col = 'maxwind_kph'

                 if source_col and source_col in weather_forecast_df.columns:
                     aligned_weather_df[regressor] = weather_forecast_df[source_col].reindex(aligned_weather_df.index)
                     aligned_weather_df[regressor] = pd.to_numeric(aligned_weather_df[regressor], errors='coerce')
                 else:
                     log.warning(f"No mapping or source data for regressor '{regressor}'. Column initialized with NaN.")
                     aligned_weather_df[regressor] = np.nan 

             cols_to_fill = list(aligned_weather_df.columns)
             if cols_to_fill:
                 if aligned_weather_df[cols_to_fill].isnull().any().any():
                     log.warning(f"NaNs detected before fill. Applying ffill/bfill.")
                     aligned_weather_df[cols_to_fill] = aligned_weather_df[cols_to_fill].ffill().bfill()
                     if aligned_weather_df[cols_to_fill].isnull().any().any():
                          missing = aligned_weather_df[cols_to_fill].isnull().sum(); raise PredictionError(f"Unfillable NaNs remain: {missing[missing > 0].to_dict()}")
                     else: log.info("Successfully filled future NaNs.")
                 else: log.info("No NaNs found in regressors after reindex/mapping.")

             future_dates_df_full = pd.merge(
                 future_dates_df_full[['ds']], 
                 aligned_weather_df.reset_index(), 
                 on='ds',
                 how='left'
             )

             missing_final_cols = set(model_regressors) - set(future_dates_df_full.columns)
             if missing_final_cols: raise PredictionError(f"Final future df missing regressors after merge: {missing_final_cols}")


        except (APIError, ValueError, KeyError, PredictionError, Exception) as e:
             log.error(f"Error preparing weather regressors: {e}. Cannot predict.", exc_info=True)
             raise PredictionError(f"Failed weather regressor prep: {e}") from e
    else:
        log.info("Model does not require weather regressors.")

    # 4. Generate Initial Prophet Forecast
    try:
        log.info(f"Generating Prophet forecast for {target_city}...")
        required_pred_cols = ['ds'] + model_regressors
        if future_dates_df_full[required_pred_cols].isnull().any().any():
             nan_cols = future_dates_df_full[required_pred_cols].isnull().sum(); raise PredictionError(f"NaNs before predict: {nan_cols[nan_cols > 0].to_dict()}")
        log.debug(f"DataFrame sent to model.predict():\n{future_dates_df_full[required_pred_cols]}")
        full_range_forecast = trained_model.predict(future_dates_df_full[required_pred_cols])
    except Exception as e: log.error(f"Failed Prophet predict step: {e}", exc_info=True); raise PredictionError(f"Prophet predict failed: {e}") from e

    # 5. Slice out the target forecast days (tomorrow -> today + days_ahead)
    target_forecast_mask = (full_range_forecast['ds'] >= start_target_forecast_date) & (full_range_forecast['ds'] <= end_target_forecast_date)
    forecast = full_range_forecast.loc[target_forecast_mask, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    if forecast.empty: log.warning(f"No forecast data for target date range after slicing."); return None


    # 6. Apply Historical Residual Correction
    forecast['yhat_adjusted'] = forecast['yhat'].copy()
    forecast['residual'] = 0.0
    if apply_residual_correction and historical_residual != 0.0:
        log.info(f"Applying loaded historical residual ({historical_residual:.2f}) with decay...")
        try:
            base_residual = historical_residual
            applied_residuals = []; current_decay = 1.0; decay_factor = RESIDUAL_DECAY_FACTOR
            for _ in range(len(forecast)): applied_residuals.append(base_residual * current_decay); current_decay *= decay_factor
            if len(applied_residuals) == len(forecast):
                 forecast['residual'] = applied_residuals
                 forecast['yhat_adjusted'] = (forecast['yhat'] + forecast['residual']).clip(lower=REALISTIC_MIN_AQI)
                 log.info(f"Applied decaying historical residual correction (factor={decay_factor}).")
            else: log.warning("Residual/Forecast len mismatch. Applying constant."); forecast['residual'] = base_residual; forecast['yhat_adjusted'] = (forecast['yhat'] + base_residual).clip(lower=REALISTIC_MIN_AQI)
        except Exception as e: log.error(f"Error applying historical residual: {e}. Using raw.", exc_info=True); forecast['residual'] = 0.0; forecast['yhat_adjusted'] = forecast['yhat'].copy().clip(lower=REALISTIC_MIN_AQI)
    else: log.info("Residual correction not applied or historical residual is zero."); forecast['residual'] = 0.0; forecast['yhat_adjusted'] = forecast['yhat'].copy().clip(lower=REALISTIC_MIN_AQI)


    log.info(f"Forecast generation complete for {target_city}.")
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'residual', 'yhat_adjusted']]


# --- UI and Risk Interpretation Helpers ---

def format_forecast_for_ui(forecast_df):
    """
    Formats a forecast DataFrame into a simple list of dictionaries for UI display.

    Args:
        forecast_df: The DataFrame returned by `generate_forecast`.

    Returns:
        A list of dicts, e.g., [{'date': '2023-01-01', 'predicted_aqi': 150}, ...].
    """
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

def get_predicted_weekly_risks(city_name, days_ahead=DEFAULT_FORECAST_DAYS):
    """
    Generates a forecast and interprets its health implications for each day.

    This high-level function is used by the UI to display predicted risks.

    Args:
        city_name: The city to generate risks for.
        days_ahead: The number of days to forecast.

    Returns:
        A list of dictionaries, each containing date, AQI, and interpreted risk info.
    """
    log.info(f"Getting predicted weekly risks (Section 6) for {city_name}...")
    forecast_df = None
    try:
        forecast_df = generate_forecast(target_city=city_name, days_ahead=days_ahead, apply_residual_correction=True)
    except PredictionError as pe: log.error(f"PredictionError generating forecast for weekly risks: {pe}")
    except Exception as e: log.error(f"Unhandled error generating forecast for weekly risks: {e}", exc_info=True)
    if forecast_df is None or forecast_df.empty:
        log.error(f"Failed to get forecast for {city_name}. Cannot determine risks.")
        return []
    predicted_risks_list = []
    log.info(f"Interpreting health implications for {len(forecast_df)} days...")
    for index, row in forecast_df.iterrows():
        try:
            predicted_aqi = row['yhat_adjusted']; forecast_date = pd.to_datetime(row['ds']).strftime('%Y-%m-%d')
            if pd.isna(predicted_aqi): continue
            aqi_value_int = int(round(predicted_aqi))
            aqi_category_info = get_aqi_info(aqi_value_int)
            if aqi_category_info: predicted_risks_list.append({"date": forecast_date, "predicted_aqi": aqi_value_int, "level": aqi_category_info.get("level", "N/A"), "color": aqi_category_info.get("color", "#FFFFFF"), "implications": aqi_category_info.get("implications", "N/A")})
            else: log.warning(...); predicted_risks_list.append({"date": forecast_date, "predicted_aqi": aqi_value_int, "level": "Unknown", "color": "#808080", "implications": "Category undefined."})
        except Exception as e: log.error(...); continue
    log.info(f"Finished interpreting risks for {city_name}.")
    return predicted_risks_list

# This new function will be the primary entry point for the UI to get forecasts.
def get_calibrated_forecast_and_risks(city_name: str, days_ahead: int = 3):
    """
    Generates a calibrated forecast and interprets risks in a single, robust workflow.

    This function orchestrates the entire prediction process by:
    1. Fetching the real-time, live AQI value.
    2. Getting the model's raw forecast for today to calculate a "live residual".
    3. Generating the raw forecast for the future.
    4. Applying the live residual (with decay) to the future forecast.
    5. Interpreting the final, calibrated forecast into health risks.

    Args:
        city_name (str): The city to generate the forecast for.
        days_ahead (int): The number of days to forecast.

    Returns:
        list[dict]: A list of dictionaries, each containing the final, calibrated
                    data for one forecast day (date, AQI, level, implications, etc.).
                    Returns an empty list if any critical step fails.
    """
    # --- Step 0: Load Model Metadata to get historical residual ---
    try:
        _, model_metadata = load_prophet_model_and_metadata(city_name)
    except (ModelFileNotFoundError, ModelLoadError):
        log.error(f"Cannot load metadata for {city_name}. Blended residual will not be used.")
        model_metadata = {}

    # --- Step 1: Fetch Live AQI ---
    log.info(f"Starting calibrated forecast for '{city_name}'. Fetching live AQI...")
    try:
        # The client function expects the "City, Country" format for this API call.
        live_aqi_data = get_current_aqi_for_city(f"{city_name}, India")
        
        # Check if the API call was successful and returned a valid AQI value.
        if live_aqi_data and live_aqi_data.get('aqi') is not None:
            live_aqi_value = live_aqi_data['aqi']
            log.info(f"Successfully fetched live AQI for {city_name}: {live_aqi_value}")
        else:
            # If we can't get the live AQI, we cannot calibrate the forecast.
            # We will log the error and return an empty list, signaling a failure.
            error_msg = live_aqi_data.get('error', 'Live AQI value was None or missing.')
            log.error(f"Cannot proceed with calibrated forecast for {city_name}: {error_msg}")
            return [] # Return empty list to indicate failure
    except Exception as e:
        # Catch any other unexpected errors during the API call.
        log.error(f"An unexpected error occurred while fetching live AQI for {city_name}: {e}", exc_info=True)
        return [] # Return empty list to indicate failure

    # --- Step 2: Calculate Live Residual ---
    log.info(f"Getting model's raw prediction for today to calculate residual...")
    try:
        # We need to know the last date in the model's training history.
        last_train_date = None
        temp_model, _ = load_prophet_model_and_metadata(city_name)
        if temp_model:
            last_train_date = temp_model.history_dates.max()
        else:
            # This should not happen if the model is valid, but as a safeguard:
            raise ModelLoadError("Failed to load model to determine last training date.")

        # Calculate the number of days between the last training day and today.
        days_since_training = (pd.Timestamp.now().normalize() - last_train_date).days
        if days_since_training < 0:
            log.warning("System time appears to be before the model's last training date. Residual cannot be calculated.")
            # Fallback: We can't calculate a live residual, so we proceed without it.
            live_residual = 0.0
        else:
            # Generate a forecast from the end of training up to today.
            todays_forecast_df = generate_forecast(
                target_city=city_name,
                days_ahead=days_since_training,
                apply_residual_correction=False # We want the RAW model output
            )

            if todays_forecast_df is None or todays_forecast_df.empty:
                raise PredictionError("Generating today's forecast returned no data.")

            # Get the model's prediction for today (the last row in the dataframe).
            yhat_for_today = todays_forecast_df.iloc[-1]['yhat']

            # This is the core calculation!
            live_residual = live_aqi_value - yhat_for_today
            log.info(f"Calculated raw live residual: {live_residual:.2f}")

            # --- Apply Capping as a Sanity Check ---
            if abs(live_residual) > MAX_RESIDUAL_CAP:
                log.warning(f"Live residual {live_residual:.2f} exceeds cap of {MAX_RESIDUAL_CAP}. Capping it.")
                live_residual = math.copysign(MAX_RESIDUAL_CAP, live_residual) # Keeps the sign (+ or -)

            log.info(f"Final (capped) live residual to be applied: {live_residual:.2f}")

    except Exception as e:
        log.error(f"Unexpected error calculating live residual for {city_name}: {e}. Proceeding without correction.", exc_info=True)
        live_residual = 0.0 # Default to no correction on error

    # --- NEW: Calculate the Blended Residual ---
    # Get the model's long-term historical residual from its metadata.
    # Note: This requires a slight modification to the start of this function.
    historical_residual = model_metadata.get("last_historical_residual", 0.0)
    log.info(f"Model's historical residual is: {historical_residual:.2f}")

    # Create the blended residual.
    blended_residual = (live_residual * LIVE_RESIDUAL_WEIGHT) + (historical_residual * (1 - LIVE_RESIDUAL_WEIGHT))
    log.info(f"Blended Residual Calculation: ({live_residual:.2f} * {LIVE_RESIDUAL_WEIGHT}) + ({historical_residual:.2f} * {1-LIVE_RESIDUAL_WEIGHT:.2f}) = {blended_residual:.2f}")

    # --- Step 3: Generate and Apply Corrected Forecast ---
        # --- Step 3: Generate the future forecast and apply the live residual ---
    log.info(f"Generating {days_ahead}-day future forecast to apply live residual...")
    try:
        # Get the raw forecast for the next N days.
        future_forecast_df = generate_forecast(
            target_city=city_name,
            days_ahead=days_ahead,
            apply_residual_correction=False # MUST be False, we are doing our own correction
        )

        if future_forecast_df is None or future_forecast_df.empty:
            raise PredictionError("Generating the future forecast returned no data.")

        # Apply the live residual with a decay factor.
        # This makes the correction strongest for tomorrow and weaker for subsequent days.
        if blended_residual != 0.0:
            log.info(f"Applying blended residual ({blended_residual:.2f}) with decay to future forecast.")
            # The exponent 'i' starts at 0...
            decaying_residuals = [blended_residual * (RESIDUAL_DECAY_FACTOR ** i) for i in range(len(future_forecast_df))]
        else:
            # If there was no residual to apply, the adjusted value is the same as the raw value.
            log.info("No live residual to apply. Using raw model forecast.")
            future_forecast_df['yhat_adjusted'] = future_forecast_df['yhat'].clip(lower=REALISTIC_MIN_AQI)
        
        log.info("Successfully generated calibrated forecast.")

    except (ModelFileNotFoundError, ModelLoadError, PredictionError) as e:
        log.error(f"Could not generate calibrated forecast for {city_name} due to modeling error: {e}")
        return [] # Return empty list on failure
    except Exception as e:
        log.error(f"Unexpected error generating calibrated forecast for {city_name}: {e}", exc_info=True)
        return []

    # --- Step 4: Interpret risks from the final calibrated forecast ---
    log.info(f"Interpreting health risks for the calibrated forecast...")
    predicted_risks_list = []
    for _, row in future_forecast_df.iterrows():
        try:
            # Use the newly calculated 'yhat_adjusted' for risk interpretation.
            predicted_aqi = int(round(row['yhat_adjusted']))
            forecast_date = pd.to_datetime(row['ds']).strftime('%Y-%m-%d')
            
            aqi_category_info = get_aqi_info(predicted_aqi)
            if aqi_category_info:
                predicted_risks_list.append({
                    "date": forecast_date,
                    "predicted_aqi": predicted_aqi,
                    "level": aqi_category_info.get("level", "N/A"),
                    "color": aqi_category_info.get("color", "#FFFFFF"),
                    "implications": aqi_category_info.get("implications", "N/A")
                })
            else:
                # Fallback for unexpected cases where get_aqi_info might fail.
                log.warning(f"Could not get AQI info for value {predicted_aqi}. Using fallback.")
                predicted_risks_list.append({
                    "date": forecast_date,
                    "predicted_aqi": predicted_aqi,
                    "level": "Unknown",
                    "color": "#808080", # Gray color for unknown
                    "implications": "Category could not be determined."
                })
        except (ValueError, TypeError) as e:
            log.warning(f"Skipping risk interpretation for a row due to data issue: {e}. Row: {row}")
            continue
            
    log.info(f"Finished generating calibrated risks for {city_name}.")
    return predicted_risks_list, future_forecast_df


# --- Example Usage / Direct Execution ---
if __name__ == "__main__":
    # This block runs only when the script is executed directly.
    # It serves as a quick test and demonstration of the module's functions.
    if not logging.getLogger().hasHandlers:
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s')
         log.info("Configured fallback logging for direct script run of predictor.py.")

    print("\n" + "="*40)
    print(" Running predictor.py Self-Test ")
    print("="*40 + "\n")

    test_city = "Delhi"
    days_to_forecast = CONFIG.get('modeling', {}).get('forecast_days', 3)
    print(f"--- Test Case: Generating {days_to_forecast}-day forecast for '{test_city}' ---")

    try:
        # Test 1: Standard forecast with residual correction
        print("\n[1] Generating forecast WITH historical residual correction...")
        forecast_corrected = generate_forecast(
            test_city,
            days_ahead=days_to_forecast,
            apply_residual_correction=True
        )
        if forecast_corrected is not None:
            print(forecast_corrected[['ds', 'yhat', 'residual', 'yhat_adjusted']].round(2))
        else:
            print(f"--> FAILED: Could not generate corrected forecast for {test_city}.")

        # Test 2: Forecast without residual correction
        print("\n[2] Generating forecast WITHOUT historical residual correction...")
        forecast_raw = generate_forecast(
            test_city,
            days_ahead=days_to_forecast,
            apply_residual_correction=False
        )
        if forecast_raw is not None:
            print(forecast_raw[['ds', 'yhat', 'residual', 'yhat_adjusted']].round(2))
        else:
            print(f"--> FAILED: Could not generate raw forecast for {test_city}.")

        # Test 3: UI Formatting
        print("\n[3] Formatting forecast for UI...")
        if forecast_corrected is not None:
            ui_data = format_forecast_for_ui(forecast_corrected)
            import pprint
            pprint.pprint(ui_data)
        else:
            print("--> SKIPPED: Cannot format UI data because corrected forecast failed.")

        # Test 4: Weekly Risk Generation
        print("\n[4] Getting predicted weekly risks...")
        weekly_risks = get_predicted_weekly_risks(test_city, days_ahead=days_to_forecast)
        if weekly_risks:
            import pprint
            pprint.pprint(weekly_risks)
        else:
            print(f"--> FAILED: Could not generate predicted weekly risks for {test_city}.")

        # Test 5: Non-existent Model
        print("\n[5] Testing for a city with no model ('Atlantis')...")
        forecast_none = generate_forecast("Atlantis", days_ahead=days_to_forecast)
        if forecast_none is None:
            print("--> SUCCESS: Correctly returned None as expected.")
        else:
            print("--> FAILURE: Expected None but received a forecast.")

    except Exception as e:
        print(f"\n!!! AN ERROR OCCURRED DURING THE SELF-TEST: {e} !!!")
        log.error("Error in predictor.py __main__ block", exc_info=True)

    print("\n" + "="*40)
    print(" predictor.py Self-Test Finished ")
    print("="*40 + "\n")