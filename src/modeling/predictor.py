# File: src/modeling/predictor.py (v3 - Historical Residual Correction)

"""
Handles loading saved Prophet models (trained with weather regressors) and
generating AQI forecasts incorporating weather forecasts and historical residuals.

Provides functions to:
- Load city-specific Prophet models and their associated historical residual (with caching).
- Generate forecasts including weather regressors and apply residual correction based on stored historical error.
- Format the forecast output for easy UI consumption (JSON).
- Interpret the forecast to provide predicted weekly health risks based on AQI levels.

Relies on configuration for model paths/versions/regressors and external modules
for Weather Forecast API access and AQI category interpretation.
"""

import pandas as pd
import os
import sys
import logging # Standard logging import
import json
import numpy as np
from prophet import Prophet
from prophet.serialize import model_from_json

# --- Setup Path & Import Config/Exceptions ---
# (Keep existing path setup logic)
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')
    if 'src' not in os.listdir(PROJECT_ROOT):
        alt_root = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
        if 'src' in os.listdir(alt_root): PROJECT_ROOT = alt_root
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

try:
    from src.config_loader import CONFIG
    # NOTE: No longer need get_current_aqi_for_city from client.py for residual correction
    from src.api_integration.weather_client import get_weather_forecast # Need forecast API
    from src.health_rules.info import get_aqi_info
    from src.exceptions import ModelFileNotFoundError, ModelLoadError, PredictionError, APIError, APITimeoutError, ConfigError # Add relevant API errors if needed by weather forecast
    # Logging configured by config_loader
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"Predictor: Could not import dependencies. Error: {e}", exc_info=True)
    CONFIG = {}
    # Define dummy exceptions/functions
    class ModelFileNotFoundError(FileNotFoundError): pass
    class ModelLoadError(Exception): pass
    class PredictionError(Exception): pass
    class APIError(Exception): pass # Keep APIError for weather forecast
    class APITimeoutError(Exception): pass # Keep for weather forecast
    class ConfigError(Exception): pass
    def get_weather_forecast(city_name, days): logging.error("Weather forecast client unavailable."); return None
    def get_aqi_info(aqi_value): logging.error("AQI info unavailable."); return None
except Exception as e:
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


# --- Get Logger ---
log = logging.getLogger(__name__)
print("DEBUG: Reached end of imports") # DEBUG PRINT

# --- Configuration Values ---
# (Keep logic for getting config values)
relative_models_dir = CONFIG.get('paths', {}).get('models_dir', 'models')
MODELS_DIR = os.path.join(PROJECT_ROOT, relative_models_dir)
MODEL_VERSION = CONFIG.get('modeling', {}).get('prophet_model_version', 'v3_weather')
DEFAULT_FORECAST_DAYS = CONFIG.get('modeling', {}).get('forecast_days', 3)
WEATHER_REGRESSORS_CONFIG = CONFIG.get('modeling', {}).get('weather_regressors', [])
RESIDUAL_DECAY_FACTOR = CONFIG.get('modeling', {}).get('residual_decay_factor', 0.85)
REALISTIC_MIN_AQI = CONFIG.get('modeling', {}).get('min_realistic_aqi', 1)

# --- Model & Metadata Loading Cache ---
_loaded_models_cache = {}
_loaded_metadata_cache = {}

def load_prophet_model_and_metadata(city_name, version=MODEL_VERSION, models_dir=MODELS_DIR):
    """
    Loads a trained Prophet model and its associated metadata (containing historical residual).
    Uses an in-memory cache for both model and metadata.

    Args:
        city_name (str): City name.
        version (str, optional): Model version suffix. Defaults to config.
        models_dir (str, optional): Model directory path. Defaults to config.

    Returns:
        tuple(Prophet, dict): A tuple containing:
                              - The loaded Prophet model instance.
                              - The loaded metadata dictionary (e.g., {'last_historical_residual': -5.2}).
                              Returns (None, None) if either file cannot be loaded.

    Raises:
        ModelFileNotFoundError: If the required model or metadata file doesn't exist.
        ModelLoadError: If the model/metadata file exists but cannot be loaded/deserialized.
    """
    global _loaded_models_cache, _loaded_metadata_cache

    model_key = f"{city_name}_{version}"
    metadata_key = f"{city_name}_{version}_meta"

    # Check cache first
    cached_model = _loaded_models_cache.get(model_key)
    cached_metadata = _loaded_metadata_cache.get(metadata_key)
    if cached_model is not None and cached_metadata is not None:
        log.info(f"Returning cached model and metadata for {city_name} (v{version}).")
        return cached_model, cached_metadata

    # --- Load Model ---
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
        # Verify regressors (optional, keep as before)
        loaded_regressors = set(model.extra_regressors.keys())
        expected_regressors = set(WEATHER_REGRESSORS_CONFIG)
        if loaded_regressors != expected_regressors:
             log.warning(f"Model/Config regressor mismatch for {city_name} v{version}. Expected: {expected_regressors}, Got: {loaded_regressors}")
    except Exception as e:
        msg = f"Error loading model {model_path}: {e}"
        log.error(msg, exc_info=True); raise ModelLoadError(msg) from e

    # --- Load Metadata (Historical Residual) ---
    metadata_filename = f"{city_name}_prophet_metadata_{version}.json"
    metadata_path = os.path.join(models_dir, metadata_filename)
    metadata = {"last_historical_residual": 0.0} # Default if file missing/corrupt
    log.info(f"Loading metadata for {city_name} (v{version}) from: {metadata_path}")
    if not os.path.exists(metadata_path):
        # This might be an error or just mean correction is disabled
        log.warning(f"Metadata file not found: {metadata_path}. Residual correction will use 0.")
        # Raise ModelFileNotFoundError(f"Metadata file not found: {metadata_path}") # Option: Make metadata required
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
            # Raise ModelLoadError(f"Error loading metadata {metadata_path}: {e}") from e # Option: Make metadata required

    # Cache and return
    _loaded_models_cache[model_key] = model
    _loaded_metadata_cache[metadata_key] = metadata
    return model, metadata


# --- Prediction Function (Using HISTORICAL Residual) ---
def generate_forecast(target_city, days_ahead=DEFAULT_FORECAST_DAYS, apply_residual_correction=True):
    """Generates AQI forecast using V3 model, aligned weather forecasts, and historical residual."""
    log.info(f"Generating {days_ahead}-day forecast WITH WEATHER for {target_city}...")

    # 1. Load Model and Metadata
    try:
        trained_model, model_metadata = load_prophet_model_and_metadata(target_city)
        if trained_model is None or model_metadata is None: raise ModelLoadError("Model/Metadata load failed.")
        model_regressors = list(trained_model.extra_regressors.keys())
        historical_residual = model_metadata.get("last_historical_residual", 0.0)
        log.info(f"Model loaded. Regressors: {model_regressors}. Historical residual: {historical_residual:.2f}")
    except (ModelFileNotFoundError, ModelLoadError) as e: log.error(f"Cannot forecast: {e}"); return None
    except Exception as e: log.error(f"Unexpected error loading model/meta: {e}", exc_info=True); return None

    # 2. Determine Date Ranges
    try:
        last_train_date = trained_model.history_dates.max()
        today_actual_date = pd.Timestamp.now().normalize()
        start_target_forecast_date = today_actual_date + pd.Timedelta(days=1)
        end_target_forecast_date = today_actual_date + pd.Timedelta(days=days_ahead)

        # Calculate total periods needed for prediction (from day after train end to forecast end)
        total_periods_needed = (end_target_forecast_date - last_train_date).days
        if total_periods_needed <= 0: raise PredictionError("Forecast end date not after last train date.")

        # Create full future dates dataframe needed by Prophet internal prediction
        future_dates_df_full = trained_model.make_future_dataframe(periods=total_periods_needed, include_history=False)
        future_dates_df_full = future_dates_df_full[future_dates_df_full['ds'] > last_train_date].reset_index(drop=True)
        if future_dates_df_full.empty: raise PredictionError("make_future_dataframe created empty df.")
        log.info(f"Model prediction range: {future_dates_df_full['ds'].min().date()} to {future_dates_df_full['ds'].max().date()}")

    except Exception as e: log.error(f"Error creating future dates: {e}"); raise PredictionError(f"Date range failed: {e}") from e

    # 3. Fetch & Prepare Weather Forecast Data for Regressors
    if model_regressors:
        log.info("Fetching and preparing weather forecast regressors...")
        try:
             # Fetch forecast first
             weather_api_query_city = target_city
             if target_city in CONFIG.get('modeling', {}).get('target_cities', []):
                  weather_api_query_city = f"{target_city}, India"
             weather_forecast_list = get_weather_forecast(weather_api_query_city, days=days_ahead)
             if not weather_forecast_list:
                  raise PredictionError(f"Weather forecast unavailable for {weather_api_query_city}.")

             # Create the weather DataFrame immediately
             weather_forecast_df = pd.DataFrame(weather_forecast_list)
             if 'date' not in weather_forecast_df.columns:
                  raise PredictionError("Weather forecast response missing 'date' column.")
             weather_forecast_df['ds'] = pd.to_datetime(weather_forecast_df['date'])
             weather_forecast_df.set_index('ds', inplace=True) # Index by date

             # Create the full future date range needed by Prophet for alignment
             full_date_range_index = pd.date_range(start=future_dates_df_full['ds'].min(), end=future_dates_df_full['ds'].max(), freq='D')
             aligned_weather_df = pd.DataFrame(index=full_date_range_index)
             aligned_weather_df.index.name = 'ds'

             log.info("Mapping forecast data and filling NaNs...")
             # Now map from weather_forecast_df to aligned_weather_df
             for regressor in model_regressors:
                 source_col = None
                 # --- *** START: MAPPING LOGIC - USER MUST VERIFY/EDIT *** ---
                 if regressor == 'temperature_2m':           source_col = 'avgtemp_c'
                 elif regressor == 'relative_humidity_2m':   source_col = 'avghumidity'
                 elif regressor == 'wind_speed_10m':         source_col = 'maxwind_kph'
                 # Add mappings for other regressors if used
                 # --- *** END: MAPPING LOGIC *** ---

                 if source_col and source_col in weather_forecast_df.columns:
                     # Reindex the specific source column series to the full required date range
                     aligned_weather_df[regressor] = weather_forecast_df[source_col].reindex(aligned_weather_df.index)
                     # Ensure numeric after reindex
                     aligned_weather_df[regressor] = pd.to_numeric(aligned_weather_df[regressor], errors='coerce')
                 else:
                     log.warning(f"No mapping or source data for regressor '{regressor}'. Column initialized with NaN.")
                     aligned_weather_df[regressor] = np.nan # Initialize column if no source

             # Fill NaNs using ffill/bfill in the aligned dataframe
             cols_to_fill = list(aligned_weather_df.columns)
             if cols_to_fill:
                 if aligned_weather_df[cols_to_fill].isnull().any().any():
                     log.warning(f"NaNs detected before fill. Applying ffill/bfill.")
                     aligned_weather_df[cols_to_fill] = aligned_weather_df[cols_to_fill].ffill().bfill()
                     if aligned_weather_df[cols_to_fill].isnull().any().any():
                          missing = aligned_weather_df[cols_to_fill].isnull().sum(); raise PredictionError(f"Unfillable NaNs remain: {missing[missing > 0].to_dict()}")
                     else: log.info("Successfully filled future NaNs.")
                 else: log.info("No NaNs found in regressors after reindex/mapping.")

             # Merge the processed weather data onto the future dates frame
             future_dates_df_full = pd.merge(
                 future_dates_df_full[['ds']], # Original future dates
                 aligned_weather_df.reset_index(), # Weather data with 'ds' column
                 on='ds',
                 how='left'
             )

             # Final check for required columns
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
        # Use the potentially updated future_dates_df_full here
        if future_dates_df_full[required_pred_cols].isnull().any().any():
             nan_cols = future_dates_df_full[required_pred_cols].isnull().sum(); raise PredictionError(f"NaNs before predict: {nan_cols[nan_cols > 0].to_dict()}")
        log.debug(f"DataFrame sent to model.predict():\n{future_dates_df_full[required_pred_cols]}")
        # Predict on the *full* range
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


# --- Helper Function for UI Formatting ---
# (Keep format_forecast_for_ui exactly as it was)
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
# (Keep get_predicted_weekly_risks exactly as it was)
def get_predicted_weekly_risks(city_name, days_ahead=DEFAULT_FORECAST_DAYS):
    """Generates forecast and interprets health implications."""
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

# --- Example Usage Block (Modified Test 2) ---
if __name__ == "__main__":
    print("DEBUG: Entering __main__ block of predictor.py")
    # (Keep logging setup)
    if not logging.getLogger().handlers:
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s')
         log = logging.getLogger(__name__)
         log.info("Configured fallback logging for __main__ block.")

    print("\n" + "="*30); print(" Running predictor.py Tests "); print("="*30 + "\n")
    test_city = "Delhi"; days_to_forecast = CONFIG.get('modeling', {}).get('forecast_days', 3)
    log.info(f"Using forecast horizon: {days_to_forecast} days")
    forecast_api = None
    try:
        print(f"\n--- Test 1: Predicting {days_to_forecast} days for '{test_city}' WITH Historical Residual Correction ---")
        # Renamed test to reflect change
        forecast_api = generate_forecast(test_city, days_ahead=days_to_forecast, apply_residual_correction=True)
        if forecast_api is not None: print(forecast_api)
        else: print(f"Failed test 1 for {test_city}")

        # Test 2 is now less relevant as manual override isn't the primary mechanism, but keep for demo
        print(f"\n--- Test 2: Predicting {days_to_forecast} days WITHOUT Residual Correction ---")
        forecast_raw = generate_forecast(test_city, days_ahead=days_to_forecast, apply_residual_correction=False)
        if forecast_raw is not None: print(forecast_raw)
        else: print(f"Failed test 2 for {test_city}")

        # Renumber subsequent tests
        print("\n--- Test 3: Formatting Forecast for UI ---")
        if forecast_api is not None:
            ui_formatted_data = format_forecast_for_ui(forecast_api)
            print(f"UI Formatted Data for {test_city} (Hist. Corrected Forecast):"); import pprint; pprint.pprint(ui_formatted_data)
        else: print("Skipping test 3 (forecast_api is None).")

        print("\n--- Test 4: Getting Predicted Weekly Risks (Section 6) ---")
        print(f"--- Getting predicted weekly risks for {test_city} ---")
        weekly_risks = get_predicted_weekly_risks(test_city, days_ahead=days_to_forecast)
        if weekly_risks: import pprint; print("Predicted Weekly Risks/Implications:"); pprint.pprint(weekly_risks)
        else: print(f"Could not generate predicted weekly risks for {test_city}.")

        test_city_2 = "Mumbai"
        model_version_test = CONFIG.get('modeling', {}).get('prophet_model_version', 'v3_weather')
        models_dir_test = os.path.join(PROJECT_ROOT, CONFIG.get('paths', {}).get('models_dir', 'models'))
        model_exists = os.path.exists(os.path.join(models_dir_test, f"{test_city_2}_prophet_model_{model_version_test}.json"))
        if model_exists:
             print(f"\n--- Test 5: Predicting {days_to_forecast} days for '{test_city_2}' WITH Historical Residual Correction ---")
             forecast_mumbai = generate_forecast(test_city_2, days_ahead=days_to_forecast, apply_residual_correction=True)
             if forecast_mumbai is not None: print(f"Forecast for {test_city_2}:\n{forecast_mumbai}")
             else: print(f"Failed test 5 for {test_city_2}.")
        else: print(f"\nSkipping Test 5: Model file for {test_city_2} version {model_version_test} not found.")

        print("\n--- Test 6: Predicting for 'Atlantis' (No Model) ---")
        forecast_none = generate_forecast("Atlantis", days_ahead=days_to_forecast)
        if forecast_none is None: print("Success! Correctly returned None.")
        else: print("Failure! Expected None.")

    except Exception as e: print(f"\n!!! Error in main test block: {e} !!!"); log.error("Error in predictor.py __main__", exc_info=True)
    print("\n" + "="*30); print(" predictor.py Tests Finished "); print("="*30 + "\n")