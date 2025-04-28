# File: src/modeling/predictor.py (v3 - Weather Regressors - Final + Fixes)

"""
Handles loading saved Prophet models (trained with weather regressors) and
generating AQI forecasts incorporating weather forecasts.

Provides functions to:
- Load city-specific Prophet models (with caching).
- Generate forecasts including weather regressors and residual correction.
- Format the forecast output for easy UI consumption (JSON).
- Interpret the forecast to provide predicted weekly health risks based on AQI levels.

Relies on configuration for model paths/versions/regressors and external modules
for API access (AQI and Weather Forecasts) and AQI category interpretation.
"""

import pandas as pd
import os
import sys
import logging # Standard logging import
import json
import numpy as np # Import added
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
    # Import API clients, info func, exceptions
    from src.api_integration.client import get_current_aqi_for_city
    from src.api_integration.weather_client import get_weather_forecast, get_current_weather # Import both weather funcs
    from src.health_rules.info import get_aqi_info
    from src.exceptions import ModelFileNotFoundError, ModelLoadError, PredictionError, APIKeyError, APITimeoutError, APINotFoundError, APIError, ConfigError
    # Logging configured by config_loader
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
    def get_current_aqi_for_city(city_name): logging.error("AQI client unavailable."); return None
    def get_weather_forecast(city_name, days): logging.error("Weather forecast client unavailable."); return None
    def get_current_weather(city_name): logging.error("Weather current client unavailable."); return None
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
     def get_current_aqi_for_city(city_name): logging.error("AQI client unavailable."); return None
     def get_weather_forecast(city_name, days): logging.error("Weather forecast client unavailable."); return None
     def get_current_weather(city_name): logging.error("Weather current client unavailable."); return None
     def get_aqi_info(aqi_value): logging.error("AQI info unavailable."); return None


# --- Get Logger ---
log = logging.getLogger(__name__)
print("DEBUG: Reached end of imports") # DEBUG PRINT

# --- Configuration Values ---
# (Keep logic for getting config values)
relative_models_dir = CONFIG.get('paths', {}).get('models_dir', 'models')
MODELS_DIR = os.path.join(PROJECT_ROOT, relative_models_dir)
MODEL_VERSION = CONFIG.get('modeling', {}).get('prophet_model_version', 'v3_weather') # Expect v3
DEFAULT_FORECAST_DAYS = CONFIG.get('modeling', {}).get('forecast_days', 3) # Expect 3
WEATHER_REGRESSORS_CONFIG = CONFIG.get('modeling', {}).get('weather_regressors', []) # Read list from config
RESIDUAL_DECAY_FACTOR = CONFIG.get('modeling', {}).get('residual_decay_factor', 0.85)

# --- Model Loading Cache ---
_loaded_models_cache = {}

def load_prophet_model(city_name, version=MODEL_VERSION, models_dir=MODELS_DIR):
    """Loads a Prophet model (v3 expected), using cache. Verifies expected regressors."""
    # (Keep function logic as is)
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
        log.error(msg); raise ModelFileNotFoundError(msg)
    try:
        with open(model_path, 'r') as fin:
            model = model_from_json(json.load(fin))
        log.info(f"Model for {city_name} (v{version}) loaded successfully.")
        loaded_regressors = set(model.extra_regressors.keys())
        expected_regressors = set(WEATHER_REGRESSORS_CONFIG)
        if loaded_regressors != expected_regressors:
             log.warning(f"Mismatch! Expected regressors ({expected_regressors}) vs loaded ({loaded_regressors}) for {city_name} v{version}.")
        _loaded_models_cache[model_key] = model
        return model
    except Exception as e:
        msg = f"Error loading model {model_path}: {e}"
        log.error(msg, exc_info=True)
        raise ModelLoadError(msg) from e


# --- Prediction Function (REVISED AGAIN FOR DATE ALIGNMENT) ---
def generate_forecast(target_city, days_ahead=DEFAULT_FORECAST_DAYS, apply_residual_correction=True, last_known_aqi=None):
    """Generates AQI forecast using V3 model and weather forecasts, handling date alignment."""
    log.info(f"Generating {days_ahead}-day forecast WITH WEATHER for {target_city}...")

    # 1. Load Model
    try:
        trained_model = load_prophet_model(target_city)
        model_regressors = list(trained_model.extra_regressors.keys())
        log.info(f"Model loaded. Expects regressors: {model_regressors}")
    except (ModelFileNotFoundError, ModelLoadError) as e:
        log.error(f"Cannot generate forecast: {e}"); return None
    except Exception as e:
        log.error(f"Unexpected error loading model: {e}", exc_info=True); return None

    # 2. Determine Date Ranges Needed
    try:
        last_train_date = trained_model.history_dates.max()
        today_actual_date = pd.Timestamp.now().normalize() # Today's date
        start_forecast_date = today_actual_date + pd.Timedelta(days=1) # Forecast starts tomorrow
        end_forecast_date = today_actual_date + pd.Timedelta(days=days_ahead) # Forecast ends N days from today

        log.info(f"Last Train Date: {last_train_date.date()}, Today: {today_actual_date.date()}")
        log.info(f"Target Forecast Range: {start_forecast_date.date()} to {end_forecast_date.date()}")

        # Calculate total periods needed for make_future_dataframe
        # Needs to go from last_train_date up to end_forecast_date
        if end_forecast_date <= last_train_date:
             log.error(f"Forecast end date ({end_forecast_date.date()}) is not after last training date ({last_train_date.date()}). Cannot forecast future.")
             return None
        total_periods_needed = (end_forecast_date - last_train_date).days
        log.info(f"Total periods to predict from model: {total_periods_needed}")

        if total_periods_needed <= 0:
             log.error(f"Calculated periods needed ({total_periods_needed}) is not positive.")
             return None

        # Create future dataframe covering the whole span needed for prediction + potential correction ref
        future_dates_df_full = trained_model.make_future_dataframe(
            periods=total_periods_needed,
            include_history=False
        )
        # Ensure it only contains dates strictly after training data
        future_dates_df_full = future_dates_df_full[future_dates_df_full['ds'] > last_train_date].reset_index(drop=True)

        if future_dates_df_full.empty:
            log.error("make_future_dataframe created empty dataframe after filtering.")
            raise PredictionError("Failed to create necessary future dates.")

        log.debug(f"Full future dates range: {future_dates_df_full['ds'].min().date()} to {future_dates_df_full['ds'].max().date()}")

    except Exception as e:
         log.error(f"Error creating future dates: {e}", exc_info=True)
         raise PredictionError(f"Failed date range calculation: {e}") from e

    # 3. Fetch Weather Forecast and Merge (If regressors used)
    if model_regressors:
        log.info("Fetching weather forecast for required future dates...")
        try:
            # Fetch forecast only for the days we actually need to display
            weather_api_query_city = target_city
            if target_city in CONFIG.get('modeling', {}).get('target_cities', []):
                 weather_api_query_city = f"{target_city}, India"
            # Fetch forecast for 'days_ahead' starting from today/tomorrow
            weather_forecast_list = get_weather_forecast(weather_api_query_city, days=days_ahead)

            if not weather_forecast_list:
                 raise PredictionError(f"Weather forecast unavailable for {weather_api_query_city}.")

            weather_forecast_df = pd.DataFrame(weather_forecast_list)
            weather_forecast_df['ds'] = pd.to_datetime(weather_forecast_df['date'])

            # Merge onto the *full* future dates required by the model
            # Use left merge; dates without forecasts will have NaN weather initially
            columns_to_merge = list(weather_forecast_df.columns.drop('date')) # This list now includes 'ds' and weather fields
            future_dates_df_full = pd.merge(
                future_dates_df_full,
                weather_forecast_df[columns_to_merge], # Use the corrected column list
                on='ds',
                how='left'
            )
            log.info("Mapping forecast data and filling NaNs for regressor columns...")
            for regressor in model_regressors:
                if regressor not in future_dates_df_full.columns:
                    # Map available forecast data to regressor columns
                    source_col = None
                    # --- *** START: MAPPING LOGIC - USER MUST VERIFY/EDIT *** ---
                    if regressor == 'temperature_2m':           source_col = 'avgtemp_c'
                    elif regressor == 'relative_humidity_2m':   source_col = 'avghumidity'
                    elif regressor == 'wind_speed_10m':         source_col = 'maxwind_kph'
                    # --- *** END: MAPPING LOGIC *** ---
                    if source_col and source_col in future_dates_df_full.columns:
                         future_dates_df_full[regressor] = future_dates_df_full[source_col]
                    else:
                         log.warning(f"No mapping for regressor '{regressor}'. Initializing with NaN.")
                         future_dates_df_full[regressor] = np.nan

                # Convert mapped/existing column to numeric
                if regressor in future_dates_df_full.columns:
                     future_dates_df_full[regressor] = pd.to_numeric(future_dates_df_full[regressor], errors='coerce')


            # --- Fill NaNs in Regressors in the FULL future dataframe ---
            cols_to_fill = model_regressors
            if future_dates_df_full[cols_to_fill].isnull().any().any():
                 log.warning(f"NaNs detected in future regressors before filling. Applying ffill/bfill.")
                 future_dates_df_full[cols_to_fill] = future_dates_df_full[cols_to_fill].ffill().bfill()
                 # Check *again*
                 if future_dates_df_full[cols_to_fill].isnull().any().any():
                      missing_details = future_dates_df_full[cols_to_fill].isnull().sum()
                      log.error(f"Unfillable NaNs remain in future regressors: {missing_details[missing_details > 0].to_dict()}. Cannot predict.")
                      raise PredictionError(f"Unfillable NaNs in future regressors: {missing_details[missing_details > 0].to_dict()}")
                 else: log.info("Successfully filled NaNs in future regressors.")
            else: log.info("No NaNs found in future regressor columns after merge/mapping.")

            # Ensure all required columns exist before prediction
            missing_final_cols = set(model_regressors) - set(future_dates_df_full.columns)
            if missing_final_cols: raise PredictionError(f"Final future df missing regressors: {missing_final_cols}")

        except (APIError, ValueError, KeyError, PredictionError, Exception) as e:
            log.error(f"Error preparing weather regressors: {e}. Cannot predict.", exc_info=True)
            raise PredictionError(f"Failed weather regressor prep: {e}") from e
    else:
        log.info("Model does not require weather regressors.")


    # 4. Generate Initial Prophet Forecast for the required range
    try:
        log.info(f"Generating Prophet forecast for {target_city}...")
        required_pred_cols = ['ds'] + model_regressors
        # Final check before predict
        if future_dates_df_full[required_pred_cols].isnull().any().any():
             nan_cols = future_dates_df_full[required_pred_cols].isnull().sum()
             log.error(f"FATAL: NaNs detected just before predict: {nan_cols[nan_cols > 0].to_dict()}")
             raise PredictionError("NaNs found just before prediction.")

        # Predict on the *full* future range needed
        full_range_forecast = trained_model.predict(future_dates_df_full[required_pred_cols])

    except Exception as e:
        log.error(f"Failed Prophet predict step: {e}", exc_info=True)
        raise PredictionError(f"Prophet predict step failed: {e}") from e

    # 5. Slice out the target forecast days (tomorrow -> today + days_ahead)
    target_forecast_mask = (full_range_forecast['ds'] >= start_forecast_date) & (full_range_forecast['ds'] <= end_forecast_date)
    forecast = full_range_forecast.loc[target_forecast_mask, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

    if forecast.empty:
         log.warning(f"No forecast data generated for target date range after slicing. Check dates.")
         return None # Return None if slicing resulted in empty df


    # 6. Apply Residual Correction (Best Effort)
    forecast['yhat_adjusted'] = forecast['yhat'].copy()
    forecast['residual'] = 0.0
    if apply_residual_correction:
        log.info(f"Attempting residual correction for {target_city}...")
        try:
            # Use the *actual* last training date as the reference point
            today_ds_ref = last_train_date
            log.info(f"Correction ref date (last train date): {today_ds_ref.date()}")
            today_df = pd.DataFrame({'ds': [today_ds_ref]})

            log.warning("Residual correction using CURRENT weather as proxy for historical ref date.")
            weather_api_query_city = target_city
            if target_city in CONFIG.get('modeling', {}).get('target_cities', []):
                  weather_api_query_city = f"{target_city}, India"
            current_weather = get_current_weather(weather_api_query_city)

            # Add ALL expected regressors to today_df using current weather or defaults
            if model_regressors:
                 if current_weather:
                     log.info("Mapping current weather for correction...")
                     for regressor in model_regressors:
                         # --- *** START: MAPPING LOGIC (Current Weather) - USER MUST VERIFY/EDIT *** ---
                         source_col = None
                         if regressor == 'temperature_2m':           source_col = 'temp_c'
                         elif regressor == 'relative_humidity_2m':   source_col = 'humidity'
                         elif regressor == 'wind_speed_10m':         source_col = 'wind_kph'
                         # Add mappings for rain, pressure, cloud_cover, gusts if they are in model_regressors
                         elif regressor == 'rain':                   source_col = None # Cannot map
                         elif regressor == 'pressure_msl':           source_col = 'pressure_mb'
                         elif regressor == 'cloud_cover':            source_col = None # Cannot map
                         elif regressor == 'wind_gusts_10m':         source_col = 'wind_kph' # Proxy
                         # --- *** END: MAPPING LOGIC (Current Weather) *** ---
                         value = 0.0 # Default value
                         if source_col and source_col in current_weather and pd.notna(current_weather.get(source_col)):
                              num_val = pd.to_numeric(current_weather.get(source_col), errors='coerce')
                              if pd.notna(num_val):
                                   value = num_val
                              else:
                                   log.warning(f"Current value for '{source_col}' (for {regressor}) not numeric. Using 0.0.")
                         else:
                              log.warning(f"No valid current value or mapping for regressor '{regressor}'. Using 0.0.")
                         today_df[regressor] = value
                 else: # If current weather fetch failed
                     log.warning("Cannot get current weather for regressors. Using 0.0 for all.")
                     for regressor in model_regressors: today_df[regressor] = 0.0

            # Ensure all needed columns exist before predict
            missing_pred_cols_today = set(['ds'] + model_regressors) - set(today_df.columns)
            if missing_pred_cols_today: raise PredictionError(f"today_df missing regressors: {missing_pred_cols_today}")
            if today_df[model_regressors].isnull().any().any():
                 nan_cols = today_df[model_regressors].isnull().sum(); raise PredictionError(f"NaNs found in regressors for correction ref date: {nan_cols[nan_cols > 0].to_dict()}")

            # Predict for reference date WITH regressors
            log.debug(f"DataFrame for correction prediction:\n{today_df}")
            # We need the prediction corresponding to today_ds_ref from the *original* training data history
            # Re-predicting just this point might be slightly different from history if model changed, but is necessary
            today_forecast = trained_model.predict(today_df[['ds'] + model_regressors]) # Pass df with ds + ALL regressors

            if not today_forecast.empty:
                predicted_today = today_forecast['yhat'].iloc[0]; log.info(f"Model's prediction for {today_ds_ref.date()}: {predicted_today:.2f}")
                actual_today = None
                if last_known_aqi is not None: actual_today = float(last_known_aqi); log.info(f"Using provided last_known_aqi: {actual_today}")
                else:
                    current_aqi_info = get_current_aqi_for_city(target_city)
                    if current_aqi_info and 'aqi' in current_aqi_info: actual_today = float(current_aqi_info['aqi']); log.info(f"Actual current AQI from API: {actual_today}")
                    else: log.warning(f"Could not get current AQI. Skipping correction.")
                if actual_today is not None:
                    base_residual = actual_today - predicted_today; log.info(f"Calculated base residual: {base_residual:.2f}")
                    applied_residuals = []; current_decay = 1.0; decay_factor = RESIDUAL_DECAY_FACTOR
                    for _ in range(len(forecast)): applied_residuals.append(base_residual * current_decay); current_decay *= decay_factor
                    if len(applied_residuals) == len(forecast):
                         forecast['residual'] = applied_residuals
                         forecast['yhat_adjusted'] = (forecast['yhat'] + forecast['residual']).clip(lower=0)
                         log.info(f"Applied decaying residual correction (factor={decay_factor}).")
                    else:
                         log.warning("Residual/Forecast len mismatch. Using constant residual."); forecast['residual'] = base_residual; forecast['yhat_adjusted'] = (forecast['yhat'] + base_residual).clip(lower=0)
            else: log.warning(f"Could not predict for ref date {today_ds_ref.date()}. Skipping correction."); forecast['residual'] = 0.0; forecast['yhat_adjusted'] = forecast['yhat'].copy()
        except PredictionError as pe:
             log.error(f"PredictionError during residual correction setup: {pe}. Proceeding without correction.")
             forecast['residual'] = 0.0; forecast['yhat_adjusted'] = forecast['yhat'].copy()
        except Exception as e:
            log.error(f"Error during residual correction: {e}. Proceeding without.", exc_info=True)
            forecast['residual'] = 0.0; forecast['yhat_adjusted'] = forecast['yhat'].copy()
    else: log.info("Residual correction not applied.")

    log.info(f"Forecast generation complete for {target_city}.")
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'residual', 'yhat_adjusted']]


# --- Helper Function for UI Formatting ---
# (Keep format_forecast_for_ui exactly as it was)
def format_forecast_for_ui(forecast_df):
    # ... (no changes) ...
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
    # ... (no changes) ...
    log.info(f"Getting predicted weekly risks (Section 6) for {city_name}...")
    forecast_df = None
    try:
        forecast_df = generate_forecast(target_city=city_name, days_ahead=days_ahead, apply_residual_correction=True)
    except PredictionError as pe: log.error(f"PredictionError getting forecast for weekly risks: {pe}")
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


print("DEBUG: predictor.py module fully parsed and loaded.")

# --- Example Usage Block (Restored) ---
if __name__ == "__main__":
    print("DEBUG: Entering __main__ block of predictor.py")
    # (Keep existing __main__ block test code, including the try/except around it)
    if not logging.getLogger().handlers:
         log_level = logging.INFO
         log_format = '%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s'
         try:
              log_level_str = CONFIG.get('logging', {}).get('level', 'INFO')
              log_format = CONFIG.get('logging', {}).get('format', log_format)
              log_level = getattr(logging, log_level_str.upper(), logging.INFO)
         except Exception: pass
         logging.basicConfig(level=log_level, format=log_format)
         log = logging.getLogger(__name__)
         log.info("Configured fallback logging for __main__ block.")
    print("\n" + "="*30); print(" Running predictor.py Tests "); print("="*30 + "\n")
    test_city = "Delhi"; days_to_forecast = CONFIG.get('modeling', {}).get('forecast_days', 3)
    log.info(f"Using forecast horizon: {days_to_forecast} days")
    forecast_api = None
    try:
        print(f"\n--- Test 1: Predicting {days_to_forecast} days for '{test_city}' WITH API Correction ---")
        forecast_api = generate_forecast(test_city, days_ahead=days_to_forecast, apply_residual_correction=True)
        if forecast_api is not None: print(forecast_api)
        else: print(f"Failed test 1 for {test_city}")
        simulated_aqi = 150.0
        print(f"\n--- Test 2: Predicting {days_to_forecast} days for '{test_city}' WITH Manual Correction (Last AQI={simulated_aqi}) ---")
        forecast_manual = generate_forecast(test_city, days_ahead=days_to_forecast, apply_residual_correction=True, last_known_aqi=simulated_aqi)
        if forecast_manual is not None: print(forecast_manual)
        else: print(f"Failed test 2 for {test_city}")
        print(f"\n--- Test 3: Predicting {days_to_forecast} days for '{test_city}' WITHOUT Correction ---")
        forecast_raw = generate_forecast(test_city, days_ahead=days_to_forecast, apply_residual_correction=False)
        if forecast_raw is not None: print(forecast_raw)
        else: print(f"Failed test 3 for {test_city}")
        print("\n--- Test 4: Formatting Forecast for UI ---")
        if forecast_api is not None:
            ui_formatted_data = format_forecast_for_ui(forecast_api)
            print(f"UI Formatted Data for {test_city} (API Corrected Forecast):"); import pprint; pprint.pprint(ui_formatted_data)
        else: print("Skipping test 4 (forecast_api is None).")
        print("\n--- Test 5: Getting Predicted Weekly Risks (Section 6) ---")
        print(f"--- Getting predicted weekly risks for {test_city} ---")
        weekly_risks = get_predicted_weekly_risks(test_city, days_ahead=days_to_forecast)
        if weekly_risks: import pprint; print("Predicted Weekly Risks/Implications:"); pprint.pprint(weekly_risks)
        else: print(f"Could not generate predicted weekly risks for {test_city}.")
        test_city_2 = "Mumbai"
        model_version_test = CONFIG.get('modeling', {}).get('prophet_model_version', 'v3_weather')
        models_dir_test = os.path.join(PROJECT_ROOT, CONFIG.get('paths', {}).get('models_dir', 'models'))
        model_exists = os.path.exists(os.path.join(models_dir_test, f"{test_city_2}_prophet_model_{model_version_test}.json"))
        if model_exists:
             print(f"\n--- Test 6: Predicting {days_to_forecast} days for '{test_city_2}' WITH API Correction ---")
             forecast_mumbai = generate_forecast(test_city_2, days_ahead=days_to_forecast, apply_residual_correction=True)
             if forecast_mumbai is not None: print(f"Forecast for {test_city_2}:\n{forecast_mumbai}")
             else: print(f"Failed test 6 for {test_city_2}.")
        else: print(f"\nSkipping Test 6: Model file for {test_city_2} version {model_version_test} not found.")
        print("\n--- Test 7: Predicting for 'Atlantis' (No Model) ---")
        forecast_none = generate_forecast("Atlantis", days_ahead=days_to_forecast)
        if forecast_none is None: print("Success! Correctly returned None.")
        else: print("Failure! Expected None.")
    except Exception as e: print(f"\n!!! An error occurred during the main test execution block: {e} !!!"); log.error("Error in predictor.py __main__ block", exc_info=True)
    print("\n" + "="*30); print(" predictor.py Tests Finished "); print("="*30 + "\n")