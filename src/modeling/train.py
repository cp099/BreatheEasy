# File: src/modeling/train.py (v3 - Weather Regressors - Simplified Data Prep)

import pandas as pd
import numpy as np
import os
import sys
import logging # Standard logging import
import json
from prophet import Prophet
from prophet.serialize import model_to_json

# --- Setup Path & Import Config/Exceptions ---
# (Keep existing setup logic)
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
    from src.exceptions import DataFileNotFoundError
    # Logging configured by config_loader
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"Train Script: Could not import dependencies. Error: {e}", exc_info=True)
    CONFIG = {}
    class DataFileNotFoundError(FileNotFoundError): pass
except Exception as e:
     logging.basicConfig(level=logging.WARNING)
     logging.error(f"Train Script: Error importing dependencies: {e}")
     CONFIG = {}
     class DataFileNotFoundError(FileNotFoundError): pass

log = logging.getLogger(__name__)

# --- Configuration Values ---
# (Keep existing config loading logic)
relative_data_path = CONFIG.get('paths', {}).get('data_file', 'data/Post-Processing/CSV_Files/Master_AQI_Dataset.csv')
relative_models_dir = CONFIG.get('paths', {}).get('models_dir', 'models')
TARGET_CITIES = CONFIG.get('modeling', {}).get('target_cities', [])
MODEL_VERSION = CONFIG.get('modeling', {}).get('prophet_model_version', 'v3_weather')
WEATHER_REGRESSORS = CONFIG.get('modeling', {}).get('weather_regressors', [])

DATA_PATH = os.path.join(PROJECT_ROOT, relative_data_path)
MODELS_DIR = os.path.join(PROJECT_ROOT, relative_models_dir)

# --- Helper Functions ---

def load_data(data_path=DATA_PATH):
    """Loads dataset, parses dates, ensures numeric regressors. Raises exceptions."""
    log.info(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        msg = f"Data file not found: {data_path}"
        log.error(msg); raise DataFileNotFoundError(msg)
    try:
        df = pd.read_csv(data_path)
        date_format = '%d/%m/%y' # Assumes dates are still DD/MM/YY
        log.info(f"Parsing 'Date' column with format: {date_format}")
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
        log.info(f"Data loaded successfully. Shape: {df.shape}")
        log.info(f"Ensuring numeric types for specified regressors: {WEATHER_REGRESSORS}")
        for col in WEATHER_REGRESSORS:
             if col in df.columns:
                  df[col] = pd.to_numeric(df[col], errors='coerce')
                  # Optional: Check for NaNs introduced by coercion here if needed
                  if df[col].isnull().any():
                       log.warning(f"Column '{col}' contains non-numeric values after coercion.")
             else:
                  log.warning(f"Configured regressor column '{col}' not found in dataset.")
        return df
    except ValueError as ve:
        log.error(f"Error parsing 'Date' column with format '{date_format}'. Check CSV. Error: {ve}"); raise
    except Exception as e:
        log.error(f"Failed during data loading or initial processing: {e}", exc_info=True)
        raise RuntimeError(f"Failed during data loading: {e}") from e

def reconstruct_city_column(df):
    """Adds 'City' column if needed. Raises ValueError."""
    # (Keep exact logic)
    if df is None: raise ValueError("Input DataFrame is None.")
    if 'City' in df.columns: return df
    log.info("Reconstructing 'City' column...")
    city_columns = [col for col in df.columns if col.startswith('City_')]
    if not city_columns: raise ValueError("CRITICAL: No 'City_*' columns found.")
    def get_city_name(row):
        for city_col in city_columns:
            if row[city_col] > 0: return city_col.replace('City_', '')
        return 'Unknown'
    df['City'] = df.apply(get_city_name, axis=1)
    log.info("Finished reconstructing 'City' column.")
    return df

def prepare_prophet_data_with_regressors(df_master, city_name, regressors):
    """
    Filters data, prepares 'ds','y' & regressors. Assumes input df has no NaN rows
    for the relevant period after filtering by city. Raises ValueError on critical failure.
    """
    if df_master is None: raise ValueError("Input DataFrame is None.")
    log.info(f"Preparing data with regressors for city: {city_name}")
    city_df = df_master[df_master['City'] == city_name].copy()
    if city_df.empty: raise ValueError(f"No data found for city: {city_name} after initial filter.")

    # Check which requested regressors are actually present
    available_regressors = [col for col in regressors if col in city_df.columns]
    missing_config_regressors = set(regressors) - set(available_regressors)
    if missing_config_regressors:
         log.warning(f"Configured regressors not found for {city_name}: {missing_config_regressors}. Using only: {available_regressors}")

    required_cols = ['Date', 'AQI'] + available_regressors
    # Verify all needed columns exist *before* selecting
    missing_data_cols = set(required_cols) - set(city_df.columns)
    if missing_data_cols:
         raise ValueError(f"Required columns {missing_data_cols} missing from data for {city_name}.")

    prophet_df = city_df[required_cols].rename(columns={'Date': 'ds', 'AQI': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    # *** Simplified NaN Check ***
    # Check for NaNs in the final selected columns AFTER filtering for the city
    # This assumes the input CSV was cleaned beforehand
    check_nan_cols = ['y'] + available_regressors
    if prophet_df[check_nan_cols].isnull().any().any():
        nan_counts = prophet_df[check_nan_cols].isnull().sum()
        log.error(f"NaNs found in critical columns for {city_name}: {nan_counts[nan_counts > 0].to_dict()}. Cannot train.")
        raise ValueError(f"NaNs found in data for {city_name} - clean the source CSV first.")

    prophet_df = prophet_df.sort_values(by='ds')
    log.info(f"Data prepared for {city_name} using regressors {available_regressors}. Shape: {prophet_df.shape}. Date range: {prophet_df['ds'].min().date()} to {prophet_df['ds'].max().date()}")
    # Return the dataframe and the list of regressors actually used
    return prophet_df, available_regressors

def train_prophet_model_with_regressors(city_data_df, city_name, regressors_to_add):
    """Instantiates, adds regressors, configures, and trains Prophet model."""
    # (Keep exact logic from previous version)
    if city_data_df is None: raise ValueError("Input city_data_df is None.")
    log.info(f"Instantiating FINAL Prophet model for {city_name} (Multiplicative, Default Priors)...")
    model = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0, holidays_prior_scale=10.0)
    if regressors_to_add:
         log.info(f"Adding {len(regressors_to_add)} regressors: {regressors_to_add}")
         for regressor in regressors_to_add:
             if regressor in city_data_df.columns: model.add_regressor(regressor)
             else: log.error(f"Attempted add regressor '{regressor}' not in df for {city_name}. Skipping.")
    else: log.info(f"No regressors to add for {city_name}.")
    try: model.add_country_holidays(country_name='IN')
    except Exception as e: log.warning(f"Could not add holidays for {city_name}: {e}")
    log.info(f"Fitting model with regressors for {city_name}...")
    try:
        model.fit(city_data_df)
        log.info(f"Model fitting complete for {city_name}.")
        return model
    except Exception as e:
        log.error(f"Error during model fitting for {city_name}: {e}", exc_info=True)
        raise RuntimeError(f"Error fitting Prophet model for {city_name}: {e}") from e

def save_model(model, city_name, version=MODEL_VERSION, models_dir=MODELS_DIR):
    """Saves the model. Raises RuntimeError on failure."""
    # (Keep exact logic from previous version)
    if model is None: raise ValueError("Cannot save None model.")
    os.makedirs(models_dir, exist_ok=True)
    model_filename = f"{city_name}_prophet_model_{version}.json"
    model_path = os.path.join(models_dir, model_filename)
    log.info(f"Saving model for {city_name} (v{version}) to JSON: {model_path}")
    try:
        with open(model_path, 'w') as fout: json.dump(model_to_json(model), fout)
        log.info(f"Model for {city_name} saved successfully.")
        return True
    except Exception as e:
        log.error(f"Error saving model for {city_name}: {e}", exc_info=True)
        raise RuntimeError(f"Error saving model for {city_name}: {e}") from e

# --- Main Execution ---
def main():
    """Loads data, loops through cities, trains models with weather, saves them."""
    # (Keep exact logic from previous version)
    log.info("Starting model training process with Weather Regressors...")
    log.info(f"Target cities from config: {TARGET_CITIES}")
    log.info(f"Model version: {MODEL_VERSION}")
    log.info(f"Requested weather regressors from config: {WEATHER_REGRESSORS}")
    df_master = None
    try:
        df_master = load_data()
        df_master = reconstruct_city_column(df_master)
    except (DataFileNotFoundError, ValueError, RuntimeError) as e: log.critical(f"Failed load/process: {e}. Exiting."); return
    except Exception as e: log.critical(f"Unexpected error loading: {e}", exc_info=True); return

    successful_cities = []
    failed_cities = []
    for city in TARGET_CITIES:
        log.info(f"\n===== Processing City: {city} =====")
        try:
            prophet_df, actual_regressors_used = prepare_prophet_data_with_regressors(df_master, city, WEATHER_REGRESSORS)
            trained_model = train_prophet_model_with_regressors(prophet_df, city, actual_regressors_used)
            save_model(trained_model, city)
            successful_cities.append(city)
        except (ValueError, RuntimeError) as e: log.error(f"Failed processing city {city}: {e}"); failed_cities.append(city)
        except Exception as e: log.error(f"Unexpected error processing city {city}: {e}", exc_info=True); failed_cities.append(city)

    log.info("\n===== Model Training Summary =====")
    log.info(f"Successfully trained/saved models for: {successful_cities}")
    if failed_cities: log.warning(f"Failed processing/saving models for: {failed_cities}")
    log.info("Model training process finished.")

if __name__ == "__main__":
    main()