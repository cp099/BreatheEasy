# File: src/modeling/train.py

import pandas as pd
import numpy as np
import os
import sys
import logging # Standard logging import
import json
from prophet import Prophet
from prophet.serialize import model_to_json

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

# --- Import Configuration & Exceptions ---
# This import also sets up logging via config_loader.py
try:
    from src.config_loader import CONFIG
    from src.exceptions import DataFileNotFoundError # Import relevant custom exception
    # Logging configured by config_loader
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"Train Script: Could not import dependencies. Error: {e}", exc_info=True)
    CONFIG = {}
    class DataFileNotFoundError(FileNotFoundError): pass # Dummy
except Exception as e:
     logging.basicConfig(level=logging.WARNING)
     logging.error(f"Train Script: Error importing dependencies: {e}")
     CONFIG = {}
     class DataFileNotFoundError(FileNotFoundError): pass

# --- Get Logger ---
log = logging.getLogger(__name__)

# --- Configuration Values ---
# (Keep logic for getting config values: relative_data_path, etc.)
relative_data_path = CONFIG.get('paths', {}).get('data_file', 'data/Post-Processing/CSV_Files/Master_AQI_Dataset.csv')
relative_models_dir = CONFIG.get('paths', {}).get('models_dir', 'models')
TARGET_CITIES = CONFIG.get('modeling', {}).get('target_cities', ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Hyderabad'])
MODEL_VERSION = CONFIG.get('modeling', {}).get('prophet_model_version', 'v2')
DATA_PATH = os.path.join(PROJECT_ROOT, relative_data_path)
MODELS_DIR = os.path.join(PROJECT_ROOT, relative_models_dir)

# --- Helper Functions ---

def load_data(data_path=DATA_PATH):
    """Loads dataset, parses dates. Raises exceptions on failure."""
    log.info(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        msg = f"Data file not found: {data_path}"
        log.error(msg)
        raise DataFileNotFoundError(msg)
    try:
        df = pd.read_csv(data_path)
        date_format = '%d/%m/%y'
        log.info(f"Parsing 'Date' column with format: {date_format}")
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
        log.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except ValueError as ve:
        log.error(f"Error parsing 'Date' column with format '{date_format}'. Check CSV. Error: {ve}")
        raise # Re-raise ValueError
    except FileNotFoundError: # Should be caught above, but belt-and-suspenders
        msg = f"Data file not found (exception): {data_path}"
        log.error(msg)
        raise DataFileNotFoundError(msg)
    except Exception as e:
        log.error(f"Failed during data loading: {e}", exc_info=True)
        raise RuntimeError(f"Failed during data loading: {e}") from e # Wrap other errors


def reconstruct_city_column(df):
    """Adds 'City' column. Raises ValueError if one-hot cols missing."""
    if df is None:
         raise ValueError("Input DataFrame is None for city reconstruction.")
    if 'City' in df.columns:
        log.info("'City' column already exists.")
        return df
    log.info("Reconstructing 'City' column...")
    city_columns = [col for col in df.columns if col.startswith('City_')]
    if not city_columns:
        msg = "CRITICAL: No 'City_' columns found for reconstruction."
        log.error(msg)
        raise ValueError(msg)
    def get_city_name(row):
        for city_col in city_columns:
            if row[city_col] > 0: return city_col.replace('City_', '')
        return 'Unknown'
    df['City'] = df.apply(get_city_name, axis=1)
    log.info("Finished reconstructing 'City' column.")
    return df

def prepare_prophet_data(df_master, city_name):
    """Prepares data for Prophet. Raises ValueError on critical failure."""
    if df_master is None:
        raise ValueError("Input DataFrame is None for Prophet preparation.")
    log.info(f"Preparing data for city: {city_name}")
    city_df = df_master[df_master['City'] == city_name].copy()
    if city_df.empty:
        msg = f"No data found for city: {city_name}"
        log.error(msg)
        raise ValueError(msg) # Raise error if no data for city
    prophet_df = city_df[['Date', 'AQI']].rename(columns={'Date': 'ds', 'AQI': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    initial_nan_count = prophet_df['y'].isnull().sum()
    if initial_nan_count > 0:
        log.warning(f"Found {initial_nan_count} missing AQI ('y') for {city_name}. Applying ffill/bfill.")
        prophet_df['y'] = prophet_df['y'].ffill().bfill()
        if prophet_df['y'].isnull().any():
            msg = f"Could not fill all missing AQI for {city_name}."
            log.error(msg)
            raise ValueError(msg) # Raise if NaNs persist
    prophet_df = prophet_df.sort_values(by='ds')
    log.info(f"Data prepared for {city_name}. Shape: {prophet_df.shape}")
    return prophet_df

def train_prophet_model(city_data_df, city_name):
    """Trains Prophet model. Raises RuntimeError on fitting failure."""
    if city_data_df is None:
         raise ValueError("Input city_data_df is None for training.")
    log.info(f"Instantiating Improved Prophet model for {city_name}...")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    seasonality_mode='multiplicative', changepoint_prior_scale=0.1, seasonality_prior_scale=10.0)
    try:
        model.add_country_holidays(country_name='IN')
        log.info(f"Added India holidays for {city_name}.")
    except Exception as e:
         log.warning(f"Could not add country holidays for 'IN'. Error: {e}")
    log.info(f"Fitting model to data for {city_name} (using all historical data)...")
    try:
        model.fit(city_data_df)
        log.info(f"Model fitting complete for {city_name}.")
        return model
    except Exception as e:
        log.error(f"Error during model fitting for {city_name}: {e}", exc_info=True)
        raise RuntimeError(f"Error fitting model for {city_name}: {e}") from e # Raise error

def save_model(model, city_name, version=MODEL_VERSION, models_dir=MODELS_DIR):
    """Saves the model. Raises RuntimeError on failure."""
    if model is None:
        raise ValueError("Cannot save a None model object.")
    os.makedirs(models_dir, exist_ok=True)
    model_filename = f"{city_name}_prophet_model_{version}.json"
    model_path = os.path.join(models_dir, model_filename)
    log.info(f"Saving model for {city_name} (v{version}) to JSON: {model_path}")
    try:
        with open(model_path, 'w') as fout:
            json.dump(model_to_json(model), fout)
        log.info(f"Model for {city_name} saved successfully.")
        # Return True on success if needed by caller, but raising on error is primary
        return True
    except Exception as e:
        log.error(f"Error saving model for {city_name}: {e}", exc_info=True)
        raise RuntimeError(f"Error saving model for {city_name}: {e}") from e # Raise error

# --- Main Execution ---
def main():
    """Loads data, loops through configured cities, trains, and saves models."""
    log.info("Starting model training process using config...")
    log.info(f"Target cities from config: {TARGET_CITIES}")
    log.info(f"Model version from config: {MODEL_VERSION}")

    df_master = None # Initialize
    try:
        df_master = load_data()
        df_master = reconstruct_city_column(df_master)
    except (DataFileNotFoundError, ValueError, RuntimeError) as e:
         log.critical(f"Failed to load or initially process master data: {e}. Exiting training.")
         return # Exit if basic data loading fails
    except Exception as e: # Catch any other unexpected error during load/reconstruct
         log.critical(f"Unexpected error loading/processing master data: {e}", exc_info=True)
         return

    successful_cities = []
    failed_cities = []

    for city in TARGET_CITIES:
        log.info(f"\n===== Processing City: {city} =====")
        try:
            # Chain the processing steps for clarity
            prophet_df = prepare_prophet_data(df_master, city)
            trained_model = train_prophet_model(prophet_df, city)
            save_model(trained_model, city) # Uses configured version/dir implicitly
            successful_cities.append(city) # Add to success list only if all steps pass
        except (ValueError, RuntimeError) as e:
            # Catch errors specific to this city's processing/training/saving
            log.error(f"Failed processing city {city}: {e}")
            failed_cities.append(city)
        except Exception as e: # Catch any other unexpected error for this city
            log.error(f"Unexpected error processing city {city}: {e}", exc_info=True)
            failed_cities.append(city)

    log.info("\n===== Model Training Summary =====")
    log.info(f"Successfully trained and saved models for: {successful_cities}")
    if failed_cities: log.warning(f"Failed processing or saving models for: {failed_cities}")
    log.info("Model training process finished.")

if __name__ == "__main__":
    main()