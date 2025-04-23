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
    logging.warning(f"Train Script: __file__ not defined. Assuming project root: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)
     logging.info(f"Train Script: Added project root to sys.path: {PROJECT_ROOT}")

# --- Import Configuration ---
# This import also sets up logging via config_loader.py
try:
    from src.config_loader import CONFIG
    logging.info("Train Script: Successfully imported config.")
except ImportError as e:
    # Basic logging if config fails
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s')
    logging.error(f"Train Script: Could not import CONFIG. Using defaults. Error: {e}", exc_info=True)
    CONFIG = {} # Define as empty dict
except Exception as e:
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s')
     logging.error(f"Train Script: Error importing config: {e}")
     CONFIG = {}

# --- Get Logger ---
# Get the logger instance for this module, configured by config_loader
log = logging.getLogger(__name__)

# --- Configuration Values ---
# Get paths and parameters from config, providing defaults if config loading failed or key missing
relative_data_path = CONFIG.get('paths', {}).get('data_file', 'data/Post-Processing/CSV_Files/Master_AQI_Dataset.csv')
relative_models_dir = CONFIG.get('paths', {}).get('models_dir', 'models')
TARGET_CITIES = CONFIG.get('modeling', {}).get('target_cities', ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Hyderabad'])
MODEL_VERSION = CONFIG.get('modeling', {}).get('prophet_model_version', 'v2')

# Construct absolute paths
DATA_PATH = os.path.join(PROJECT_ROOT, relative_data_path)
MODELS_DIR = os.path.join(PROJECT_ROOT, relative_models_dir)


# --- Helper Functions ---

def load_data(data_path=DATA_PATH): # Use module-level absolute path
    """Loads the master dataset and parses dates."""
    log.info(f"Loading data from: {data_path}")
    # (Keep rest of function logic as is)
    if not os.path.exists(data_path):
        log.error(f"Data file not found: {data_path}")
        return None
    try:
        df = pd.read_csv(data_path)
        date_format = '%d/%m/%y'
        log.info(f"Attempting to parse 'Date' column with format: {date_format}")
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
        log.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except ValueError as ve:
        log.error(f"Error parsing 'Date' column with format '{date_format}'. Check CSV. Error: {ve}")
        return None
    except Exception as e:
        log.error(f"Failed during data loading: {e}", exc_info=True)
        return None


def reconstruct_city_column(df):
    """Adds a 'City' column based on one-hot encoded columns if it doesn't exist."""
    # (Keep function logic as is)
    if df is None: return None
    if 'City' in df.columns:
        log.info("'City' column already exists.")
        return df
    log.info("Reconstructing 'City' column...")
    city_columns = [col for col in df.columns if col.startswith('City_')]
    if not city_columns:
        log.error("CRITICAL: No 'City_' columns found.")
        return None
    def get_city_name(row):
        for city_col in city_columns:
            if row[city_col] > 0: return city_col.replace('City_', '')
        return 'Unknown'
    df['City'] = df.apply(get_city_name, axis=1)
    log.info("Finished reconstructing 'City' column.")
    return df

def prepare_prophet_data(df_master, city_name):
    """Filters data for a city, prepares 'ds' and 'y' columns, handles NaNs."""
    # (Keep function logic as is)
    if df_master is None: return None
    log.info(f"Preparing data for city: {city_name}")
    city_df = df_master[df_master['City'] == city_name].copy()
    if city_df.empty:
        log.error(f"No data found for city: {city_name}")
        return None
    prophet_df = city_df[['Date', 'AQI']].rename(columns={'Date': 'ds', 'AQI': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    initial_nan_count = prophet_df['y'].isnull().sum()
    if initial_nan_count > 0:
        log.warning(f"Found {initial_nan_count} missing AQI ('y') for {city_name}. Applying ffill/bfill.")
        prophet_df['y'] = prophet_df['y'].ffill().bfill()
        if prophet_df['y'].isnull().any():
            log.error(f"Could not fill all missing AQI for {city_name}.")
            return None
    prophet_df = prophet_df.sort_values(by='ds')
    log.info(f"Data prepared for {city_name}. Shape: {prophet_df.shape}")
    return prophet_df

def train_prophet_model(city_data_df, city_name):
    """Instantiates, configures, and trains the improved Prophet model."""
    # (Keep function logic as is)
    if city_data_df is None: return None
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
        return None

def save_model(model, city_name, version=MODEL_VERSION, models_dir=MODELS_DIR): # Use module-level config vars
    """Saves the trained Prophet model to a JSON file."""
    # (Keep function logic as is)
    if model is None: return False
    os.makedirs(models_dir, exist_ok=True)
    model_filename = f"{city_name}_prophet_model_{version}.json"
    model_path = os.path.join(models_dir, model_filename)
    log.info(f"Saving model for {city_name} (v{version}) to JSON: {model_path}")
    try:
        with open(model_path, 'w') as fout:
            json.dump(model_to_json(model), fout)
        log.info(f"Model for {city_name} saved successfully.")
        return True
    except Exception as e:
        log.error(f"Error saving model for {city_name}: {e}", exc_info=True)
        return False

# --- Main Execution ---
def main():
    """Loads data, loops through configured cities, trains, and saves models."""
    log.info("Starting model training process using config...")
    log.info(f"Target cities from config: {TARGET_CITIES}") # Use config variable
    log.info(f"Model version from config: {MODEL_VERSION}") # Use config variable

    df_master = load_data() # Uses global DATA_PATH
    if df_master is None:
        log.critical("Failed to load master data. Exiting training.")
        return

    df_master = reconstruct_city_column(df_master)
    if df_master is None:
        log.critical("Failed to reconstruct city column. Exiting training.")
        return

    successful_cities = []
    failed_cities = []

    for city in TARGET_CITIES: # Use config variable
        log.info(f"\n===== Processing City: {city} =====")
        prophet_df = prepare_prophet_data(df_master, city)
        if prophet_df is None:
            log.error(f"Skipping city {city}: data preparation error.")
            failed_cities.append(city)
            continue
        trained_model = train_prophet_model(prophet_df, city)
        if trained_model is None:
            log.error(f"Skipping city {city}: model training error.")
            failed_cities.append(city)
            continue
        # Pass MODEL_VERSION and MODELS_DIR which are now module-level vars from config
        save_success = save_model(trained_model, city) # Uses defaults
        if save_success: successful_cities.append(city)
        else: failed_cities.append(city)

    log.info("\n===== Model Training Summary =====")
    log.info(f"Successfully trained and saved models for: {successful_cities}")
    if failed_cities: log.warning(f"Failed to train or save models for: {failed_cities}")
    log.info("Model training process finished.")

if __name__ == "__main__":
    main() # Execute the main function when script is run directly