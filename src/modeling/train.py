# File: src/modeling/train.py

"""
Handles the training and saving of Prophet forecasting models for target cities.

This script loads the master historical AQI dataset, prepares the data for each
configured target city, trains an improved Prophet model (with specific seasonality
and prior settings), and saves the serialized model to the designated models directory.
Configuration values (data path, models path, target cities, model version) are
read from the central config file via config_loader.
"""

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
# (Keep existing import logic with fallbacks)
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

# --- Get Logger ---
log = logging.getLogger(__name__)

# --- Configuration Values ---
# (Keep logic for getting config values)
relative_data_path = CONFIG.get('paths', {}).get('data_file', 'data/Post-Processing/CSV_Files/Master_AQI_Dataset.csv')
relative_models_dir = CONFIG.get('paths', {}).get('models_dir', 'models')
TARGET_CITIES = CONFIG.get('modeling', {}).get('target_cities', ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Hyderabad'])
MODEL_VERSION = CONFIG.get('modeling', {}).get('prophet_model_version', 'v2')
DATA_PATH = os.path.join(PROJECT_ROOT, relative_data_path)
MODELS_DIR = os.path.join(PROJECT_ROOT, relative_models_dir)

# --- Helper Functions ---

def load_data(data_path=DATA_PATH):
    """Loads the master dataset and parses the 'Date' column.

    Args:
        data_path (str, optional): The absolute path to the dataset CSV file.
                                   Defaults to DATA_PATH derived from config.

    Returns:
        pandas.DataFrame or None: Loaded DataFrame with 'Date' as datetime objects,
                                  or None if loading/parsing fails.

    Raises:
        DataFileNotFoundError: If the file at data_path does not exist.
        ValueError: If date parsing fails due to incorrect format in the file.
        RuntimeError: For other unexpected file reading errors.
    """
    # (Function code remains the same)
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
        raise
    except FileNotFoundError:
        msg = f"Data file not found (exception): {data_path}"
        log.error(msg)
        raise DataFileNotFoundError(msg)
    except Exception as e:
        log.error(f"Failed during data loading: {e}", exc_info=True)
        raise RuntimeError(f"Failed during data loading: {e}") from e

def reconstruct_city_column(df):
    """Adds a 'City' column based on one-hot encoded columns if it doesn't exist.

    Iterates through columns starting with 'City_' and assigns the corresponding
    city name to a new 'City' column.

    Args:
        df (pd.DataFrame or None): The input DataFrame (potentially from load_data).

    Returns:
        pd.DataFrame: The DataFrame with the 'City' column added or confirmed.

    Raises:
        ValueError: If df is None or if no 'City_*' columns are found when needed.
    """
    # (Function code remains the same)
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
    """Filters data for a city, prepares 'ds'/'y' columns, handles NaNs.

    Args:
        df_master (pd.DataFrame or None): The master DataFrame containing data for all cities.
        city_name (str): The specific city to prepare data for.

    Returns:
        pd.DataFrame: A DataFrame ready for Prophet (columns 'ds', 'y'), sorted by date.

    Raises:
        ValueError: If df_master is None, no data is found for the city, or
                    missing AQI ('y') values cannot be filled.
    """
    # (Function code remains the same)
    if df_master is None:
        raise ValueError("Input DataFrame is None for Prophet preparation.")
    log.info(f"Preparing data for city: {city_name}")
    city_df = df_master[df_master['City'] == city_name].copy()
    if city_df.empty:
        msg = f"No data found for city: {city_name}"
        log.error(msg)
        raise ValueError(msg)
    prophet_df = city_df[['Date', 'AQI']].rename(columns={'Date': 'ds', 'AQI': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    initial_nan_count = prophet_df['y'].isnull().sum()
    if initial_nan_count > 0:
        log.warning(f"Found {initial_nan_count} missing AQI ('y') for {city_name}. Applying ffill/bfill.")
        prophet_df['y'] = prophet_df['y'].ffill().bfill()
        if prophet_df['y'].isnull().any():
            msg = f"Could not fill all missing AQI for {city_name}."
            log.error(msg)
            raise ValueError(msg)
    prophet_df = prophet_df.sort_values(by='ds')
    log.info(f"Data prepared for {city_name}. Shape: {prophet_df.shape}")
    return prophet_df

def train_prophet_model(city_data_df, city_name):
    """Instantiates, configures, and trains the final Prophet model for a city.

    Uses Multiplicative Seasonality and Default Prior Scales based on notebook
    evaluation. Adds Indian holidays. Trains on the provided city-specific data.

    Args:
        city_data_df (pd.DataFrame or None): The prepared DataFrame ('ds', 'y') for the city.
        city_name (str): The name of the city (for logging).

    Returns:
        Prophet: The fitted Prophet model instance.

    Raises:
        ValueError: If city_data_df is None.
        RuntimeError: If the model fitting process fails.
    """
    if city_data_df is None:
         raise ValueError("Input city_data_df is None for training.")

    log.info(f"Instantiating FINAL Prophet model for {city_name} (Multiplicative, Default Priors)...")
    # Parameters chosen after notebook evaluation (Cell 3B was best)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative', # FINAL CHOICE
        changepoint_prior_scale=0.05,    # FINAL CHOICE (Default)
        seasonality_prior_scale=10.0,    # FINAL CHOICE (Default)
        holidays_prior_scale=10.0        # FINAL CHOICE (Default)
    )

    # Add holidays
    try:
        model.add_country_holidays(country_name='IN')
        log.info(f"Added India holidays for {city_name}.")
    except Exception as e:
         # Log warning but continue training even if holidays fail
         log.warning(f"Could not add country holidays for 'IN' for {city_name}. Error: {e}")

    # Fit model
    log.info(f"Fitting model to data for {city_name} (using all historical data)...")
    try:
        model.fit(city_data_df)
        log.info(f"Model fitting complete for {city_name}.")
        return model
    except Exception as e:
        # Log error and raise a specific runtime error if fitting fails
        log.error(f"Error during model fitting for {city_name}: {e}", exc_info=True)
        raise RuntimeError(f"Error fitting Prophet model for {city_name}: {e}") from e

def save_model(model, city_name, version=MODEL_VERSION, models_dir=MODELS_DIR):
    """Saves the trained Prophet model to a JSON file using Prophet serialization.

    Constructs the filename using city name and version. Creates the models
    directory if it doesn't exist.

    Args:
        model (Prophet or None): The fitted Prophet model instance to save.
        city_name (str): The city name used for the filename.
        version (str, optional): The model version suffix for the filename.
                                 Defaults to MODEL_VERSION from config.
        models_dir (str, optional): The directory path to save the model file.
                                    Defaults to MODELS_DIR from config.

    Returns:
        bool: True if saving was successful.

    Raises:
        ValueError: If the input model is None.
        RuntimeError: If an error occurs during file writing or serialization.
    """
    # (Function code remains the same)
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
        return True
    except Exception as e:
        log.error(f"Error saving model for {city_name}: {e}", exc_info=True)
        raise RuntimeError(f"Error saving model for {city_name}: {e}") from e

# --- Main Execution ---
def main():
    """Main function to orchestrate the model training process for all target cities.

    Loads data, iterates through cities specified in the config, prepares data,
    trains the model, and saves the model for each city. Logs a summary at the end.
    Catches errors during processing for individual cities to allow the script
    to continue with other cities.
    """
    # (Function code remains the same)
    log.info("Starting model training process using config...")
    log.info(f"Target cities from config: {TARGET_CITIES}")
    log.info(f"Model version from config: {MODEL_VERSION}")

    df_master = None
    try:
        df_master = load_data()
        df_master = reconstruct_city_column(df_master)
    except (DataFileNotFoundError, ValueError, RuntimeError) as e:
         log.critical(f"Failed to load or initially process master data: {e}. Exiting training.")
         return
    except Exception as e:
         log.critical(f"Unexpected error loading/processing master data: {e}", exc_info=True)
         return

    successful_cities = []
    failed_cities = []

    for city in TARGET_CITIES:
        log.info(f"\n===== Processing City: {city} =====")
        try:
            prophet_df = prepare_prophet_data(df_master, city)
            trained_model = train_prophet_model(prophet_df, city)
            save_model(trained_model, city)
            successful_cities.append(city)
        except (ValueError, RuntimeError) as e:
            log.error(f"Failed processing city {city}: {e}")
            failed_cities.append(city)
        except Exception as e:
            log.error(f"Unexpected error processing city {city}: {e}", exc_info=True)
            failed_cities.append(city)

    log.info("\n===== Model Training Summary =====")
    log.info(f"Successfully trained and saved models for: {successful_cities}")
    if failed_cities: log.warning(f"Failed processing or saving models for: {failed_cities}")
    log.info("Model training process finished.")

if __name__ == "__main__":
    main()