# File: src/modeling/train.py

import pandas as pd
import numpy as np
import os
import sys
import logging
import json
from prophet import Prophet
from prophet.serialize import model_to_json

# --- Setup Project Root Path ---
# Add project root to sys.path to allow importing project modules
try:
    # Assumes this script is in src/modeling/
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    logging.info(f"Project root determined using __file__: {PROJECT_ROOT}")
except NameError:
    # Fallback if __file__ is not defined (e.g., different execution context)
    # Assume the current working directory IS the project root BREATHEEASY/
    PROJECT_ROOT = os.path.abspath('.')
    logging.warning(f"__file__ not defined. Assuming current working directory is project root: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)
     logging.info(f"Added project root to sys.path: {PROJECT_ROOT}")

# Now import project modules if needed (none needed directly for training itself)
# from src.some_module import some_function # Example

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s')

# --- Configuration ---
DATA_FILENAME = 'Master_AQI_Dataset.csv'
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Post-Processing', 'CSV_Files', DATA_FILENAME)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
TARGET_CITIES = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Hyderabad']
MODEL_VERSION = "v2" # Version indicator for saved models

# --- Helper Functions ---

def load_data(data_path):
    """Loads the master dataset and parses dates."""
    logging.info(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        logging.error(f"Data file not found: {data_path}")
        return None
    try:
        df = pd.read_csv(data_path)
        # Explicit Date Parsing using correct format
        date_format = '%d/%m/%y'
        logging.info(f"Attempting to parse 'Date' column with format: {date_format}")
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except ValueError as ve:
        logging.error(f"Error parsing 'Date' column with format '{date_format}'. Check CSV. Error: {ve}")
        return None
    except Exception as e:
        logging.error(f"Failed during data loading: {e}", exc_info=True)
        return None

def reconstruct_city_column(df):
    """Adds a 'City' column based on one-hot encoded columns if it doesn't exist."""
    if 'City' in df.columns:
        logging.info("'City' column already exists.")
        return df

    logging.info("Reconstructing 'City' column from City_... columns...")
    city_columns = [col for col in df.columns if col.startswith('City_')]
    if not city_columns:
        logging.error("CRITICAL: No 'City_' columns found for reconstruction.")
        return None # Cannot proceed without city info

    def get_city_name(row):
        for city_col in city_columns:
            if row[city_col] > 0:
                return city_col.replace('City_', '')
        return 'Unknown' # Should not happen with proper one-hot encoding

    df['City'] = df.apply(get_city_name, axis=1)
    logging.info("Finished reconstructing 'City' column.")
    return df

def prepare_prophet_data(df_master, city_name):
    """Filters data for a city, prepares 'ds' and 'y' columns, handles NaNs."""
    logging.info(f"Preparing data for city: {city_name}")
    city_df = df_master[df_master['City'] == city_name].copy()

    if city_df.empty:
        logging.error(f"No data found for city: {city_name}")
        return None

    prophet_df = city_df[['Date', 'AQI']].rename(columns={'Date': 'ds', 'AQI': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds']) # Ensure datetime

    # Handle missing values in 'y'
    initial_nan_count = prophet_df['y'].isnull().sum()
    if initial_nan_count > 0:
        logging.warning(f"Found {initial_nan_count} missing AQI values ('y') for {city_name}. Applying ffill then bfill.")
        prophet_df['y'] = prophet_df['y'].ffill().bfill()
        if prophet_df['y'].isnull().any():
            logging.error(f"Could not fill all missing AQI values for {city_name}. Check data source.")
            return None # Cannot train with NaNs

    prophet_df = prophet_df.sort_values(by='ds')
    logging.info(f"Data prepared for {city_name}. Shape: {prophet_df.shape}")
    return prophet_df

def train_prophet_model(city_data_df, city_name):
    """Instantiates, configures, and trains the improved Prophet model."""
    logging.info(f"Instantiating Improved Prophet model for {city_name}...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10.0
    )
    try:
        model.add_country_holidays(country_name='IN')
        logging.info(f"Added India holidays for {city_name}.")
    except Exception as e:
         logging.warning(f"Could not add country holidays for 'IN'. Error: {e}")

    logging.info(f"Fitting model to data for {city_name} (using all historical data)...")
    try:
        # Train on ALL available historical data for the city for best future prediction
        model.fit(city_data_df)
        logging.info(f"Model fitting complete for {city_name}.")
        return model
    except Exception as e:
        logging.error(f"Error during model fitting for {city_name}: {e}", exc_info=True)
        return None

def save_model(model, city_name, version, models_dir):
    """Saves the trained Prophet model to a JSON file."""
    os.makedirs(models_dir, exist_ok=True) # Ensure directory exists
    model_filename = f"{city_name}_prophet_model_{version}.json"
    model_path = os.path.join(models_dir, model_filename)

    logging.info(f"Saving model for {city_name} to JSON: {model_path}")
    try:
        with open(model_path, 'w') as fout:
            json.dump(model_to_json(model), fout)
        logging.info(f"Model for {city_name} saved successfully.")
        return True
    except Exception as e:
        logging.error(f"Error saving model for {city_name}: {e}", exc_info=True)
        return False

# --- Main Execution ---

def main():
    """Loads data, loops through cities, trains, and saves models."""
    logging.info("Starting model training process...")

    df_master = load_data(DATA_PATH)
    if df_master is None:
        logging.critical("Failed to load master data. Exiting training.")
        return

    df_master = reconstruct_city_column(df_master)
    if df_master is None:
        logging.critical("Failed to reconstruct city column. Exiting training.")
        return

    successful_cities = []
    failed_cities = []

    for city in TARGET_CITIES:
        logging.info(f"\n===== Processing City: {city} =====")

        # 1. Prepare data for the current city
        prophet_df = prepare_prophet_data(df_master, city)
        if prophet_df is None:
            logging.error(f"Skipping city {city} due to data preparation error.")
            failed_cities.append(city)
            continue

        # 2. Train the model
        trained_model = train_prophet_model(prophet_df, city)
        if trained_model is None:
            logging.error(f"Skipping city {city} due to model training error.")
            failed_cities.append(city)
            continue

        # 3. Save the model
        save_success = save_model(trained_model, city, MODEL_VERSION, MODELS_DIR)
        if save_success:
            successful_cities.append(city)
        else:
            failed_cities.append(city)

    logging.info("\n===== Model Training Summary =====")
    logging.info(f"Successfully trained and saved models for: {successful_cities}")
    if failed_cities:
        logging.warning(f"Failed to train or save models for: {failed_cities}")
    logging.info("Model training process finished.")


if __name__ == "__main__":
    main() # Execute the main function when script is run directly