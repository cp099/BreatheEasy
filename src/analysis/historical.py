# File: src/analysis/historical.py

"""
Provides functions for loading, preprocessing, and accessing historical AQI data.

This module handles the loading of the master dataset specified in the config,
performs necessary preprocessing like date parsing and city column reconstruction,
caches the loaded data in memory, and offers functions to retrieve specific
data slices (e.g., by city) for analysis or display.
"""

import pandas as pd
import os
import logging # Standard library logging
import sys # For path manipulation

# --- Setup Project Root Path ---
# (Keep existing PROJECT_ROOT logic)
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')):
         PROJECT_ROOT = os.path.abspath('.')
    # No logging here before config is loaded
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import Configuration & Exceptions ---
# This import also triggers the centralized logging setup in config_loader
try:
    from src.config_loader import CONFIG
    from src.exceptions import DataFileNotFoundError # Import custom exception
    # Logging is configured by config_loader import, no need to setup here
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logging.error("Historical: Could not import CONFIG or Exceptions. Using defaults.", exc_info=True)
    CONFIG = {}
    class DataFileNotFoundError(FileNotFoundError): pass
except Exception as e:
     logging.basicConfig(level=logging.WARNING)
     logging.error(f"Historical: Error importing dependencies: {e}", exc_info=True)
     CONFIG = {}
     class DataFileNotFoundError(FileNotFoundError): pass

# --- Get Logger ---
log = logging.getLogger(__name__)

# --- Configuration Values Used ---
relative_data_path = CONFIG.get('paths', {}).get('data_file', 'data/Post-Processing/CSV_Files/Master_AQI_Dataset.csv')
DATA_PATH = os.path.join(PROJECT_ROOT, relative_data_path)
log.debug(f"Historical Data Path set to: {DATA_PATH}")

# --- Data Caching ---
_df_master_cached = None

# --- Core Data Loading and Preprocessing Function ---
def load_and_preprocess_data(force_reload=False):
    """Loads, preprocesses, and caches the master historical AQI dataset.

    Reads the CSV file specified by the 'data_file' path in the configuration.
    Parses the 'Date' column (expecting 'dd/mm/yy' format).
    Reconstructs a single 'City' column from one-hot encoded 'City_*' columns if needed.
    Sets the 'Date' column as the DataFrame index and sorts by date.
    Caches the resulting DataFrame in memory for subsequent calls.

    Args:
        force_reload (bool, optional): If True, bypasses the cache and reloads
                                       the data from the file. Defaults to False.

    Returns:
        pandas.DataFrame: The preprocessed historical data with a DatetimeIndex.

    Raises:
        DataFileNotFoundError: If the configured data file cannot be found or read.
        ValueError: If date parsing fails, required columns ('AQI', 'City') are
                    missing after processing, or 'City_' columns are missing for
                    reconstruction when needed.
        RuntimeError: For other unexpected data loading or preprocessing errors.
    """
    global _df_master_cached
    if _df_master_cached is not None and not force_reload:
        log.info("Returning cached historical dataframe.")
        return _df_master_cached

    log.info(f"Attempting to load historical data from config path: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        msg = f"CRITICAL: Data file not found at specified path: {DATA_PATH}"
        log.error(msg)
        raise DataFileNotFoundError(msg)

    try:
        # Load and parse date
        date_format = '%d/%m/%y'
        df = pd.read_csv(DATA_PATH)
        log.info(f"Raw historical data loaded successfully. Shape: {df.shape}")
        log.info(f"Parsing 'Date' column with format: {date_format}")
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
    except ValueError as ve:
         log.error(f"Error parsing 'Date' column with format '{date_format}'. Check CSV. Error: {ve}")
         raise
    except FileNotFoundError:
        msg = f"FileNotFoundError: Double-check data file exists at {DATA_PATH}"
        log.error(msg)
        raise DataFileNotFoundError(msg)
    except Exception as e:
        log.error(f"Failed to load data from CSV: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load data from CSV: {e}") from e

    # Preprocessing Steps
    try:
        # Reconstruct 'City' Column
        if 'City' not in df.columns:
             city_columns = [col for col in df.columns if col.startswith('City_')]
             if not city_columns:
                 msg = "CRITICAL: No columns starting with 'City_' found."
                 log.error(msg)
                 raise ValueError(msg)
             log.info(f"Reconstructing 'City' column from: {city_columns}")
             def get_city_name(row):
                 for city_col in city_columns:
                     if row[city_col] > 0: return city_col.replace('City_', '')
                 log.warning(f"Row index {row.name} has no City_ column marked.")
                 return 'Unknown'
             df['City'] = df.apply(get_city_name, axis=1)
             log.info("Finished reconstructing 'City' column.")
             log.debug("Reconstructed 'City' counts:\n" + str(df['City'].value_counts()))
        else:
             log.info("'City' column already present.")

        # Set 'Date' as Index and Sort
        if 'Date' in df.columns:
             if df.index.name != 'Date': df.set_index('Date', inplace=True)
             df.sort_index(inplace=True)
             log.info("'Date' column set as index and sorted.")
        else:
            msg = "CRITICAL: 'Date' column missing after loading/parsing."
            log.error(msg)
            raise ValueError(msg)

        # Verify essential columns
        required_cols = ['AQI', 'City']
        missing_req_cols = [col for col in required_cols if col not in df.columns]
        if missing_req_cols:
             msg = f"CRITICAL: Essential columns missing: {missing_req_cols}"
             log.error(msg)
             raise ValueError(msg)

    except Exception as e:
        log.error(f"Error during data preprocessing: {e}", exc_info=True)
        raise RuntimeError(f"Error during data preprocessing: {e}") from e

    _df_master_cached = df
    log.info(f"Historical DataFrame processed and cached. Final shape: {_df_master_cached.shape}")
    return _df_master_cached

# --- Data Access Functions ---

def get_available_cities():
    """Gets a sorted list of unique valid city names from the historical dataset.

    Handles potential errors during data loading by returning an empty list.

    Returns:
        list[str]: A sorted list of available city names (excluding 'Unknown').
                   Returns an empty list if data cannot be loaded or processed.
    """
    try:
        df = load_and_preprocess_data()
        if df is None or 'City' not in df.columns:
             log.error("Unexpected state: Cannot get cities from unavailable data.")
             return []
        cities = df['City'].unique()
        valid_cities = [str(city) for city in cities if city != 'Unknown' and pd.notna(city)]
        return sorted(valid_cities)
    except (DataFileNotFoundError, ValueError, RuntimeError) as e:
         log.error(f"Cannot get available cities due to error: {e}")
         return []


def get_city_aqi_trend_data(city_name):
    """Retrieves the AQI time series data for a specific city for trend analysis.

    Loads the preprocessed historical data (using cache if available), filters
    by the provided city name, and returns the AQI values as a Pandas Series
    indexed by date.

    Args:
        city_name (str): The name of the city to retrieve data for.

    Returns:
        pandas.Series or None: A Series with DatetimeIndex and AQI values for the city.
                               Returns None if the city is not found or if data loading fails.
    """
    try:
        df = load_and_preprocess_data()
        if 'City' not in df.columns or 'AQI' not in df.columns:
             log.error(f"Cannot get trend data: required columns missing from loaded df.")
             return None
        if city_name not in df['City'].unique():
            log.warning(f"City '{city_name}' not found in dataset. Available: {get_available_cities()}")
            return None
        city_data = df[df['City'] == city_name].copy()
        aqi_series = city_data['AQI']
        log.info(f"Returning AQI trend data Series for '{city_name}'. Length: {len(aqi_series)}")
        return aqi_series
    except (DataFileNotFoundError, ValueError, RuntimeError) as e:
         log.error(f"Cannot get trend data for '{city_name}' due to error: {e}")
         return None


def get_city_aqi_distribution_data(city_name):
    """Retrieves the raw AQI values for a specific city for distribution analysis.

    Loads the preprocessed historical data (using cache if available), filters
    by the provided city name, and returns the AQI values as a Pandas Series.

    Args:
        city_name (str): The name of the city to retrieve data for.

    Returns:
        pandas.Series or None: A Series containing AQI values for the specified city.
                               Returns None if the city is not found or if data loading fails.
    """
    try:
        df = load_and_preprocess_data()
        if 'City' not in df.columns or 'AQI' not in df.columns:
             log.error(f"Cannot get distribution data: required columns missing from loaded df.")
             return None
        if city_name not in df['City'].unique():
            log.warning(f"City '{city_name}' not found for distribution data.")
            return None
        city_data = df[df['City'] == city_name].copy()
        aqi_values = city_data['AQI']
        log.info(f"Returning AQI distribution data Series for '{city_name}'. Length: {len(aqi_values)}")
        return aqi_values
    except (DataFileNotFoundError, ValueError, RuntimeError) as e:
         log.error(f"Cannot get distribution data for '{city_name}' due to error: {e}")
         return None


# --- Example Usage Block ---
# (Keep existing __main__ block as is)
if __name__ == "__main__":
    # ... (test code remains the same) ...
    pass # Added pass for valid syntax if test code removed/commented