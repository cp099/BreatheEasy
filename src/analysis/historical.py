# File: src/analysis/historical.py

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
# Ensure src modules can be imported
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import Configuration & Exceptions ---
# This import also triggers the centralized logging setup in config_loader
try:
    from src.config_loader import CONFIG
    from src.exceptions import DataFileNotFoundError # Import custom exception
    # Logging is configured by config_loader import, no need to setup here
except ImportError as e:
    # Basic logging ONLY if config fails
    logging.basicConfig(level=logging.WARNING)
    logging.error("Historical: Could not import CONFIG or Exceptions. Using defaults.", exc_info=True)
    CONFIG = {}
    # Define dummy exception if import failed
    class DataFileNotFoundError(FileNotFoundError): pass
except Exception as e:
     logging.basicConfig(level=logging.WARNING)
     logging.error(f"Historical: Error importing dependencies: {e}", exc_info=True)
     CONFIG = {}
     class DataFileNotFoundError(FileNotFoundError): pass

# --- Get Logger ---
# Get the logger instance for this module
log = logging.getLogger(__name__)

# --- Configuration Values Used ---
relative_data_path = CONFIG.get('paths', {}).get('data_file', 'data/Post-Processing/CSV_Files/Master_AQI_Dataset.csv')
DATA_PATH = os.path.join(PROJECT_ROOT, relative_data_path)
log.debug(f"Historical Data Path set to: {DATA_PATH}")

# --- Data Caching ---
_df_master_cached = None

# --- Core Data Loading and Preprocessing Function ---
def load_and_preprocess_data(force_reload=False):
    """
    Loads the Master AQI dataset using path from config, preprocesses, and caches.

    Args:
        force_reload (bool): Force reload, bypass cache.

    Returns:
        pandas.DataFrame: Preprocessed DataFrame.

    Raises:
        DataFileNotFoundError: If the configured data file cannot be found or read.
        ValueError: If date parsing fails or essential columns are missing.
        Exception: For other unexpected data loading or preprocessing errors.
    """
    global _df_master_cached
    if _df_master_cached is not None and not force_reload:
        log.info("Returning cached historical dataframe.")
        return _df_master_cached

    log.info(f"Attempting to load historical data from config path: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        msg = f"CRITICAL: Data file not found at specified path: {DATA_PATH}"
        log.error(msg)
        # Raise specific exception
        raise DataFileNotFoundError(msg)

    try:
        date_format = '%d/%m/%y'
        df = pd.read_csv(DATA_PATH)
        log.info(f"Raw historical data loaded successfully. Shape: {df.shape}")
        log.info(f"Parsing 'Date' column with format: {date_format}")
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
    except ValueError as ve:
         # Re-raise standard ValueError for parsing issues
         log.error(f"Error parsing 'Date' column with format '{date_format}'. Check CSV. Error: {ve}")
         raise
    except FileNotFoundError:
        # Should be caught by os.path.exists, but handle defensively
        msg = f"FileNotFoundError: Double-check data file exists at {DATA_PATH}"
        log.error(msg)
        raise DataFileNotFoundError(msg) # Raise custom exception
    except Exception as e:
        # Raise other read errors as generic Exception or a custom one
        log.error(f"Failed to load data from CSV: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load data from CSV: {e}") from e # Wrap generic exceptions

    # --- Data Preprocessing ---
    try:
        # 1. Reconstruct 'City' Column (if needed)
        if 'City' not in df.columns:
             city_columns = [col for col in df.columns if col.startswith('City_')]
             if not city_columns:
                 msg = "CRITICAL: No columns starting with 'City_' found."
                 log.error(msg)
                 raise ValueError(msg) # Raise standard ValueError
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

        # 2. Set 'Date' as Index and Sort
        if 'Date' in df.columns:
             if df.index.name != 'Date': df.set_index('Date', inplace=True)
             df.sort_index(inplace=True)
             log.info("'Date' column set as index and sorted.")
        else:
            msg = "CRITICAL: 'Date' column missing after loading/parsing."
            log.error(msg)
            raise ValueError(msg) # Raise standard ValueError

        # 3. Verify essential columns
        required_cols = ['AQI', 'City']
        missing_req_cols = [col for col in required_cols if col not in df.columns]
        if missing_req_cols:
             msg = f"CRITICAL: Essential columns missing: {missing_req_cols}"
             log.error(msg)
             raise ValueError(msg) # Raise standard ValueError

    except Exception as e:
        # Raise other preprocessing errors
        log.error(f"Error during data preprocessing: {e}", exc_info=True)
        raise RuntimeError(f"Error during data preprocessing: {e}") from e

    # --- Cache and Return ---
    _df_master_cached = df
    log.info(f"Historical DataFrame processed and cached. Final shape: {_df_master_cached.shape}")
    return _df_master_cached

# --- Data Access Functions ---
# These now need to handle potential exceptions from load_and_preprocess_data

def get_available_cities():
    """Returns a sorted list of unique valid city names from the dataset."""
    try:
        df = load_and_preprocess_data()
        if df is None or 'City' not in df.columns: # Should not happen if exceptions are raised
             log.error("Unexpected state: load_and_preprocess_data returned None or df missing 'City'.")
             return []
        cities = df['City'].unique()
        valid_cities = [str(city) for city in cities if city != 'Unknown' and pd.notna(city)]
        return sorted(valid_cities)
    except (DataFileNotFoundError, ValueError, RuntimeError) as e:
         log.error(f"Cannot get available cities due to error: {e}")
         return [] # Return empty list on error


def get_city_aqi_trend_data(city_name):
    """Retrieves the AQI time series data for a specific city."""
    try:
        df = load_and_preprocess_data()
        # Basic check added here, though load should raise if AQI/City missing
        if 'City' not in df.columns or 'AQI' not in df.columns:
             log.error(f"Cannot get trend data: required columns missing from loaded dataframe.")
             return None
        if city_name not in df['City'].unique():
            log.warning(f"City '{city_name}' not found in dataset. Available: {get_available_cities()}") # Might log error if get_available_cities fails too
            return None
        city_data = df[df['City'] == city_name].copy()
        aqi_series = city_data['AQI']
        log.info(f"Returning AQI trend data Series for '{city_name}'. Length: {len(aqi_series)}")
        return aqi_series
    except (DataFileNotFoundError, ValueError, RuntimeError) as e:
         log.error(f"Cannot get trend data for '{city_name}' due to error: {e}")
         return None # Return None on error


def get_city_aqi_distribution_data(city_name):
    """Retrieves the raw AQI values for a specific city."""
    try:
        df = load_and_preprocess_data()
        if 'City' not in df.columns or 'AQI' not in df.columns:
             log.error(f"Cannot get distribution data: required columns missing from loaded dataframe.")
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
         return None # Return None on error


# --- Example Usage Block (Adjusted to handle exceptions) ---
if __name__ == "__main__":
    print("\n" + "="*30)
    print(" Running historical.py Tests ")
    print("="*30 + "\n")

    try:
        # --- Test 1: Load data and get available cities ---
        print("[Test 1: Get Available Cities]")
        available_cities = get_available_cities()
        if available_cities:
            print(f"Success! Available cities found: {available_cities}")
        else:
            print("Failure! Could not retrieve available cities. Check logs.")
            # Don't exit immediately, let other tests show failures too
            available_cities = [] # Set empty to avoid errors below

        if not available_cities: # Skip further tests if cities couldn't be loaded
            print("\nSkipping further tests as city list could not be obtained.")
        else:
            # --- Test 2: Get Trend Data ---
            print("\n[Test 2: Get Trend Data]")
            test_city = available_cities[0]
            print(f"Attempting to get trend data for: '{test_city}'")
            trend_data = get_city_aqi_trend_data(test_city)
            if trend_data is not None: print(f"Success! Trend data received for {test_city}.\n{trend_data.head()}")
            else: print(f"Failure! Could not retrieve trend data for {test_city}.")

            # --- Test 3: Get Distribution Data ---
            print("\n[Test 3: Get Distribution Data]")
            print(f"Attempting to get distribution data for: '{test_city}'")
            dist_data = get_city_aqi_distribution_data(test_city)
            if dist_data is not None: print(f"Success! Distribution data received for {test_city}.\n{dist_data.head()}\n{dist_data.describe()}")
            else: print(f"Failure! Could not retrieve distribution data for {test_city}.")

            # --- Test 4: Non-existent City ---
            print("\n[Test 4: Non-existent City]")
            invalid_city = "Atlantis"
            print(f"Attempting data for invalid city: '{invalid_city}'")
            non_existent_trend = get_city_aqi_trend_data(invalid_city)
            if non_existent_trend is None: print("Success! Correctly returned None for trend.")
            else: print("Failure! Expected None for invalid city trend.")
            non_existent_dist = get_city_aqi_distribution_data(invalid_city)
            if non_existent_dist is None: print("Success! Correctly returned None for distribution.")
            else: print("Failure! Expected None for invalid city distribution.")

            # --- Test 5: Caching Check ---
            print("\n[Test 5: Caching Check]")
            print(f"Requesting trend data for '{test_city}' again...")
            trend_data_cached = get_city_aqi_trend_data(test_city)
            if trend_data_cached is not None: print(f"Successfully retrieved data for '{test_city}' again.")
            else: print(f"Failure! Could not retrieve data for '{test_city}' again.")

    except Exception as e:
        print(f"\nAn unexpected error occurred during the test run: {e}")
        log.error("Error in historical.py __main__ block", exc_info=True)

    print("\n" + "="*30)
    print(" historical.py Tests Finished ")
    print("="*30 + "\n")