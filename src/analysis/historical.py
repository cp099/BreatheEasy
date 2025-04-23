# File: src/analysis/historical.py

import pandas as pd
import os
import logging
import sys # For path manipulation

# --- Setup Project Root Path ---
# Assuming this file is in src/analysis/, project root is two levels up
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    # Fallback if __file__ is not defined
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..')) # Assumes running from src/
    # Adjust fallback if running context is different (e.g., from project root directly)
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')):
         PROJECT_ROOT = os.path.abspath('.') # Assume running from project root
    logging.warning(f"Historical: __file__ not defined. Assuming project root: {PROJECT_ROOT}")
# Ensure src is importable if needed by other modules called from here (although not directly needed now)
# if PROJECT_ROOT not in sys.path:
#      sys.path.insert(0, PROJECT_ROOT)

# --- Import Configuration ---
# Must happen after PROJECT_ROOT potentially added to path if config_loader is in src
try:
    from src.config_loader import CONFIG
    log_level_str = CONFIG.get('logging', {}).get('level', 'INFO') # Default INFO
    log_format = CONFIG.get('logging', {}).get('format', '%(asctime)s - [%(levelname)s] - %(message)s')
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
except ImportError:
    logging.error("Could not import CONFIG from src.config_loader. Using defaults.")
    CONFIG = {} # Define CONFIG as empty dict to prevent errors later
    log_level = logging.INFO
    log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
except Exception as e:
     logging.error(f"Error setting up config/logging in historical.py: {e}")
     CONFIG = {}
     log_level = logging.INFO
     log_format = '%(asctime)s - [%(levelname)s] - %(message)s'


# --- Setup Logging ---
# Configure logging using values from config file
logging.basicConfig(level=log_level, format=log_format)
log = logging.getLogger(__name__) # Get logger for this module

# --- Configuration Values Used ---
# Construct absolute data path from project root and relative path in config
relative_data_path = CONFIG.get('paths', {}).get('data_file', 'data/Post-Processing/CSV_Files/Master_AQI_Dataset.csv') # Provide default
DATA_PATH = os.path.join(PROJECT_ROOT, relative_data_path)


# --- Data Caching ---
_df_master_cached = None

# --- Core Data Loading and Preprocessing Function ---

def load_and_preprocess_data(force_reload=False):
    """
    Loads the Master AQI dataset using path from config, preprocesses, and caches.
    (Core logic remains the same, only DATA_PATH source changed)

    Args:
        force_reload (bool): Force reload, bypass cache.

    Returns:
        pandas.DataFrame: Preprocessed DataFrame or None.
    """
    global _df_master_cached
    if _df_master_cached is not None and not force_reload:
        log.info("Returning cached historical dataframe.")
        return _df_master_cached

    log.info(f"Attempting to load historical data from config path: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        log.error(f"CRITICAL: Data file not found at specified path: {DATA_PATH}")
        return None

    try:
        # Define date format based on previous findings
        date_format = '%d/%m/%y'
        df = pd.read_csv(DATA_PATH) # Load first
        log.info(f"Raw historical data loaded successfully. Shape: {df.shape}")
        # Parse dates explicitly
        log.info(f"Parsing 'Date' column with format: {date_format}")
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)

    except ValueError as ve:
         log.error(f"Error parsing 'Date' column with format '{date_format}'. Check CSV. Error: {ve}")
         return None
    except FileNotFoundError:
        log.error(f"FileNotFoundError: Double-check data file exists at {DATA_PATH}")
        return None
    except Exception as e:
        log.error(f"Failed to load data from CSV: {e}", exc_info=True)
        return None

    # --- Data Preprocessing ---
    try:
        # 1. Reconstruct 'City' Column (if needed)
        if 'City' not in df.columns:
             city_columns = [col for col in df.columns if col.startswith('City_')]
             if not city_columns:
                 log.error("CRITICAL: No columns starting with 'City_' found. Cannot determine city info.")
                 return None
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
            log.error("CRITICAL: 'Date' column missing after loading/parsing.")
            return None

        # 3. Verify essential columns
        required_cols = ['AQI', 'City']
        missing_req_cols = [col for col in required_cols if col not in df.columns]
        if missing_req_cols:
             log.error(f"CRITICAL: Essential columns missing: {missing_req_cols}")
             return None

    except Exception as e:
        log.error(f"Error during data preprocessing: {e}", exc_info=True)
        return None

    # --- Cache and Return ---
    _df_master_cached = df
    log.info(f"Historical DataFrame processed and cached. Final shape: {_df_master_cached.shape}")
    return _df_master_cached

# --- Data Access Functions (No changes needed inside these, they use the loaded df) ---

def get_available_cities():
    """Returns a sorted list of unique valid city names from the dataset."""
    df = load_and_preprocess_data()
    if df is None or 'City' not in df.columns:
        log.error("Cannot get available cities: Data not loaded or 'City' column missing.")
        return []
    cities = df['City'].unique()
    valid_cities = [str(city) for city in cities if city != 'Unknown' and pd.notna(city)]
    return sorted(valid_cities)

def get_city_aqi_trend_data(city_name):
    """Retrieves the AQI time series data (Date-indexed) for a specific city."""
    df = load_and_preprocess_data()
    if df is None or 'City' not in df.columns or 'AQI' not in df.columns:
        log.error(f"Cannot get trend data for '{city_name}': Pre-reqs failed.")
        return None
    if city_name not in df['City'].unique():
        log.warning(f"City '{city_name}' not found in dataset. Available: {get_available_cities()}")
        return None
    city_data = df[df['City'] == city_name].copy()
    aqi_series = city_data['AQI']
    log.info(f"Returning AQI trend data Series for '{city_name}'. Length: {len(aqi_series)}")
    return aqi_series

def get_city_aqi_distribution_data(city_name):
    """Retrieves the raw AQI values (as a Series) for a specific city."""
    df = load_and_preprocess_data()
    if df is None or 'City' not in df.columns or 'AQI' not in df.columns:
        log.error(f"Cannot get distribution data for '{city_name}': Pre-reqs failed.")
        return None
    if city_name not in df['City'].unique():
        log.warning(f"City '{city_name}' not found for distribution data.")
        return None
    city_data = df[df['City'] == city_name].copy()
    aqi_values = city_data['AQI']
    log.info(f"Returning AQI distribution data Series for '{city_name}'. Length: {len(aqi_values)}")
    return aqi_values


# --- Example Usage Block (Should still work) ---
if __name__ == "__main__":
    print("\n" + "="*30)
    print(" Running historical.py Tests ")
    print("="*30 + "\n")
    # (Keep the existing test block code exactly as it was)
    # ... It will now use the config-driven data path indirectly ...
    # --- Test 1: Load data and get available cities ---
    print("[Test 1: Get Available Cities]")
    available_cities = get_available_cities()
    if available_cities:
        print(f"Success! Available cities found: {available_cities}")
    else:
        print("Failure! Could not retrieve available cities. Check logs above for errors.")
        exit()
    # --- Test 2: Get Trend Data for a Valid City ---
    print("\n[Test 2: Get Trend Data]")
    test_city = available_cities[0]
    print(f"Attempting to get trend data for: '{test_city}'")
    trend_data = get_city_aqi_trend_data(test_city)
    if trend_data is not None and isinstance(trend_data, pd.Series):
        print(f"Success! Trend data received for {test_city}.")
        print(trend_data.head())
    else: print(f"Failure! Could not retrieve trend data for {test_city}.")
    # --- Test 3: Get Distribution Data for a Valid City ---
    print("\n[Test 3: Get Distribution Data]")
    print(f"Attempting to get distribution data for: '{test_city}'")
    dist_data = get_city_aqi_distribution_data(test_city)
    if dist_data is not None and isinstance(dist_data, pd.Series):
        print(f"Success! Distribution data received for {test_city}.")
        print(dist_data.head())
        print(dist_data.describe())
    else: print(f"Failure! Could not retrieve distribution data for {test_city}.")
    # --- Test 4: Get Data for a Non-existent City ---
    print("\n[Test 4: Non-existent City]")
    invalid_city = "Atlantis"
    print(f"Attempting to get trend data for invalid city: '{invalid_city}'")
    non_existent_trend = get_city_aqi_trend_data(invalid_city)
    if non_existent_trend is None: print("Success! Correctly returned None.")
    else: print(f"Failure! Expected None for invalid city.")
    non_existent_dist = get_city_aqi_distribution_data(invalid_city)
    if non_existent_dist is None: print("Success! Correctly returned None.")
    else: print(f"Failure! Expected None for invalid city.")
    # --- Test 5: Check Caching Mechanism ---
    print("\n[Test 5: Caching Check]")
    print(f"Requesting trend data for '{test_city}' again...")
    trend_data_cached = get_city_aqi_trend_data(test_city) # Check logs for "cached" msg
    if trend_data_cached is not None: print(f"Successfully retrieved data for '{test_city}' on second request.")
    else: print(f"Failure! Could not retrieve data for '{test_city}' on second request.")
    print("\n" + "="*30)
    print(" historical.py Tests Finished ")
    print("="*30 + "\n")