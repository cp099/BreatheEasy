# File: src/analysis/historical.py

import pandas as pd
import os
import logging # Using logging for better tracking and error reporting

# --- Configuration ---

# Determine the absolute path to the project directory
# Assumes this script (historical.py) is in BREATHEEASY/src/analysis/
# We need to go up two directories to reach BREATHEEASY/
try:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
except NameError:
    # Handle case where __file__ is not defined (e.g., interactive environments)
    # Fallback assumes script is run from project root in some cases, less reliable
    PROJECT_ROOT = os.path.abspath('.')
    logging.warning(f"__file__ not defined. Assuming project root is current directory: {PROJECT_ROOT}")


DATA_FILENAME = 'Master_AQI_Dataset.csv'
# Construct the absolute path to the data file relative to the project root
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Post-Processing', 'CSV_Files', DATA_FILENAME)

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# --- Data Caching ---
# Simple in-memory cache for the loaded DataFrame to avoid repeated file reads.
# A global variable holds the loaded data. More sophisticated caching could be used.
_df_master_cached = None

# --- Core Data Loading and Preprocessing Function ---

def load_and_preprocess_data(force_reload=False):
    """
    Loads the Master AQI dataset from the CSV file, performs initial preprocessing
    (date parsing, city column reconstruction, indexing), and caches the result.

    Args:
        force_reload (bool): If True, forces reloading the data from the file,
                             bypassing the cache. Defaults to False.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame with Date index, 'City' column,
                          and sorted by date, or None if loading/processing fails.
    """
    global _df_master_cached

    # Return cached version if available and not forcing reload
    if _df_master_cached is not None and not force_reload:
        logging.info("Returning cached historical dataframe.")
        return _df_master_cached

    # --- File Loading ---
    logging.info(f"Attempting to load data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        logging.error(f"CRITICAL: Data file not found at specified path: {DATA_PATH}")
        return None

    try:
        # Load CSV, ensuring Date parsing uses dayfirst=True based on previous notebook results
        df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
        logging.info(f"Data loaded successfully from CSV. Shape: {df.shape}")

    except FileNotFoundError:
        # This case is checked above, but good practice to handle specific exceptions
        logging.error(f"FileNotFoundError: Double-check data file exists at {DATA_PATH}")
        return None
    except Exception as e:
        logging.error(f"Failed to load data from CSV: {e}", exc_info=True) # Log traceback
        return None

    # --- Data Preprocessing ---
    try:
        # 1. Reconstruct 'City' Column
        city_columns = [col for col in df.columns if col.startswith('City_')]
        if not city_columns:
            logging.error("CRITICAL: No columns starting with 'City_' found. Cannot determine city information.")
            # Depending on requirements, could raise an error or return None
            return None # Cannot proceed without city info
        else:
            logging.info(f"Reconstructing 'City' column from one-hot columns: {city_columns}")
            def get_city_name(row):
                for city_col in city_columns:
                    if row[city_col] > 0: # Check for the '1' marker
                        return city_col.replace('City_', '') # Extract city name
                logging.warning(f"Row with index {row.name} has no City_ column marked as 1.")
                return 'Unknown' # Assign 'Unknown' if no city is marked for a row

            df['City'] = df.apply(get_city_name, axis=1)
            logging.info("Finished reconstructing 'City' column.")
            # Log value counts to verify reconstruction
            logging.debug("Reconstructed 'City' column value counts:\n" + str(df['City'].value_counts()))

        # 2. Set 'Date' as Index and Sort
        if 'Date' in df.columns:
             if df.index.name != 'Date':
                 df.set_index('Date', inplace=True)
             df.sort_index(inplace=True) # Ensure chronological order
             logging.info("'Date' column successfully set as index and sorted.")
        else:
            # This should not happen if read_csv worked correctly, but check defensively
            logging.error("CRITICAL: 'Date' column not found in DataFrame after loading. Cannot perform time series operations.")
            return None

        # 3. Verify essential columns exist
        required_cols = ['AQI', 'City'] # Add other pollutants if needed by other functions
        missing_req_cols = [col for col in required_cols if col not in df.columns]
        if missing_req_cols:
             logging.error(f"CRITICAL: Essential columns missing after preprocessing: {missing_req_cols}")
             return None

    except Exception as e:
        logging.error(f"An error occurred during data preprocessing: {e}", exc_info=True)
        return None


    # --- Cache and Return ---
    _df_master_cached = df # Store the processed dataframe in the global cache
    logging.info(f"Historical DataFrame processed and cached successfully. Final shape: {_df_master_cached.shape}")
    return _df_master_cached

# --- Data Access Functions (for UI Backend) ---

def get_available_cities():
    """
    Returns a sorted list of unique valid city names found in the dataset.
    Returns an empty list if data cannot be loaded or processed.
    """
    df = load_and_preprocess_data() # Get the cached or newly loaded data
    if df is None or 'City' not in df.columns:
        logging.error("Cannot get available cities: Data not loaded or 'City' column missing.")
        return [] # Return empty list to indicate failure or no data

    cities = df['City'].unique()
    # Filter out 'Unknown' city category if it exists
    valid_cities = [str(city) for city in cities if city != 'Unknown' and pd.notna(city)]
    return sorted(valid_cities) # Return sorted list of valid city names


def get_city_aqi_trend_data(city_name):
    """
    Retrieves the AQI time series data (Date-indexed) for a specific city.

    Args:
        city_name (str): The name of the city (e.g., 'Delhi', 'Mumbai').
                         Should match names derived from columns like 'City_Delhi'.

    Returns:
        pandas.Series: A Series with Date index and AQI values for the requested city.
                       Returns None if the city is not found, data fails to load,
                       or required columns ('AQI', 'City') are missing.
    """
    df = load_and_preprocess_data() # Get the data
    if df is None or 'City' not in df.columns or 'AQI' not in df.columns:
        logging.error(f"Cannot get trend data for '{city_name}': Data not loaded or essential columns missing.")
        return None

    # Check if the requested city exists in our reconstructed 'City' column
    if city_name not in df['City'].unique():
        logging.warning(f"City '{city_name}' not found in the dataset. Available cities: {get_available_cities()}")
        return None

    # Filter data for the specific city
    # Use .copy() to avoid SettingWithCopyWarning if modifications were intended later
    city_data = df[df['City'] == city_name].copy()

    # Return the AQI column as a Series (Date index is already set)
    aqi_series = city_data['AQI']
    logging.info(f"Returning AQI trend data Series for '{city_name}'. Length: {len(aqi_series)}")
    return aqi_series


def get_city_aqi_distribution_data(city_name):
    """
    Retrieves the raw AQI values (as a Series) for a specific city.
    Useful for generating distribution plots (histograms, box plots) on the frontend.

    Args:
        city_name (str): The name of the city.

    Returns:
        pandas.Series: A Series containing only the AQI values for the specified city.
                       Returns None if the city is not found, data loading failed,
                       or required columns ('AQI', 'City') are missing.
    """
    df = load_and_preprocess_data() # Get the data
    if df is None or 'City' not in df.columns or 'AQI' not in df.columns:
        logging.error(f"Cannot get distribution data for '{city_name}': Data not loaded or essential columns missing.")
        return None

    # Check if the requested city exists
    if city_name not in df['City'].unique():
        logging.warning(f"City '{city_name}' not found in the dataset for distribution data.")
        return None

    # Filter data for the specific city
    city_data = df[df['City'] == city_name].copy()

    # Return just the AQI values as a Series
    aqi_values = city_data['AQI']
    logging.info(f"Returning AQI distribution data Series for '{city_name}'. Length: {len(aqi_values)}")
    return aqi_values


# --- Example Usage Block (for testing the module directly) ---
if __name__ == "__main__":
    # This block executes only when the script is run directly (e.g., python src/analysis/historical.py)

    print("\n" + "="*30)
    print(" Running historical.py Tests ")
    print("="*30 + "\n")

    # --- Test 1: Load data and get available cities ---
    print("[Test 1: Get Available Cities]")
    available_cities = get_available_cities()
    if available_cities:
        print(f"Success! Available cities found: {available_cities}")
    else:
        print("Failure! Could not retrieve available cities. Check logs above for errors.")
        # Exit if we can't even get cities, as other tests will fail
        exit()

    # --- Test 2: Get Trend Data for a Valid City ---
    print("\n[Test 2: Get Trend Data]")
    test_city = available_cities[0] # Use the first city found (e.g., 'Bangalore')
    print(f"Attempting to get trend data for: '{test_city}'")
    trend_data = get_city_aqi_trend_data(test_city)
    if trend_data is not None and isinstance(trend_data, pd.Series):
        print(f"Success! Trend data received for {test_city}.")
        print("First 5 entries of the trend Series:")
        print(trend_data.head())
        print(f"Index type: {trend_data.index.dtype}, Series dtype: {trend_data.dtype}")
    else:
        print(f"Failure! Could not retrieve valid trend data for {test_city}.")

    # --- Test 3: Get Distribution Data for a Valid City ---
    print("\n[Test 3: Get Distribution Data]")
    print(f"Attempting to get distribution data for: '{test_city}'")
    dist_data = get_city_aqi_distribution_data(test_city)
    if dist_data is not None and isinstance(dist_data, pd.Series):
        print(f"Success! Distribution data received for {test_city}.")
        print("First 5 AQI values for distribution:")
        print(dist_data.head())
        print(f"Distribution data Series dtype: {dist_data.dtype}")
        print("Basic descriptive stats for distribution data:")
        print(dist_data.describe())
    else:
        print(f"Failure! Could not retrieve valid distribution data for {test_city}.")

    # --- Test 4: Get Data for a Non-existent City ---
    print("\n[Test 4: Non-existent City]")
    invalid_city = "Atlantis"
    print(f"Attempting to get trend data for invalid city: '{invalid_city}'")
    non_existent_trend = get_city_aqi_trend_data(invalid_city)
    if non_existent_trend is None:
        print("Success! Correctly returned None for non-existent city trend data.")
    else:
        print(f"Failure! Expected None but received data for invalid city '{invalid_city}'.")

    non_existent_dist = get_city_aqi_distribution_data(invalid_city)
    if non_existent_dist is None:
        print("Success! Correctly returned None for non-existent city distribution data.")
    else:
        print(f"Failure! Expected None but received data for invalid city '{invalid_city}'.")


    # --- Test 5: Check Caching Mechanism ---
    print("\n[Test 5: Caching Check]")
    print(f"Requesting trend data for '{test_city}' again...")
    # Call the function again, check logs for "Returning cached historical dataframe."
    trend_data_cached = get_city_aqi_trend_data(test_city)
    if trend_data_cached is not None:
        print(f"Successfully retrieved data for '{test_city}' on second request.")
        # You would typically check the log output manually or use more advanced testing
        # to confirm the "cached" message appeared after the first load.
    else:
        print(f"Failure! Could not retrieve data for '{test_city}' on second request.")

    print("\n" + "="*30)
    print(" historical.py Tests Finished ")
    print("="*30 + "\n")