# File: src/analysis/historical.py
"""
Provides functions for loading and accessing historical AQI data.

This module handles the loading of the master dataset, performs necessary
preprocessing, caches the data in memory, and offers functions to retrieve
specific data slices (e.g., by city) for analysis.
"""

import pandas as pd
import os
import logging
import sys

# --- Setup Project Root Path ---
# This allows the script to be run from anywhere and still find the project root.
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    # Fallback for environments where __file__ is not defined.
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')):
         PROJECT_ROOT = os.path.abspath('.')

if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import Configuration & Exceptions ---
# The config_loader import also triggers the centralized logging setup.
try:
    from src.config_loader import CONFIG
    from src.exceptions import DataFileNotFoundError
except ImportError:
    # Fallback if dependencies or project structure are not found.
    logging.basicConfig(level=logging.WARNING)
    logging.error("Historical: Could not import CONFIG or Exceptions. Using dummy fallbacks.", exc_info=True)
    CONFIG = {}
    class DataFileNotFoundError(FileNotFoundError): pass
except Exception as e:
    # Broad exception for other potential import errors.
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"Historical: Critical error importing dependencies: {e}", exc_info=True)
    CONFIG = {}
    class DataFileNotFoundError(FileNotFoundError): pass

log = logging.getLogger(__name__)

# --- Module Configuration ---
relative_data_path = CONFIG.get('paths', {}).get('data_file', 'data/default_path.csv')
DATA_PATH = os.path.join(PROJECT_ROOT, relative_data_path)
log.debug(f"Historical data path configured to: {DATA_PATH}")

# In-memory cache for the loaded DataFrame.
_df_master_cached = None

# --- Core Data Loading and Preprocessing ---
def load_and_preprocess_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Loads, preprocesses, and caches the master historical AQI dataset.

    Reads the CSV file, standardizes the 'Date' column, and reconstructs a
    'City' column from one-hot encoded 'City_*' columns if needed. The
    processed DataFrame is then cached in memory.

    Args:
        force_reload (bool, optional): If True, bypasses the cache and reloads
                                       the data from the file. Defaults to False.

    Returns:
        pd.DataFrame: The preprocessed historical data with a DatetimeIndex.

    Raises:
        DataFileNotFoundError: If the configured data file cannot be found.
        ValueError: If essential columns are missing after processing.
        RuntimeError: For other unexpected data loading or preprocessing errors.
    """
    global _df_master_cached
    if _df_master_cached is not None and not force_reload:
        log.info("Returning cached historical dataframe.")
        return _df_master_cached

    log.info(f"Loading historical data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        msg = f"Data file not found at specified path: {DATA_PATH}"
        log.error(msg)
        raise DataFileNotFoundError(msg)

    try:
        df = pd.read_csv(DATA_PATH)
        log.info(f"Raw historical data loaded successfully. Shape: {df.shape}")
        
        # Standardize the 'Date' column to datetime objects.
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')

    except ValueError as ve:
         log.error(f"Error parsing 'Date' column with format '%d/%m/%y'. Check CSV. Error: {ve}")
         raise
    except Exception as e:
        log.error(f"Failed to load data from CSV: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load data from CSV: {e}") from e

    # --- Preprocessing Steps ---
    try:
        # Reconstruct 'City' column if it doesn't exist (from one-hot encoding).
        if 'City' not in df.columns:
             city_columns = [col for col in df.columns if col.startswith('City_')]
             if not city_columns:
                 raise ValueError("CRITICAL: No 'City' or 'City_*' columns found.")
             
             log.info(f"Reconstructing 'City' column from: {city_columns}")
             # This uses the first 'City_*' column with a '1' to determine the city name.
             df['City'] = df[city_columns].idxmax(axis=1).str.replace('City_', '')
             log.info("Finished reconstructing 'City' column.")
        
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        # Verify essential columns are present after processing.
        required_cols = ['AQI', 'City']
        if not all(col in df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df.columns]
             raise ValueError(f"CRITICAL: Essential columns missing: {missing}")

    except Exception as e:
        log.error(f"Error during data preprocessing: {e}", exc_info=True)
        raise RuntimeError(f"Error during data preprocessing: {e}") from e

    _df_master_cached = df
    log.info(f"Historical DataFrame processed and cached. Final shape: {_df_master_cached.shape}")
    return _df_master_cached


# --- Data Access Functions ---

def get_available_cities() -> list[str]:
    """
    Gets a sorted list of unique city names from the historical dataset.

    Returns:
        list[str]: A sorted list of available city names, excluding 'Unknown'.
                   Returns an empty list if data loading fails.
    """
    try:
        df = load_and_preprocess_data()
        cities = df['City'].unique()
        # Ensure cities are strings and not NaN before sorting.
        valid_cities = [str(city) for city in cities if city != 'Unknown' and pd.notna(city)]
        return sorted(valid_cities)
    except (DataFileNotFoundError, ValueError, RuntimeError) as e:
         log.error(f"Could not get available cities due to an error: {e}")
         return []


def get_city_aqi_trend_data(city_name: str) -> pd.Series | None:
    """
    Retrieves the AQI time series for a specific city.

    Args:
        city_name (str): The name of the city to retrieve data for.

    Returns:
        pd.Series | None: A Series with DatetimeIndex and AQI values.
                          Returns None if the city is not found or data fails to load.
    """
    try:
        df = load_and_preprocess_data()
        if city_name not in df['City'].unique():
            log.warning(f"City '{city_name}' not found in dataset.")
            return None
        
        city_data = df[df['City'] == city_name]
        aqi_series = city_data['AQI']
        log.info(f"Returning AQI trend data Series for '{city_name}'. Length: {len(aqi_series)}")
        return aqi_series
    except (DataFileNotFoundError, ValueError, RuntimeError) as e:
         log.error(f"Cannot get trend data for '{city_name}': {e}")
         return None


def get_city_aqi_distribution_data(city_name: str) -> pd.Series | None:
    """
    Retrieves the raw AQI values for a specific city for distribution analysis.

    Args:
        city_name (str): The name of the city to retrieve data for.

    Returns:
        pd.Series | None: A Series of AQI values for the specified city.
                          Returns None if the city is not found or data fails to load.
    """
    try:
        df = load_and_preprocess_data()
        if city_name not in df['City'].unique():
            log.warning(f"City '{city_name}' not found for distribution data.")
            return None
        
        city_data = df[df['City'] == city_name]
        aqi_values = city_data['AQI']
        log.info(f"Returning AQI distribution data for '{city_name}'. Length: {len(aqi_values)}")
        return aqi_values
    except (DataFileNotFoundError, ValueError, RuntimeError) as e:
         log.error(f"Cannot get distribution data for '{city_name}': {e}")
         return None


# --- Example Usage / Direct Execution ---
if __name__ == "__main__":
    # This block runs only when the script is executed directly.
    # It serves as a quick test and demonstration of the module's functions.
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plotting_available = True
        sns.set_style("whitegrid")
    except ImportError:
        log.warning("Matplotlib or Seaborn not installed. Plotting tests will be skipped.")
        plotting_available = False

    print("\n" + "="*30)
    print(" Running historical.py Self-Test ")
    print("="*30 + "\n")

    # Define an output directory for any generated plots.
    plots_output_dir = os.path.join(PROJECT_ROOT, 'output_plots', 'historical_trends')
    if plotting_available:
        os.makedirs(plots_output_dir, exist_ok=True)
        log.info(f"Plotting is enabled. Plots will be saved to: {plots_output_dir}")

    try:
        print("[1] Getting available cities...")
        available_cities = get_available_cities()
        if not available_cities:
            raise RuntimeError("Failed to retrieve city list. Aborting tests.")
        print(f"--> Success! Found: {available_cities}")

        print("\n[2] Checking data caching...")
        get_available_cities() # This call should be from cache. Check logs for confirmation.
        print("--> Second call made. Check logs for 'Returning cached historical dataframe'.")

        print("\n[3] Testing data retrieval for a sample city...")
        test_city = available_cities[0]
        trend_data = get_city_aqi_trend_data(test_city)
        if trend_data is not None:
            print(f"--> Success! Trend data retrieved for '{test_city}'.")
        else:
            print(f"--> Failure! Could not get trend data for '{test_city}'.")

        if plotting_available:
            print("\n[4] Generating and saving plots for all cities...")
            for city in available_cities:
                print(f"  - Plotting for {city}...")
                city_trend_data = get_city_aqi_trend_data(city)
                if city_trend_data is not None and not city_trend_data.empty:
                    try:
                        plt.figure(figsize=(12, 6))
                        city_trend_data.plot(title=f"AQI Trend for {city}")
                        plt.ylabel("AQI Value")
                        plt.xlabel("Date")
                        plt.tight_layout()
                        plot_filename = os.path.join(plots_output_dir, f"{city.replace(' ', '_')}_AQI_Trend.png")
                        plt.savefig(plot_filename)
                        plt.close() # Important to close figure to free memory.
                    except Exception as plot_e:
                        log.error(f"Failed to generate plot for {city}: {plot_e}", exc_info=True)
                else:
                    log.warning(f"No trend data to plot for {city}.")
            print("--> Plot generation complete.")

    except Exception as e:
        print(f"\nAN ERROR OCCURRED DURING THE SELF-TEST: {e}")
        log.error("Error in historical.py __main__ block", exc_info=True)

    print("\n" + "="*30)
    print(" historical.py Self-Test Finished ")
    print("="*30 + "\n")