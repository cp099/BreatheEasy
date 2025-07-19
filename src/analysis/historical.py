# File: src/analysis/historical.py (Refactored for Hourly Data)
"""
Provides functions for accessing the new, feature-rich hourly dataset.
"""
import pandas as pd
import os
import sys
import logging

# --- Setup Path & Get Logger ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
log = logging.getLogger(__name__)

# --- Configuration ---
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Post-Processing", "CSV_Files", "Master_Features_AQI_Data.csv")
_df_cached = None

def get_historical_data_for_city(city_name: str):
    """
    Loads the master feature dataset and returns the data for a specific city.
    """
    global _df_cached
    if _df_cached is None:
        log.info(f"Loading historical feature data from: {DATA_PATH}")
        try:
            _df_cached = pd.read_csv(DATA_PATH, parse_dates=['Datetime'])
            _df_cached = _df_cached.set_index('Datetime')
        except FileNotFoundError:
            log.error(f"FATAL: Master feature data file not found at {DATA_PATH}")
            return pd.DataFrame() # Return empty dataframe on error
    
    city_data = _df_cached[_df_cached['City'] == city_name].copy()
    log.info(f"Returning {len(city_data)} historical records for {city_name}.")
    return city_data