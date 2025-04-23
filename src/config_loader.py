# File: src/config_loader.py

import yaml
import os
import logging
from functools import lru_cache # Use LRU cache for efficient loading

# --- Determine Project Root ---
# Assuming this file is in src/, project root is one level up
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
except NameError:
    # Fallback if __file__ not defined (e.g. interactive)
    PROJECT_ROOT = os.path.abspath('.') # Assumes running from project root
    logging.warning(f"ConfigLoader: __file__ not defined. Assuming project root: {PROJECT_ROOT}")

# --- Define Config Path ---
CONFIG_FILE_NAME = 'config.yaml'
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', CONFIG_FILE_NAME)

# --- Configuration Loading Function ---
@lru_cache() # Cache the result of this function
def load_config(config_path=CONFIG_PATH):
    """
    Loads the configuration from the YAML file.
    Uses LRU cache to load the file only once.

    Args:
        config_path (str): The path to the configuration YAML file.

    Returns:
        dict: A dictionary containing the configuration settings.
              Returns an empty dictionary if loading fails.
    """
    log = logging.getLogger(__name__) # Get logger
    log.info(f"Attempting to load configuration from: {config_path}")
    if not os.path.exists(config_path):
        log.error(f"Configuration file not found at: {config_path}")
        return {} # Return empty dict on failure

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: # Handle empty config file
             log.warning(f"Configuration file is empty: {config_path}")
             return {}
        log.info("Configuration loaded successfully.")
        return config
    except yaml.YAMLError as e:
        log.error(f"Error parsing YAML configuration file: {config_path}. Error: {e}", exc_info=True)
        return {} # Return empty dict on failure
    except Exception as e:
        log.error(f"An unexpected error occurred loading configuration: {e}", exc_info=True)
        return {}

# --- Convenience Accessor ---
# Load config once when the module is imported
CONFIG = load_config()

def get_config():
    """Returns the cached configuration dictionary."""
    # This function primarily exists to make it explicit we're getting the cached config
    # In most cases, directly importing and using 'CONFIG' is sufficient.
    return CONFIG

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s')
    print("\n--- Testing Config Loader ---")
    loaded_config = load_config() # Call the function

    if loaded_config:
        print("Config loaded. Sample values:")
        print(f"  Data file path (relative): {loaded_config.get('paths', {}).get('data_file')}")
        print(f"  AQICN Base URL: {loaded_config.get('apis', {}).get('aqicn', {}).get('base_url')}")
        print(f"  Target Cities: {loaded_config.get('modeling', {}).get('target_cities')}")
        print(f"  Logging Level: {loaded_config.get('logging', {}).get('level')}")

        # Test cache - call again, check logs in actual run for "cached" message if logger was set earlier
        print("\nCalling load_config() again (should use cache)...")
        cached_config = load_config()
        # Simple check that it returned the same object (due to cache)
        print(f"Is config object the same? {loaded_config is cached_config}")

        # Test convenience accessor
        print("\nAccessing config via imported CONFIG variable:")
        print(f"  Model Version: {CONFIG.get('modeling', {}).get('prophet_model_version')}")
    else:
        print("Failed to load configuration.")

    print("\nConfig loader test finished.")