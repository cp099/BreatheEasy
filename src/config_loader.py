# File: src/config_loader.py

import yaml
import os
import logging
import logging.handlers # Import handlers for file logging
import sys # Import sys for path manipulation
from functools import lru_cache # Use LRU cache for efficient loading

# --- Determine Project Root ---
# Assuming this file is in src/, project root is one level up
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
except NameError:
    # Fallback if __file__ is not defined (e.g. interactive)
    # Assume the current working directory IS the project root BREATHEEASY/
    PROJECT_ROOT = os.path.abspath('.')
    # Basic check if this assumption seems valid
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')):
         # If 'src' isn't here, maybe CWD is src/? Go up one level.
         alt_root = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
         if os.path.exists(os.path.join(alt_root, 'src')):
             PROJECT_ROOT = alt_root
         else:
              # Log basic warning if root is uncertain
              logging.basicConfig(level=logging.WARNING)
              logging.warning(f"ConfigLoader: Could not reliably determine project root from CWD: {PROJECT_ROOT}")

# --- Define Config Path ---
CONFIG_FILE_NAME = 'config.yaml'
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', CONFIG_FILE_NAME)

# --- Add Project Root to Path (for importing src.exceptions) ---
# Do this early so exception import works
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)
     # Initial basic log before config is loaded
     # logging.info(f"ConfigLoader: Added project root to sys.path: {PROJECT_ROOT}")

# --- Import Custom Exceptions ---
try:
    from src.exceptions import ConfigFileNotFoundError, ConfigError
except ModuleNotFoundError:
    # Define dummy exceptions if src structure/exceptions.py is missing
    # This allows the rest of the file to parse without crashing immediately
    logging.basicConfig(level=logging.ERROR) # Ensure logging is available
    logging.error("Could not import custom exceptions from src.exceptions. Using fallback definitions.")
    class ConfigFileNotFoundError(FileNotFoundError): pass
    class ConfigError(Exception): pass
except ImportError as e_imp:
     logging.basicConfig(level=logging.ERROR)
     logging.error(f"ImportError loading exceptions: {e_imp}")
     class ConfigFileNotFoundError(FileNotFoundError): pass
     class ConfigError(Exception): pass


# --- Configuration Loading Function ---
@lru_cache()
def load_config(config_path=CONFIG_PATH):
    """
    Loads the configuration from the YAML file.
    Uses LRU cache to load the file only once.

    Args:
        config_path (str): The path to the configuration YAML file.

    Raises:
        ConfigFileNotFoundError: If the config file doesn't exist.
        ConfigError: If the file cannot be parsed or other load errors occur.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    # Get logger instance inside function to ensure it's configured
    log = logging.getLogger(__name__)
    log.info(f"Attempting to load configuration from: {config_path}")
    if not os.path.exists(config_path):
        msg = f"Configuration file not found at: {config_path}"
        log.error(msg)
        raise ConfigFileNotFoundError(msg)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
             log.warning(f"Configuration file is empty: {config_path}")
             return {} # Return empty dict for empty file
        log.info("Configuration loaded successfully.")
        return config
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML configuration file: {config_path}. Error: {e}"
        log.error(msg, exc_info=True)
        raise ConfigError(msg) from e
    except Exception as e:
        msg = f"An unexpected error occurred loading configuration: {e}"
        log.error(msg, exc_info=True)
        raise ConfigError(msg) from e

# --- Central Logging Setup Function ---
def setup_logging(config):
    """Configures logging based on the loaded configuration."""
    # Ensure config is a dictionary
    if not isinstance(config, dict):
        config = {}

    log_cfg = config.get('logging', {})
    log_level_str = log_cfg.get('level', 'INFO')
    log_format = log_cfg.get('format', '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
    log_to_file = log_cfg.get('log_to_file', False)
    log_filename = log_cfg.get('log_filename', 'app.log')
    log_file_level_str = log_cfg.get('log_file_level', 'DEBUG')
    log_console_level_str = log_cfg.get('log_console_level', 'INFO')

    # Get numeric log levels safely
    root_log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    file_log_level = getattr(logging, log_file_level_str.upper(), logging.DEBUG)
    console_log_level = getattr(logging, log_console_level_str.upper(), logging.INFO)

    root_logger = logging.getLogger()
    # Set lowest level needed by any handler on the root logger
    root_logger.setLevel(min(root_log_level, file_log_level, console_log_level))

    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Setup Console Handler
    console_handler = logging.StreamHandler(sys.stdout) # Explicitly use stdout
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    # Log initial message *after* adding handler
    logging.info(f"Console logging configured at level: {logging.getLevelName(console_log_level)}")

    # Setup File Handler
    if log_to_file:
        try:
            log_file_path = os.path.join(PROJECT_ROOT, log_filename)
            # Use RotatingFileHandler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
            )
            file_handler.setLevel(file_log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"File logging configured at level: {logging.getLevelName(file_log_level)} to {log_file_path}")
        except Exception as e:
             # Log to console if file handler fails
             logging.error(f"Failed to configure file logging: {e}", exc_info=True)
    else:
         logging.info("File logging is disabled in configuration.")


# --- Load config and Setup Logging on Import ---
# Global CONFIG variable accessible after import
CONFIG = {}
try:
    CONFIG = load_config()
    # Setup logging using the loaded config
    setup_logging(CONFIG)
except (ConfigFileNotFoundError, ConfigError) as e:
     # Fallback basic config ONLY if config load failed during import
     logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
     logging.critical(f"CRITICAL: Failed to load configuration: {e}. Using fallback logging and empty config.", exc_info=True)
     # CONFIG remains {}
except Exception as e:
     # Catch any other unexpected errors during setup
     logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
     logging.critical(f"CRITICAL: Unexpected error during config/logging setup: {e}", exc_info=True)
     # CONFIG remains {}

# --- Convenience Accessor ---
def get_config():
    """Returns the cached configuration dictionary."""
    # Mostly useful if the initial load failed and you want to check if CONFIG is empty
    return CONFIG

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Logging should already be configured by the import process above
    log = logging.getLogger(__name__) # Get logger for this test block
    print("\n--- Testing Config Loader & Logging ---")

    # Log messages at different levels to test handlers
    log.debug("This is a DEBUG message (should appear in file if enabled).")
    log.info("This is an INFO message (should appear in console and file).")
    log.warning("This is a WARNING message.")
    log.error("This is an ERROR message.")
    log.critical("This is a CRITICAL message.")


    if CONFIG: # Check if config dictionary is not empty
        print("\nConfig loaded. Sample values:")
        print(f"  Data file path (relative): {CONFIG.get('paths', {}).get('data_file')}")
        print(f"  AQICN Base URL: {CONFIG.get('apis', {}).get('aqicn', {}).get('base_url')}")
        print(f"  Target Cities: {CONFIG.get('modeling', {}).get('target_cities')}")
        print(f"  Logging Level (Config): {CONFIG.get('logging', {}).get('level')}")
        print(f"  Log to File (Config): {CONFIG.get('logging', {}).get('log_to_file')}")

        # Test cache - call again
        print("\nCalling load_config() again (should use cache)...")
        # We expect a log message "Attempting to load configuration..." but the actual
        # file reading/parsing won't happen if cache hits. Hard to test cache hit
        # directly here without inspecting logs carefully or more complex test setup.
        cached_config = load_config()
        print(f"Is config object the same? {CONFIG is cached_config}") # Should be True due to @lru_cache

        # Test convenience accessor
        print("\nAccessing config via imported CONFIG variable:")
        print(f"  Model Version: {CONFIG.get('modeling', {}).get('prophet_model_version')}")
    else:
        print("\nFailed to load configuration (CONFIG dictionary is empty). Check logs.")

    print("\nConfig loader test finished.")