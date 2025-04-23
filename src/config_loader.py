# File: src/config_loader.py

import yaml
import os
import logging
import logging.handlers # Import handlers for file logging
from functools import lru_cache

# --- Determine Project Root ---
# (Keep existing PROJECT_ROOT logic)
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')
    logging.warning(f"ConfigLoader: __file__ not defined. Assuming project root: {PROJECT_ROOT}")

# --- Define Config Path ---
CONFIG_FILE_NAME = 'config.yaml'
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', CONFIG_FILE_NAME)

# --- Configuration Loading Function ---
@lru_cache()
def load_config(config_path=CONFIG_PATH):
    # (Keep existing load_config logic)
    log = logging.getLogger(__name__) # Get logger instance
    log.info(f"Attempting to load configuration from: {config_path}")
    # ... (rest of existing function) ...
    if not os.path.exists(config_path):
        log.error(f"Configuration file not found at: {config_path}")
        return {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
             log.warning(f"Configuration file is empty: {config_path}")
             return {}
        log.info("Configuration loaded successfully.")
        return config
    except yaml.YAMLError as e:
        log.error(f"Error parsing YAML configuration file: {config_path}. Error: {e}", exc_info=True)
        return {}
    except Exception as e:
        log.error(f"An unexpected error occurred loading configuration: {e}", exc_info=True)
        return {}

# --- NEW: Central Logging Setup Function ---
def setup_logging(config):
    """Configures logging based on the loaded configuration."""
    log_cfg = config.get('logging', {})
    log_level_str = log_cfg.get('level', 'INFO')
    log_format = log_cfg.get('format', '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
    log_to_file = log_cfg.get('log_to_file', False)
    log_filename = log_cfg.get('log_filename', 'app.log')
    log_file_level_str = log_cfg.get('log_file_level', 'DEBUG') # Default DEBUG to file
    log_console_level_str = log_cfg.get('log_console_level', 'INFO') # Default INFO to console

    # Get numeric log levels
    root_log_level = getattr(logging, log_level_str.upper(), logging.INFO) # Overall minimum level
    file_log_level = getattr(logging, log_file_level_str.upper(), logging.DEBUG)
    console_log_level = getattr(logging, log_console_level_str.upper(), logging.INFO)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(root_log_level) # Set lowest level to handle

    # Remove existing handlers configured by basicConfig elsewhere (if any)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Create Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info(f"Console logging configured at level: {logging.getLevelName(console_log_level)}")


    # Create File Handler (if enabled)
    if log_to_file:
        try:
            log_file_path = os.path.join(PROJECT_ROOT, log_filename) # Log file in project root
            # Use RotatingFileHandler: Max 5MB per file, keep 3 backup files
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
            )
            file_handler.setLevel(file_log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"File logging configured at level: {logging.getLevelName(file_log_level)} to {log_file_path}")
        except Exception as e:
             logging.error(f"Failed to configure file logging: {e}", exc_info=True)
    else:
         logging.info("File logging is disabled in configuration.")


# --- Load config and Setup Logging on Import ---
CONFIG = load_config()
if CONFIG: # Only setup logging if config loaded successfully
     setup_logging(CONFIG)
else:
     # Fallback basic config if config load failed
     logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
     logging.warning("Using fallback basic logging configuration due to config load failure.")


def get_config():
    """Returns the cached configuration dictionary."""
    return CONFIG

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Logging should already be configured by the time this runs
    log = logging.getLogger(__name__) # Get logger for this test block
    print("\n--- Testing Config Loader & Logging ---")
    log.debug("This is a DEBUG message.")
    log.info("This is an INFO message.")
    log.warning("This is a WARNING message.")
    log.error("This is an ERROR message.")

    if CONFIG:
        print("\nConfig loaded. Sample values:")
        print(f"  Data file path: {CONFIG.get('paths', {}).get('data_file')}")
        print(f"  Target Cities: {CONFIG.get('modeling', {}).get('target_cities')}")
        print(f"  Logging Level (Config): {CONFIG.get('logging', {}).get('level')}")
        print(f"  Log to File (Config): {CONFIG.get('logging', {}).get('log_to_file')}")
    else:
        print("\nFailed to load configuration.")

    print("\nConfig loader test finished.")