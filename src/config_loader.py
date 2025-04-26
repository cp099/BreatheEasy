# File: src/config_loader.py

"""
Handles loading and caching of the project's YAML configuration file (`config/config.yaml`)
and sets up centralized logging based on the loaded configuration.

Provides a globally accessible CONFIG dictionary after initial import.
Includes fallback mechanisms if configuration loading or logging setup fails.
"""

import yaml
import os
import logging
import logging.handlers # Import handlers for file logging
import sys # Import sys for path manipulation
from functools import lru_cache # Use LRU cache for efficient loading

# --- Determine Project Root ---
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')):
         alt_root = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
         if os.path.exists(os.path.join(alt_root, 'src')):
             PROJECT_ROOT = alt_root
         else:
              logging.basicConfig(level=logging.WARNING)
              logging.warning(f"ConfigLoader: Could not reliably determine project root from CWD: {PROJECT_ROOT}")

# --- Define Config Path ---
CONFIG_FILE_NAME = 'config.yaml'
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', CONFIG_FILE_NAME)

# --- Add Project Root to Path ---
# (Keep existing sys.path logic)
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import Custom Exceptions ---
# (Keep existing exception import logic with fallbacks)
try:
    from src.exceptions import ConfigFileNotFoundError, ConfigError
except ModuleNotFoundError:
    logging.basicConfig(level=logging.ERROR)
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
    """Loads the configuration from the YAML file. (Existing Docstring is good)
    Uses LRU cache to load the file only once.

    Args:
        config_path (str): The path to the configuration YAML file.

    Raises:
        ConfigFileNotFoundError: If the config file doesn't exist.
        ConfigError: If the file cannot be parsed or other load errors occur.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
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
             return {}
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
    """Configures root logger with console and optional file handlers. (Existing Docstring is good)

    Reads logging level, format, and file settings from the provided config dict.
    Removes pre-existing handlers before adding new ones.

    Args:
        config (dict): The loaded configuration dictionary (expects a 'logging' key).
    """
    # (Function code remains the same)
    if not isinstance(config, dict): config = {}
    log_cfg = config.get('logging', {})
    log_level_str = log_cfg.get('level', 'INFO')
    log_format = log_cfg.get('format', '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
    log_to_file = log_cfg.get('log_to_file', False)
    log_filename = log_cfg.get('log_filename', 'app.log')
    log_file_level_str = log_cfg.get('log_file_level', 'DEBUG')
    log_console_level_str = log_cfg.get('log_console_level', 'INFO')
    root_log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    file_log_level = getattr(logging, log_file_level_str.upper(), logging.DEBUG)
    console_log_level = getattr(logging, log_console_level_str.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(min(root_log_level, file_log_level, console_log_level))
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    formatter = logging.Formatter(log_format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info(f"Console logging configured at level: {logging.getLevelName(console_log_level)}") # Log after adding handler
    if log_to_file:
        try:
            log_file_path = os.path.join(PROJECT_ROOT, log_filename)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
            file_handler.setLevel(file_log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"File logging configured at level: {logging.getLevelName(file_log_level)} to {log_file_path}") # Log after adding handler
        except Exception as e:
             logging.error(f"Failed to configure file logging: {e}", exc_info=True)
    else:
         logging.info("File logging is disabled in configuration.")


# --- Load config and Setup Logging on Import ---
# (Keep existing logic with try/except block)
CONFIG = {}
try:
    CONFIG = load_config()
    if CONFIG is not None:
         setup_logging(CONFIG)
    else:
        raise ConfigError("load_config returned None unexpectedly.")
except (ConfigFileNotFoundError, ConfigError) as e:
     logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
     logging.critical(f"CRITICAL: Failed to load configuration: {e}. Using fallback logging and empty config.", exc_info=True)
except Exception as e:
     logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
     logging.critical(f"CRITICAL: Unexpected error during config/logging setup: {e}", exc_info=True)

# --- Convenience Accessor ---
def get_config():
    """Returns the cached configuration dictionary. (Existing Docstring is good)

    Useful for checking if the global CONFIG variable was successfully populated,
    especially if called after potential import-time failures.
    """
    return CONFIG

# --- Example Usage Block ---
# (Keep existing __main__ block)
if __name__ == "__main__":
    # ... (test code remains the same) ...
    pass # Pass added just to have valid syntax if test code removed