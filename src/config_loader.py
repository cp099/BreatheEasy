
# File: src/config_loader.py
"""
Handles loading the project's YAML configuration and setting up centralized logging.

This module is intended to be one of the first imports in the application.
Upon import, it performs the following sequence:
1.  Determines the project's root directory.
2.  Loads the main `config/config.yaml` file into a dictionary.
3.  Configures the root logger based on settings in the loaded config.
4.  Provides a globally accessible `CONFIG` dictionary for other modules to use.

Includes fallback mechanisms for safe execution if the config file is missing
or logging setup fails.
"""

import yaml
import os
import logging
import logging.handlers 
import sys 
from functools import lru_cache 

# --- Determine Project Root ---
# This robust logic allows the application to find the root directory
# whether it's run as a script, a module, or in an interactive environment.

try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
except NameError:
    # Fallback for environments where __file__ is not defined.
    PROJECT_ROOT = os.path.abspath('.')
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')):
         alt_root = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
         if os.path.exists(os.path.join(alt_root, 'src')):
             PROJECT_ROOT = alt_root
         else:
              # A basic logger is used here as a last resort.
              logging.basicConfig(level=logging.WARNING)
              logging.warning(f"ConfigLoader: Could not reliably determine project root from CWD: {PROJECT_ROOT}")

# --- Define Global Config Path ---
CONFIG_FILE_NAME = 'config.yaml'
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', CONFIG_FILE_NAME)

# Ensure the project root is on the Python path for consistent imports.
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import Custom Exceptions (with fallbacks) ---
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



@lru_cache()
def load_config(config_path=CONFIG_PATH):
    """
    Loads the configuration from the YAML file, caching the result.

    Uses functools.lru_cache to ensure the file is read from disk only once,
    improving performance on subsequent calls.

    Args:
        config_path (str): The full path to the configuration YAML file.

    Returns:
        dict: A dictionary containing the configuration settings.

    Raises:
        ConfigFileNotFoundError: If the config file does not exist.
        ConfigError: If the file cannot be parsed or another loading error occurs.
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


def setup_logging(config):
    """
    Configures the root logger with console and optional file handlers.

    Reads logging settings from the provided config dict. It first removes any
    pre-existing handlers from the root logger to prevent duplicate logs.

    Args:
        config (dict): The loaded configuration dictionary, expecting a 'logging' key.
    """
    if not isinstance(config, dict): config = {}
    log_cfg = config.get('logging', {})

    # Default values are used if specific keys are missing from the config.
    log_level_str = log_cfg.get('level', 'INFO')
    log_format = log_cfg.get('format', '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
    log_to_file = log_cfg.get('log_to_file', False)
    log_filename = log_cfg.get('log_filename', 'app.log')
    log_file_level_str = log_cfg.get('log_file_level', 'DEBUG')
    log_console_level_str = log_cfg.get('log_console_level', 'INFO')

    # Convert string levels to logging constants.
    root_log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    file_log_level = getattr(logging, log_file_level_str.upper(), logging.DEBUG)
    console_log_level = getattr(logging, log_console_level_str.upper(), logging.INFO)


    root_logger = logging.getLogger()
    # Set the root logger to the lowest level of its handlers to capture all messages.
    root_logger.setLevel(min(root_log_level, file_log_level, console_log_level))

    # Clear any existing handlers to prevent duplicate output.
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    formatter = logging.Formatter(log_format)

    # Configure console handler.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    # This initial message confirms console logging is active.
    logging.info(f"Console logging configured at level: {logging.getLevelName(console_log_level)}") 

    # Configure optional file handler.
    if log_to_file:
        try:
            log_file_path = os.path.join(PROJECT_ROOT, log_filename)
            # Use RotatingFileHandler to prevent log files from growing indefinitely.
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


# --- Global Initialization: Load config and set up logging on module import ---
CONFIG = {}
try:
    # This is the main execution block that runs when the module is first imported.
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


def get_config():
    """
    A convenience accessor that returns the globally loaded configuration dictionary.

    Returns:
        dict: The cached configuration dictionary.
    """
    return CONFIG

# --- Example Usage / Direct Execution ---
if __name__ == "__main__":
    # This block demonstrates the module's behavior when run directly.
    print("\n--- Running config_loader.py Self-Demonstration ---\n")
    print(f"Project Root determined as: {PROJECT_ROOT}")
    print(f"Configuration file path is: {CONFIG_PATH}\n")

    print("--- Testing get_config() function ---")
    retrieved_config = get_config()
    if retrieved_config:
        print("Successfully retrieved configuration:")
        # Pretty print a snippet of the config
        import json
        print(json.dumps(retrieved_config.get('modeling', {}), indent=2))
    else:
        print("Configuration is empty, likely due to a loading error reported above.")

    print("\n--- Logging Demonstration ---")
    # The initial logging setup should already be visible above this point.
    # Here we demonstrate logging from another module's perspective.
    local_log = logging.getLogger("config_loader_main_test")
    local_log.debug("This is a debug message.")
    local_log.info("This is an info message.")
    local_log.warning("This is a warning message.")
    local_log.error("This is an error message.")
    local_log.critical("This is a critical message.")

    print("\nCheck console output and 'app.log' (if enabled) for the messages above.")
    print("\n--- Self-Demonstration Finished ---\n")