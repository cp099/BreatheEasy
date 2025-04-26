# File: src/api_integration/client.py

"""
Handles interactions with the AQICN (World Air Quality Index Project) API.

Provides functions to fetch raw real-time air quality data, extract specific
current AQI values, and retrieve pollutant data along with interpreted health risks.
Requires an AQICN_API_TOKEN environment variable (loaded from .env).
API base URL is configurable via config/config.yaml.
"""

import requests
import os
import logging # Standard logging import
from dotenv import load_dotenv
import sys # For path manipulation
import json # Import json for JSONDecodeError check in except block

# --- Setup Project Root Path ---
# (Keep existing PROJECT_ROOT logic)
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')):
         PROJECT_ROOT = os.path.abspath('.')
    # Cannot log reliably here yet
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import Configuration and Other Modules ---
# This import also sets up logging via config_loader.py
try:
    from src.config_loader import CONFIG
    from src.exceptions import APIKeyError, APITimeoutError, APINotFoundError, APIError, ConfigError
    from src.health_rules.interpreter import interpret_pollutant_risks
    # Logging configured by config_loader
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"AQICN Client: Could not import dependencies. Error: {e}", exc_info=True)
    CONFIG = {}
    class APIKeyError(Exception): pass
    class APITimeoutError(Exception): pass
    class APINotFoundError(Exception): pass
    class APIError(Exception): pass
    class ConfigError(Exception): pass
    def interpret_pollutant_risks(iaqi_data): logging.error("interpret_pollutant_risks unavailable."); return []
except Exception as e:
     logging.basicConfig(level=logging.WARNING)
     logging.error(f"AQICN Client: Error importing dependencies: {e}", exc_info=True)
     CONFIG = {}
     class APIKeyError(Exception): pass
     class APITimeoutError(Exception): pass
     class APINotFoundError(Exception): pass
     class APIError(Exception): pass
     class ConfigError(Exception): pass
     def interpret_pollutant_risks(iaqi_data): return []


# --- Get Logger ---
log = logging.getLogger(__name__)

# --- Load API Token ---
# (Keep existing dotenv loading logic)
try:
    dotenv_path = os.path.join(PROJECT_ROOT, '.env')
    loaded = load_dotenv(dotenv_path=dotenv_path)
    if loaded:
         log.info(f"AQICN Client: Loaded .env file from: {dotenv_path}")
    else:
         log.warning(f"AQICN Client: .env file not found or empty at: {dotenv_path}")
except Exception as e:
    log.error(f"AQICN Client: Error loading .env file: {e}", exc_info=True)

AQICN_TOKEN = os.getenv('AQICN_API_TOKEN')

# --- API Client Function ---
def get_city_aqi_data(city_name):
    """Fetches full real-time AQI and pollutant data from the AQICN API.

    Constructs the request URL using the base URL from configuration and the
    provided city name. Handles various potential errors during the API call
    and response processing.

    Args:
        city_name (str): The name of the city (e.g., 'Delhi', 'Mumbai') to query.
                         Can also be a specific station ID prefixed with '@'.

    Returns:
        dict or None: A dictionary containing the full API response if successful
                      (status='ok'). The structure typically includes 'status' and 'data'
                      keys, where 'data' contains 'aqi', 'city', 'time', 'iaqi', etc.
                      Returns None if the API explicitly reports 'Unknown station'.

    Raises:
        APIKeyError: If the AQICN_API_TOKEN environment variable is missing or if the
                     API returns a 401 Unauthorized error.
        ConfigError: If the AQICN base URL is missing in the configuration.
        APITimeoutError: If the request to the API times out.
        APINotFoundError: If the API returns an HTTP 404 error (endpoint not found).
        APIError: For other HTTP errors (e.g., 403, 5xx) or if the API response
                  payload indicates an error status other than 'Unknown station'.
        ValueError: If the API response is not valid JSON.
        requests.exceptions.RequestException: For underlying network connection issues.
    """
    aqicn_base_url = CONFIG.get('apis', {}).get('aqicn', {}).get('base_url', "https://api.waqi.info/feed")

    if not AQICN_TOKEN:
        msg = "AQICN_API_TOKEN not found in environment variables."
        log.error(msg)
        raise APIKeyError(msg, service="AQICN")
    if not aqicn_base_url:
         msg = "AQICN base URL not found in configuration."
         log.error(msg)
         raise ConfigError(msg)

    api_url = f"{aqicn_base_url}/{city_name}/?token={AQICN_TOKEN}"
    log.info(f"Requesting data from AQICN API: {aqicn_base_url}/{city_name}/?token=***TOKEN_HIDDEN***")

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        log.debug(f"Raw AQICN API response JSON: {data}")

        if data.get("status") == "ok":
            log.info(f"Successfully received 'ok' status from AQICN for '{city_name}'.")
            return data
        elif data.get("status") == "error":
            error_message = data.get("data", "Unknown API error reason")
            log.error(f"AQICN API returned error status for '{city_name}': {error_message}")
            if "Unknown station" in str(error_message):
                 log.warning(f"City/Station '{city_name}' not found by AQICN API.")
                 return None # Return None for this specific expected data error
            raise APIError(error_message, service="AQICN") # Raise for other errors
        else:
            msg = f"Received unexpected/missing status from AQICN for '{city_name}': {data.get('status', 'None')}"
            log.error(msg)
            raise APIError(msg, service="AQICN")

    except requests.exceptions.Timeout as e:
        msg = f"Request to AQICN timed out for '{city_name}'."
        log.error(msg)
        raise APITimeoutError(msg, service="AQICN") from e
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        msg = f"AQICN HTTP error: {http_err} - Status: {status_code}"
        log.error(msg)
        if status_code == 401: raise APIKeyError("Authorization failed. Check AQICN_API_TOKEN.", service="AQICN") from http_err
        elif status_code == 404: raise APINotFoundError(f"AQICN endpoint or city '{city_name}' not found.", service="AQICN") from http_err
        else: raise APIError(f"HTTP error {status_code}", status_code=status_code, service="AQICN") from http_err
    except requests.exceptions.RequestException as req_err:
        msg = f"AQICN request error: {req_err}"
        log.error(msg)
        raise APIError(msg, service="AQICN") from req_err
    except (ValueError, json.JSONDecodeError) as json_err:
        msg = f"AQICN JSON decoding error: {json_err}"
        log.error(msg)
        raise ValueError(msg) from json_err


# --- Function for Section 3 ---
def get_current_aqi_for_city(city_name):
    """Fetches and extracts the current overall AQI value for a specific city.

    This function wraps get_city_aqi_data, extracts the relevant fields
    (AQI, station name, timestamp), and handles potential errors gracefully
    by returning None. Designed for Section 3 UI needs.

    Args:
        city_name (str): The name of the city (e.g., 'Delhi').

    Returns:
        dict or None: A dictionary containing {'city', 'aqi', 'station', 'time'}
                      if successful. Returns None if the underlying API call fails,
                      the city is not found, or essential data fields are missing
                      in the response.
    """
    log.info(f"Getting current AQI value (Sec 3) for: {city_name}")
    try:
        full_data = get_city_aqi_data(city_name)
        if full_data is None: # Handles "Unknown station" case from get_city_aqi_data
             log.warning(f"Station/City '{city_name}' not found by AQICN, cannot get current AQI.")
             return None

        # Process data if full_data is valid
        api_data = full_data.get("data", {})
        aqi_raw = api_data.get("aqi")
        if aqi_raw is None or aqi_raw == '-':
             log.warning(f"AQI value missing or N/A in API data for {city_name}.")
             return None
        aqi = int(aqi_raw)
        station_name = api_data.get("city", {}).get("name", city_name)
        timestamp = api_data.get("time", {}).get("s")
        if timestamp is None:
             log.error(f"Could not extract 'time' from API data for {city_name}.")
             return None
        result = {"city": city_name, "aqi": aqi, "station": station_name, "time": timestamp}
        log.info(f"Extracted current AQI info for {city_name}: {result}")
        return result
    # Catch specific exceptions raised by get_city_aqi_data
    except (APIKeyError, ConfigError, APITimeoutError, APINotFoundError, APIError, ValueError, requests.exceptions.RequestException) as e:
         log.error(f"Failed to get current AQI for {city_name} due to error: {e}")
         return None
    except Exception as e: # Catch any other unexpected errors
         log.error(f"Unexpected error getting current AQI for {city_name}: {e}", exc_info=True)
         return None


# --- Function for Section 5 ---
def get_current_pollutant_risks_for_city(city_name):
    """Fetches pollutant data and interprets associated health risks.

    Wraps get_city_aqi_data and interpret_pollutant_risks to provide
    a list of current health warnings based on live pollutant levels.
    Designed for Section 5 UI needs.

    Args:
        city_name (str): The name of the city (e.g., 'Delhi').

    Returns:
        dict or None: A dictionary containing {'city', 'time', 'pollutants', 'risks'}
                      if successful. 'pollutants' is the raw iaqi dict, 'risks' is
                      a list of strings. Returns None if the underlying API call fails,
                      city is not found, or essential data is missing.
    """
    log.info(f"Getting current pollutant risks (Section 5) for city: {city_name}")
    try:
        full_data = get_city_aqi_data(city_name)
        if full_data is None: # Handle "Unknown station"
             log.warning(f"Station/City '{city_name}' not found by AQICN, cannot get pollutant risks.")
             return None

        # Process data if full_data is valid
        api_data = full_data.get("data", {})
        iaqi_data = api_data.get("iaqi")
        timestamp = api_data.get("time", {}).get("s")
        if iaqi_data and timestamp:
            risk_list = interpret_pollutant_risks(iaqi_data) # From interpreter.py
            result = {"city": city_name, "time": timestamp, "pollutants": iaqi_data, "risks": risk_list}
            log.info(f"Interpreted pollutant risks for {city_name}. Risks found: {len(risk_list)}")
            return result
        else:
            log.error(f"Could not extract 'iaqi' or 'time' for risk interpretation for {city_name}.")
            return None
    # Catch specific exceptions raised by get_city_aqi_data
    except (APIKeyError, ConfigError, APITimeoutError, APINotFoundError, APIError, ValueError, requests.exceptions.RequestException) as e:
         log.error(f"Failed to get pollutant risks for {city_name} due to error: {e}")
         return None
    except Exception as e:
         log.error(f"Unexpected error getting pollutant risks for {city_name}: {e}", exc_info=True)
         return None

# --- Example Usage Block ---
# (Keep existing __main__ block as is)
if __name__ == "__main__":
    # ... (test code remains the same) ...
    pass # Added pass for valid syntax if test code removed/commented