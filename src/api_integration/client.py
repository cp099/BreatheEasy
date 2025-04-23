# File: src/api_integration/client.py

import requests
import os
import logging # Standard logging import
from dotenv import load_dotenv
import sys # For path manipulation
import json

# --- Setup Project Root Path ---
# (Keep existing PROJECT_ROOT logic)
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')):
         PROJECT_ROOT = os.path.abspath('.')
    # Cannot log here reliably before config/logging is potentially set up
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import Configuration and Other Modules ---
# This import will trigger logging setup in config_loader
try:
    from src.config_loader import CONFIG
    # Import custom exceptions AFTER potentially adding root to path
    from src.exceptions import APIKeyError, APITimeoutError, APINotFoundError, APIError
    from src.health_rules.interpreter import interpret_pollutant_risks
    # Logging is configured by config_loader now
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"AQICN Client: Could not import dependencies. Error: {e}", exc_info=True)
    CONFIG = {}
    # Define dummy exceptions/functions if imports failed
    class APIKeyError(Exception): pass
    class APITimeoutError(Exception): pass
    class APINotFoundError(Exception): pass
    class APIError(Exception): pass
    def interpret_pollutant_risks(iaqi_data): return []
except Exception as e:
     logging.basicConfig(level=logging.WARNING)
     logging.error(f"AQICN Client: Error importing dependencies: {e}", exc_info=True)
     CONFIG = {}
     class APIKeyError(Exception): pass # Define dummies
     class APITimeoutError(Exception): pass
     class APINotFoundError(Exception): pass
     class APIError(Exception): pass
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
    """
    Fetches real-time AQI/pollutant data for a city from AQICN API using config URL.

    Returns:
        dict: A dictionary containing the fetched data ('status', 'data').

    Raises:
        APIKeyError: If AQICN_API_TOKEN is missing.
        ConfigError: If AQICN base URL is missing in config (if default removed).
        APITimeoutError: If the request times out.
        APINotFoundError: If the API returns HTTP 404.
        APIKeyError: If the API returns HTTP 401.
        APIError: For other HTTP errors or API status='error' responses.
        ValueError: If the response is not valid JSON.
        requests.exceptions.RequestException: For other network/connection issues.
    """
    aqicn_base_url = CONFIG.get('apis', {}).get('aqicn', {}).get('base_url', "https://api.waqi.info/feed")

    if not AQICN_TOKEN:
        msg = "AQICN_API_TOKEN not found in environment variables."
        log.error(msg)
        raise APIKeyError(msg, service="AQICN") # Raise specific error
    if not aqicn_base_url:
         msg = "AQICN base URL not found in configuration."
         log.error(msg)
         raise ConfigError(msg) # Use ConfigError from exceptions

    api_url = f"{aqicn_base_url}/{city_name}/?token={AQICN_TOKEN}"
    log.info(f"Requesting data from AQICN API: {aqicn_base_url}/{city_name}/?token=***TOKEN_HIDDEN***")

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status() # Raise HTTPError for 4xx/5xx
        data = response.json()      # Raise ValueError/JSONDecodeError if not JSON
        log.debug(f"Raw AQICN API response JSON: {data}")

        # Check application-level status AFTER ensuring response was successful
        if data.get("status") == "ok":
            log.info(f"Successfully received 'ok' status from AQICN for '{city_name}'.")
            return data
        elif data.get("status") == "error":
            error_message = data.get("data", "Unknown API error reason")
            log.error(f"AQICN API returned error status for '{city_name}': {error_message}")
            # Special handling for "Unknown station" - maybe return None instead of raising?
            # For now, let's raise a generic APIError for any status=error
            if "Unknown station" in str(error_message):
                 log.warning(f"City/Station '{city_name}' not found by AQICN API.")
                 # Option: return None here instead of raising, as it's an expected data state
                 return None
            raise APIError(error_message, service="AQICN")
        else:
            # Handle cases where 'status' key might be missing or has unexpected value
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
        if status_code == 401:
             raise APIKeyError("Authorization failed. Check AQICN_API_TOKEN.", service="AQICN") from http_err
        elif status_code == 404:
             raise APINotFoundError(f"AQICN endpoint or city '{city_name}' not found.", service="AQICN") from http_err
        else:
             raise APIError(f"HTTP error {status_code}", status_code=status_code, service="AQICN") from http_err
    except requests.exceptions.RequestException as req_err:
        # Handle other network-related errors (DNS failure, connection error, etc.)
        msg = f"AQICN request error: {req_err}"
        log.error(msg)
        raise APIError(msg, service="AQICN") from req_err # Raise generic API error
    except (ValueError, json.JSONDecodeError) as json_err:
        # Handle errors during JSON decoding
        msg = f"AQICN JSON decoding error: {json_err}"
        log.error(msg)
        raise ValueError(msg) from json_err # Re-raise standard ValueError

# --- Function for Section 3 ---
def get_current_aqi_for_city(city_name):
    """Fetches current overall AQI for a city, handling exceptions from API call."""
    log.info(f"Getting current AQI value (Sec 3) for: {city_name}")
    try:
        full_data = get_city_aqi_data(city_name) # Can raise exceptions
        if full_data is None: # Handle expected "Unknown station" case which returns None
             log.warning(f"Station/City '{city_name}' not found by AQICN, cannot get current AQI.")
             return None
        # Proceed if full_data is not None (implies status was 'ok')
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
    except (APIKeyError, ConfigError, APITimeoutError, APINotFoundError, APIError, ValueError, requests.exceptions.RequestException) as e:
         log.error(f"Failed to get current AQI for {city_name} due to error: {e}")
         return None # Return None on any caught exception
    except Exception as e: # Catch any other unexpected errors
         log.error(f"Unexpected error getting current AQI for {city_name}: {e}", exc_info=True)
         return None


# --- Function for Section 5 ---
def get_current_pollutant_risks_for_city(city_name):
    """Fetches pollutant data and interprets health risks, handling exceptions."""
    log.info(f"Getting current pollutant risks (Section 5) for city: {city_name}")
    try:
        full_data = get_city_aqi_data(city_name) # Can raise exceptions
        if full_data is None: # Handle expected "Unknown station"
             log.warning(f"Station/City '{city_name}' not found by AQICN, cannot get pollutant risks.")
             return None
        # Proceed if full_data is not None
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
    except (APIKeyError, ConfigError, APITimeoutError, APINotFoundError, APIError, ValueError, requests.exceptions.RequestException) as e:
         log.error(f"Failed to get pollutant risks for {city_name} due to error: {e}")
         return None # Return None on any caught exception
    except Exception as e:
         log.error(f"Unexpected error getting pollutant risks for {city_name}: {e}", exc_info=True)
         return None

# --- Example Usage Block (No changes needed here) ---
if __name__ == "__main__":
    # Logging configured by importing CONFIG above
    print("\n" + "="*30)
    print(" Running api_integration/client.py Tests ")
    print("="*30 + "\n")
    # (Keep the existing test block code exactly as it was)
    test_city_full = "Delhi"
    print(f"[Test 1: Fetching FULL data for '{test_city_full}']")
    city_data_full = get_city_aqi_data(test_city_full) # Call directly for test
    if city_data_full and city_data_full.get("status") == "ok": print(f"Success! Received FULL data for '{test_city_full}'.")
    else: print(f"Failure or API error for '{test_city_full}'. Check logs.")
    test_city_sec3 = "Mumbai"
    print(f"\n[Test 2: Fetching CURRENT AQI (Section 3) for '{test_city_sec3}']")
    current_aqi_info = get_current_aqi_for_city(test_city_sec3) # Uses wrapper
    if current_aqi_info:
         print(f"Success! Received Current AQI Info for {test_city_sec3}:")
         print(f"  AQI: {current_aqi_info.get('aqi')}")
         print(f"  Station: {current_aqi_info.get('station')}")
         print(f"  Timestamp: {current_aqi_info.get('time')}")
    else: print(f"Failure! Could not retrieve current AQI info for {test_city_sec3}.")
    print("\n[Test 3: Fetching data for 'Atlantis']")
    invalid_city_data = get_city_aqi_data("Atlantis") # Call directly
    if invalid_city_data is None: print("Success! Handled non-existent city 'Atlantis' for full data.")
    else: print("Failure! Expected None for 'Atlantis'.")
    invalid_aqi_info = get_current_aqi_for_city("Atlantis") # Uses wrapper
    if invalid_aqi_info is None: print("Success! Handled non-existent city 'Atlantis' for current AQI.")
    else: print("Failure! Expected None for 'Atlantis' current AQI.")
    print("\n[Test 4: Check for API Key Missing (Manual Simulation Needed)]")
    print("(Requires manually removing/commenting AQICN_API_TOKEN in .env or code)")
    test_city_sec5 = "Delhi"
    print(f"\n[Test 5: Fetching CURRENT POLLUTANT RISKS (Section 5) for '{test_city_sec5}']")
    current_risks_info = get_current_pollutant_risks_for_city(test_city_sec5) # Uses wrapper
    if current_risks_info:
        print(f"Success! Received Current Pollutant Risks Info for {test_city_sec5}:")
        print(f"  Timestamp: {current_risks_info.get('time')}")
        print("  Identified Risks:")
        if current_risks_info['risks']:
            for risk in current_risks_info['risks']: print(f"    - {risk}")
        else: print("    - None")
    else: print(f"Failure! Could not retrieve current pollutant risks info for {test_city_sec5}.")
    print("\n" + "="*30)
    print(" api_integration/client.py Tests Finished ")
    print("="*30 + "\n")