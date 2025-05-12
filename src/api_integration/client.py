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
import logging
from dotenv import load_dotenv
import sys
import json

# --- Setup Project Root Path ---
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')):
         PROJECT_ROOT = os.path.abspath('.')
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import Configuration and Other Modules ---
try:
    from src.config_loader import CONFIG
    from src.exceptions import APIKeyError, APITimeoutError, APINotFoundError, APIError, ConfigError
    from src.health_rules.interpreter import interpret_pollutant_risks
except ImportError as e:
    logging.basicConfig(level=logging.WARNING) # Basic logging if config_loader fails
    logging.error(f"AQICN Client: Could not import dependencies. Error: {e}", exc_info=True)
    # Define minimal fallbacks if imports fail
    CONFIG = {'apis': {'aqicn': {'base_url': "https://api.waqi.info/feed"}}, 'api_timeout_seconds': 10}
    class APIKeyError(Exception): pass
    class APITimeoutError(Exception): pass
    class APINotFoundError(Exception): pass
    class APIError(Exception): pass
    class ConfigError(Exception): pass
    def interpret_pollutant_risks(iaqi_data): 
        logging.error("Dummy interpret_pollutant_risks called due to import error.")
        return ["Pollutant interpretation unavailable."]
except Exception as e: # Catch any other unexpected error during imports
     logging.basicConfig(level=logging.WARNING)
     logging.error(f"AQICN Client: Critical error importing dependencies: {e}", exc_info=True)
     raise # Re-raise if it's a critical issue during startup beyond simple ImportErrors

# --- Get Logger ---
# Logger is configured by config_loader.py when it's imported successfully.
# If config_loader fails, the basicConfig above provides a fallback.
log = logging.getLogger(__name__) 

# --- Load API Token ---
try:
    # Construct path to .env file relative to PROJECT_ROOT
    # This ensures it looks for .env in BREATHEEASY/ directory
    dotenv_path = os.path.join(PROJECT_ROOT, '.env') 
    if os.path.exists(dotenv_path):
        loaded = load_dotenv(dotenv_path=dotenv_path)
        if loaded:
            log.info(f"AQICN Client: Loaded .env file from: {dotenv_path}")
        # else: # No warning if .env exists but empty, load_dotenv handles this
            # log.info(f"AQICN Client: .env file at {dotenv_path} processed (might be empty).")
    else:
        log.warning(f"AQICN Client: .env file not found at: {dotenv_path}. API keys must be set in environment.")
except Exception as e:
    log.error(f"AQICN Client: Error loading .env file: {e}", exc_info=True)

AQICN_TOKEN = os.getenv('AQICN_API_TOKEN')
if not AQICN_TOKEN: # Log this once at module level if token is missing after attempting to load
    log.warning("AQICN_API_TOKEN is not set. AQICN API calls will likely fail with APIKeyError.")


# --- Core API Data Fetching Function ---
def get_city_aqi_data(city_name_query):
    """Fetches full real-time AQI and pollutant data from the AQICN API.
       Expects city_name_query to be just the city name (e.g., 'Delhi').
    """
    aqicn_base_url = CONFIG.get('apis', {}).get('aqicn', {}).get('base_url', "https://api.waqi.info/feed")
    api_timeout = CONFIG.get('api_timeout_seconds', 10)

    if not AQICN_TOKEN: # This check is crucial before making the call
        msg = "AQICN_API_TOKEN not found. Please set it in .env or environment variables."
        log.error(msg)
        raise APIKeyError(msg, service="AQICN") # Raise specific error
    if not aqicn_base_url:
         msg = "AQICN base URL not found in configuration (config.yaml)."
         log.error(msg)
         raise ConfigError(msg)

    api_url = f"{aqicn_base_url}/{city_name_query}/?token={AQICN_TOKEN}"
    log.info(f"Requesting data from AQICN API: {aqicn_base_url}/{city_name_query}/?token=***TOKEN_HIDDEN***")

    try:
        response = requests.get(api_url, timeout=api_timeout)
        response.raise_for_status() 
        data = response.json()
        # log.debug(f"Raw AQICN API response JSON for '{city_name_query}': {json.dumps(data, indent=2)}")

        if data.get("status") == "ok":
            log.info(f"Successfully received 'ok' status from AQICN for '{city_name_query}'.")
            return data
        elif data.get("status") == "error":
            error_message = data.get("data", "Unknown API error reason")
            log.error(f"AQICN API returned error status for '{city_name_query}': {error_message}")
            if "Unknown station" in str(error_message):
                 log.warning(f"City/Station '{city_name_query}' resulted in 'Unknown station' from AQICN API.")
                 return None 
            raise APIError(f"AQICN API error: {error_message}", service="AQICN")
        else:
            msg = f"Received unexpected/missing status from AQICN for '{city_name_query}': {data.get('status', 'Status not present')}"
            log.error(msg); raise APIError(msg, service="AQICN")
    except requests.exceptions.Timeout as e:
        msg = f"Request to AQICN API timed out for '{city_name_query}' at URL {api_url}."
        log.error(msg); raise APITimeoutError(msg, service="AQICN") from e
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        response_text_snippet = http_err.response.text[:200] if hasattr(http_err.response, 'text') else "N/A"
        msg = f"AQICN HTTP error for '{city_name_query}': {status_code} {http_err.response.reason}. Response: {response_text_snippet}"
        log.error(msg)
        if status_code == 401: raise APIKeyError(f"AQICN Authorization failed (401) for query '{city_name_query}'. Check token.", service="AQICN") from http_err
        elif status_code == 404: raise APINotFoundError(f"AQICN endpoint or city query '{city_name_query}' not found (404).", service="AQICN") from http_err
        else: raise APIError(f"AQICN HTTP error {status_code} for query '{city_name_query}'.", status_code=status_code, service="AQICN") from http_err
    except requests.exceptions.RequestException as req_err:
        msg = f"AQICN request error for '{city_name_query}': {req_err}" # E.g., ConnectionError
        log.error(msg); raise APIError(msg, service="AQICN") from req_err
    except (ValueError, json.JSONDecodeError) as json_err:
        response_text_snippet = response.text[:200] if 'response' in locals() and hasattr(response, 'text') else 'Response object not available or no text attribute'
        msg = f"AQICN JSON decoding error for '{city_name_query}': {json_err}. Response snippet: {response_text_snippet}"
        log.error(msg); raise ValueError(msg) from json_err

# --- Helper and Wrapper Functions ---

def _extract_city_for_aqicn(city_name_full):
    """Helper to get just the city name part for AQICN query if "City, Country" is passed."""
    return city_name_full.split(',')[0].strip()

def _create_error_dict_current_aqi(city_name_part, error_message, station_name="Error"):
    """Standardized error dictionary for get_current_aqi_for_city."""
    return {'city': city_name_part, 'aqi': None, 'station': station_name, 'time': None, 'error': error_message}

def get_current_aqi_for_city(city_name_full): 
    """Fetches and extracts current AQI. Expects city_name_full like 'Delhi, India'."""
    city_query = _extract_city_for_aqicn(city_name_full)
    log.info(f"Getting current AQI (Sec 3) for '{city_name_full}' (querying AQICN as '{city_query}')")
    try:
        full_data_response = get_city_aqi_data(city_query)
        if full_data_response is None: # "Unknown station"
             return _create_error_dict_current_aqi(city_query, "Station not found by AQICN.", station_name="Unknown station")

        api_data = full_data_response.get("data", {})
        aqi_raw = api_data.get("aqi")
        station_name_reported = api_data.get("city", {}).get("name", city_query)
        timestamp_reported = api_data.get("time", {}).get("s")
        
        if aqi_raw is None or str(aqi_raw).strip() == '' or str(aqi_raw).strip() == '-':
            log.warning(f"AQI value missing or N/A for '{city_query}'. Raw: '{aqi_raw}'")
            return _create_error_dict_current_aqi(city_query, "AQI value not reported by station.", station_name=station_name_reported)
        
        return {"city": city_query, "aqi": int(aqi_raw), "station": station_name_reported, "time": timestamp_reported}
        
    except (APIKeyError, ConfigError, APITimeoutError, APINotFoundError, APIError, ValueError) as e:
         log.error(f"Handled API exception for current AQI of '{city_name_full}': {e}")
         return _create_error_dict_current_aqi(city_query, f"API Error: {str(e)}")
    except Exception as e:
         log.error(f"Unexpected error in get_current_aqi_for_city for '{city_name_full}': {e}", exc_info=True)
         return _create_error_dict_current_aqi(city_query, "Unexpected internal error.")


def _create_error_dict_pollutant_risks(city_name_part, error_message):
    return {'city': city_name_part, 'time': None, 'pollutants': {}, 'risks': [], 'error': error_message}

def get_current_pollutant_risks_for_city(city_name_full):
    """Fetches pollutant data and interprets risks. Expects city_name_full like 'Delhi, India'."""
    city_query = _extract_city_for_aqicn(city_name_full)
    log.info(f"Getting current pollutant risks (Sec 5) for '{city_name_full}' (querying AQICN as '{city_query}')")
    try:
        full_data_response = get_city_aqi_data(city_query)
        if full_data_response is None:
             return _create_error_dict_pollutant_risks(city_query, "Station not found by AQICN.")

        api_data = full_data_response.get("data", {})
        iaqi_data = api_data.get("iaqi")
        timestamp_reported = api_data.get("time", {}).get("s")

        if iaqi_data and timestamp_reported:
            risk_list = interpret_pollutant_risks(iaqi_data)
            return {"city": city_query, "time": timestamp_reported, "pollutants": iaqi_data, "risks": risk_list}
        else:
            log.error(f"Could not extract 'iaqi' or 'time' for risk interpretation for '{city_query}'.")
            return _create_error_dict_pollutant_risks(city_query, 'Pollutant data or timestamp missing.')
    except (APIKeyError, ConfigError, APITimeoutError, APINotFoundError, APIError, ValueError) as e:
         log.error(f"Handled API exception for pollutant risks of '{city_name_full}': {e}")
         return _create_error_dict_pollutant_risks(city_query, f"API Error: {str(e)}")
    except Exception as e:
         log.error(f"Unexpected error getting pollutant risks for '{city_name_full}': {e}", exc_info=True)
         return _create_error_dict_pollutant_risks(city_query, 'Unexpected internal error.')


# --- Example Usage Block (for direct script execution) ---
if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                            format='%(asctime)s - [%(levelname)s] - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
        log.info("Configured basic logging for direct script run of client.py.")

    print("\n" + "="*40)
    print(" Running api_integration/client.py Tests ")
    print("="*40 + "\n")
    
    # Map simple city names (for AQICN query) to the "full name" format Dash might pass
    test_cities_map = {
        "Delhi": "Delhi, India",
        "Mumbai": "Mumbai, India",
        "Bangalore": "Bangalore, India",
        "InvalidCityTest": "InvalidCityTest, UnknownCountry" # Used to test invalid city handling
    }

    for city_simple_query, city_full_input in test_cities_map.items():
        print(f"\n--- Testing for City Query: '{city_simple_query}' (Input to wrappers: '{city_full_input}') ---")

        # Test 1: Fetching FULL data using get_city_aqi_data (expects simple city name)
        print(f"\n[Test Case 1.1: get_city_aqi_data for '{city_simple_query}']")
        try:
            city_data_full = get_city_aqi_data(city_simple_query) 
            if city_data_full and city_data_full.get("status") == "ok":
                print(f"  SUCCESS: Full data. AQI: {city_data_full.get('data', {}).get('aqi')}, Station: {city_data_full.get('data', {}).get('city', {}).get('name')}")
            elif city_data_full is None and city_simple_query == "InvalidCityTest":
                print(f"  SUCCESS (Expected): get_city_aqi_data returned None for invalid city '{city_simple_query}'.")
            else:
                print(f"  FAILURE or API error for FULL data for '{city_simple_query}'. Response: {city_data_full}")
        except Exception as e_test:
            print(f"  ERROR during get_city_aqi_data test for '{city_simple_query}': {e_test}")


        # Test 2: Fetching CURRENT AQI using wrapper (expects "City, Country" format)
        print(f"\n[Test Case 1.2: get_current_aqi_for_city for '{city_full_input}']")
        current_aqi_info = get_current_aqi_for_city(city_full_input)
        if current_aqi_info and current_aqi_info.get('aqi') is not None and 'error' not in current_aqi_info: # Success if AQI present and no error key
            print(f"  SUCCESS: Current AQI for {city_simple_query}: AQI: {current_aqi_info.get('aqi')}, Station: {current_aqi_info.get('station')}")
        elif current_aqi_info and current_aqi_info.get('error'):
             print(f"  Handled (Expected or Actual Error): Error for '{city_full_input}': {current_aqi_info.get('error')}")
        else:
            print(f"  FAILURE: Unexpected response for current AQI for '{city_full_input}'. Response: {current_aqi_info}")

        # Test 3: Fetching CURRENT POLLUTANT RISKS using wrapper
        print(f"\n[Test Case 1.3: get_current_pollutant_risks_for_city for '{city_full_input}']")
        current_risks_info = get_current_pollutant_risks_for_city(city_full_input)
        if current_risks_info and 'error' not in current_risks_info: # Success if no error key
            print(f"  SUCCESS: Current Pollutant Risks for {city_simple_query}: Risks found: {len(current_risks_info.get('risks', []))}")
            if current_risks_info.get('risks'):
                for risk in current_risks_info['risks']: print(f"    - {risk}")
            else: print("    - None")
        elif current_risks_info and current_risks_info.get('error'):
            print(f"  Handled (Expected or Actual Error): Error for '{city_full_input}': {current_risks_info.get('error')}")
        else:
            print(f"  FAILURE: Unexpected response for current pollutant risks for '{city_full_input}'. Response: {current_risks_info}")

    print("\n[Test Case 2: Manual API Key Missing Simulation]")
    print("  (To fully test APIKeyError, temporarily remove/comment AQICN_API_TOKEN in .env and re-run this script)")
    print("  Attempting call with (potentially) missing token to see if APIKeyError is raised by get_city_aqi_data...")
    # This test relies on AQICN_TOKEN potentially being None if not set.
    # A more robust test would mock os.getenv to return None.
    if not AQICN_TOKEN:
        try:
            get_city_aqi_data("Delhi") # Should raise APIKeyError if token is truly missing
            print("  FAILURE (API Key Test): Expected APIKeyError, but no exception was raised.")
        except APIKeyError:
            print("  SUCCESS (API Key Test): Correctly caught APIKeyError for missing token.")
        except Exception as e_token_test:
            print(f"  FAILURE (API Key Test): Expected APIKeyError, got {type(e_token_test)}: {e_token_test}")
    else:
        print("  SKIPPED (API Key Test): AQICN_TOKEN is set. Cannot simulate missing key without manual .env change.")


    print("\n" + "="*40)
    print(" api_integration/client.py Tests Finished ")
    print("="*40 + "\n")