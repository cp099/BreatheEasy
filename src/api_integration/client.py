# File: src/api_integration/client.py

import requests
import os
import logging
from dotenv import load_dotenv
import sys # For path manipulation

# --- Setup Project Root Path ---
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')):
         PROJECT_ROOT = os.path.abspath('.')
    logging.warning(f"AQICN Client: __file__ not defined. Assuming project root: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)
     logging.info(f"AQICN Client: Added project root to sys.path: {PROJECT_ROOT}")

# --- Import Configuration and Other Modules ---
try:
    from src.config_loader import CONFIG
    from src.health_rules.interpreter import interpret_pollutant_risks
    log_level_str = CONFIG.get('logging', {}).get('level', 'INFO')
    log_format = CONFIG.get('logging', {}).get('format', '%(asctime)s - [%(levelname)s] - %(message)s')
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.info("AQICN Client: Successfully imported config and interpreter.")
except ImportError as e:
    logging.error(f"AQICN Client: Could not import dependencies. Error: {e}", exc_info=True)
    # Set CONFIG to empty dict to allow script to run but fail gracefully later
    CONFIG = {}
    log_level = logging.INFO
    log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    # Define a dummy function if interpreter fails to import, so later calls don't raise NameError
    def interpret_pollutant_risks(iaqi_data):
        logging.error("interpret_pollutant_risks function unavailable due to import error.")
        return []

# --- Setup Logging ---
logging.basicConfig(level=log_level, format=log_format, force=True) # force=True might be needed if logger was configured elsewhere
log = logging.getLogger(__name__)

# --- Load API Token ---
# Load environment variables from .env file in the project root
try:
    dotenv_path = os.path.join(PROJECT_ROOT, '.env')
    loaded = load_dotenv(dotenv_path=dotenv_path)
    if loaded:
         log.info(f"AQICN Client: Loaded .env file from: {dotenv_path}")
    else:
         log.warning(f"AQICN Client: .env file not found or empty at: {dotenv_path}")
except Exception as e:
    log.error(f"AQICN Client: Error finding or loading .env file: {e}", exc_info=True)

AQICN_TOKEN = os.getenv('AQICN_API_TOKEN') # Get token AFTER loading .env

# --- API Client Function ---
def get_city_aqi_data(city_name):
    """
    Fetches real-time AQI/pollutant data for a city from AQICN API using config URL.
    """
    # Get Base URL from config, with a default fallback
    aqicn_base_url = CONFIG.get('apis', {}).get('aqicn', {}).get('base_url', "https://api.waqi.info/feed")

    if not AQICN_TOKEN:
        log.error("AQICN_API_TOKEN not found in environment variables. Check .env file and ensure it's loaded.")
        return None
    if not aqicn_base_url: # Should not happen with default, but check
         log.error("AQICN base URL not found in configuration.")
         return None

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
            error_message = data.get("data", "Unknown error")
            log.error(f"AQICN API error for '{city_name}': {error_message}")
            if "Unknown station" in str(error_message):
                 log.warning(f"City/Station '{city_name}' not found by AQICN API.")
            return None
        else:
            log.error(f"Unexpected status from AQICN for '{city_name}': {data.get('status', 'Missing')}")
            return None
    # (Keep existing except blocks for Timeout, HTTPError, RequestException, ValueError, Exception)
    except requests.exceptions.Timeout:
        log.error(f"Request to AQICN timed out for '{city_name}'.")
        return None
    except requests.exceptions.HTTPError as http_err:
        log.error(f"AQICN HTTP error: {http_err} - Status: {http_err.response.status_code}")
        if http_err.response.status_code == 401: log.error("Check AQICN_API_TOKEN.")
        return None
    except requests.exceptions.RequestException as req_err:
        log.error(f"AQICN request error: {req_err}")
        return None
    except ValueError as json_err: # Includes JSONDecodeError
        log.error(f"AQICN JSON decoding error: {json_err}")
        return None
    except Exception as e:
        log.error(f"Unexpected error in get_city_aqi_data: {e}", exc_info=True)
        return None


# --- Function for Section 3 ---
def get_current_aqi_for_city(city_name):
    """Fetches current overall AQI for a city."""
    log.info(f"Getting current AQI value (Sec 3) for: {city_name}")
    full_data = get_city_aqi_data(city_name)
    if full_data and full_data.get("status") == "ok" and "data" in full_data:
        api_data = full_data["data"]
        try:
            aqi_raw = api_data.get("aqi") # Get raw AQI which might be non-integer
            if aqi_raw is None or aqi_raw == '-': # AQICN sometimes uses '-' for N/A
                 log.warning(f"AQI value missing or N/A in API data for {city_name}.")
                 return None
            aqi = int(aqi_raw) # Convert valid AQI to integer
            station_name = api_data.get("city", {}).get("name", city_name)
            timestamp = api_data.get("time", {}).get("s")

            if timestamp is not None: # AQI already checked
                result = {"city": city_name, "aqi": aqi, "station": station_name, "time": timestamp}
                log.info(f"Extracted current AQI info for {city_name}: {result}")
                return result
            else:
                log.error(f"Could not extract 'time' from API data for {city_name}. Data: {api_data}")
                return None
        except (ValueError, TypeError) as e:
             log.error(f"Error processing extracted AQI data for {city_name}: {e}. Raw AQI: {api_data.get('aqi')}.", exc_info=True)
             return None
        except Exception as e:
             log.error(f"Unexpected error extracting AQI data for {city_name}: {e}", exc_info=True)
             return None
    else:
        log.warning(f"Failed to get valid API response for {city_name} to extract current AQI.")
        return None

# --- Function for Section 5 ---
def get_current_pollutant_risks_for_city(city_name):
    """Fetches pollutant data and interprets health risks."""
    # (Logic remains the same, relies on get_city_aqi_data which now uses config)
    log.info(f"Getting current pollutant risks (Section 5) for city: {city_name}")
    full_data = get_city_aqi_data(city_name)
    if full_data and full_data.get("status") == "ok" and "data" in full_data:
        api_data = full_data["data"]
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
    else:
        log.warning(f"Failed to get valid API response for {city_name} for risk interpretation.")
        return None

# --- Example Usage Block (Should still work) ---
if __name__ == "__main__":
    print("\n" + "="*30)
    print(" Running api_integration/client.py Tests ")
    print("="*30 + "\n")
    # (Keep the existing test block code exactly as it was)
    # ... It will now use the config-driven base URL indirectly ...
    # --- Test 1: Fetching FULL data for a valid city ---
    test_city_full = "Delhi"
    print(f"[Test 1: Fetching FULL data for '{test_city_full}']")
    city_data_full = get_city_aqi_data(test_city_full)
    if city_data_full and city_data_full.get("status") == "ok": print(f"Success! Received FULL data for '{test_city_full}'.")
    else: print(f"Failure! Could not retrieve valid FULL data for '{test_city_full}'.")
    # --- Test 2: Fetching ONLY CURRENT AQI for Section 3 ---
    test_city_sec3 = "Mumbai"
    print(f"\n[Test 2: Fetching CURRENT AQI (Section 3) for '{test_city_sec3}']")
    current_aqi_info = get_current_aqi_for_city(test_city_sec3)
    if current_aqi_info:
         print(f"Success! Received Current AQI Info for {test_city_sec3}:")
         print(f"  AQI: {current_aqi_info.get('aqi')}")
         print(f"  Station: {current_aqi_info.get('station')}")
         print(f"  Timestamp: {current_aqi_info.get('time')}")
    else: print(f"Failure! Could not retrieve current AQI info for {test_city_sec3}.")
    # --- Test 3: Fetching data for an invalid city ---
    print("\n[Test 3: Fetching data for 'Atlantis']")
    invalid_city_data = get_city_aqi_data("Atlantis")
    if invalid_city_data is None: print("Success! Handled non-existent city 'Atlantis' for full data.")
    else: print("Failure! Expected None for 'Atlantis'.")
    invalid_aqi_info = get_current_aqi_for_city("Atlantis")
    if invalid_aqi_info is None: print("Success! Handled non-existent city 'Atlantis' for current AQI.")
    else: print("Failure! Expected None for 'Atlantis' current AQI.")
    # --- Test 4: Check for API Key Missing ---
    print("\n[Test 4: Check for API Key Missing (Manual Simulation Needed)]")
    print("(Requires manual modification or checking logs for 'AQICN_API_TOKEN not found')")
    # --- Test 5: Get Current Pollutant Risks (Section 5) ---
    test_city_sec5 = "Delhi"
    print(f"\n[Test 5: Fetching CURRENT POLLUTANT RISKS (Section 5) for '{test_city_sec5}']")
    current_risks_info = get_current_pollutant_risks_for_city(test_city_sec5)
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