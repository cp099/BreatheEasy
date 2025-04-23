# File: src/api_integration/weather_client.py

import requests
import os
import logging # Standard logging import
from dotenv import load_dotenv
from datetime import datetime
import sys
import json # For JSONDecodeError check

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

# --- Import Configuration & Exceptions ---
# This import will also trigger the centralized logging setup in config_loader
try:
    from src.config_loader import CONFIG
    from src.exceptions import APIKeyError, APITimeoutError, APINotFoundError, APIError, ConfigError
    # Logging is configured by config_loader now
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"Weather Client: Could not import dependencies. Error: {e}", exc_info=True)
    CONFIG = {}
    # Define dummy exceptions
    class APIKeyError(Exception): pass
    class APITimeoutError(Exception): pass
    class APINotFoundError(Exception): pass
    class APIError(Exception): pass
    class ConfigError(Exception): pass
except Exception as e:
     logging.basicConfig(level=logging.WARNING)
     logging.error(f"Weather Client: Error importing dependencies: {e}", exc_info=True)
     CONFIG = {}
     class APIKeyError(Exception): pass
     class APITimeoutError(Exception): pass
     class APINotFoundError(Exception): pass
     class APIError(Exception): pass
     class ConfigError(Exception): pass


# --- Get Logger ---
log = logging.getLogger(__name__)

# --- Load API Key ---
# (Keep existing dotenv loading logic)
try:
    dotenv_path = os.path.join(PROJECT_ROOT, '.env')
    loaded = load_dotenv(dotenv_path=dotenv_path)
    if loaded:
         log.info(f"Weather Client: Loaded .env file from: {dotenv_path}")
    else:
         log.warning(f"Weather Client: .env file not found or empty at: {dotenv_path}")
except Exception as e:
    log.error(f"Weather Client: Error loading .env file: {e}", exc_info=True)

WEATHERAPI_KEY = os.getenv('WEATHERAPI_API_KEY')

# --- API Client Function using WeatherAPI.com ---
def get_current_weather(city_name):
    """
    Fetches current weather data for a city using WeatherAPI.com service and config URL.

    Returns:
        dict: A dictionary containing key weather information suitable for UI.
              Returns None ONLY if the API successfully responds but indicates the city
              was not found (e.g., error code 1006).

    Raises:
        APIKeyError: If WEATHERAPI_API_KEY is missing or invalid (HTTP 401).
        ConfigError: If WeatherAPI base URL is missing in config.
        APITimeoutError: If the request times out.
        APINotFoundError: If the API returns HTTP 404 (endpoint not found).
        APIError: For other HTTP errors (like 403 Forbidden) or unexpected API issues.
        ValueError: If the response is not valid JSON.
        requests.exceptions.RequestException: For other network/connection issues.
    """
    weatherapi_base_url = CONFIG.get('apis', {}).get('weatherapi', {}).get('base_url', "http://api.weatherapi.com/v1/current.json")

    if not WEATHERAPI_KEY:
        msg = "WEATHERAPI_API_KEY not found in environment variables."
        log.error(msg)
        raise APIKeyError(msg, service="WeatherAPI")
    if not weatherapi_base_url:
         msg = "WeatherAPI base URL not found in configuration."
         log.error(msg)
         raise ConfigError(msg)

    params = {'key': WEATHERAPI_KEY, 'q': city_name, 'aqi': 'no'}
    log.info(f"Requesting weather from WeatherAPI for: {city_name} using URL: {weatherapi_base_url}")

    try:
        response = requests.get(weatherapi_base_url, params=params, timeout=10)
        response.raise_for_status() # Raise HTTPError for 4xx/5xx
        data = response.json()      # Raise JSONDecodeError if not JSON
        log.debug(f"Raw WeatherAPI response JSON: {data}")

        # Check for application-level errors within the JSON response
        if "error" in data:
            error_info = data["error"]
            error_msg = error_info.get('message', 'Unknown API error')
            error_code = error_info.get('code')
            log.error(f"WeatherAPI returned error for '{city_name}': {error_msg} (Code: {error_code})")
            # Specific handling for 'city not found' - return None as it's expected data state
            if error_code == 1006:
                 log.warning(f"City '{city_name}' not found by WeatherAPI.")
                 return None
            # For other API errors (like bad key code), raise a specific exception
            elif error_code in [2006, 2007, 2008, 1002, 1003, 1005]: # Example codes for key issues/request errors
                 raise APIKeyError(f"WeatherAPI Error Code {error_code}: {error_msg}", service="WeatherAPI")
            else:
                 raise APIError(f"WeatherAPI Error Code {error_code}: {error_msg}", service="WeatherAPI")

        # --- Extract Data if no error field ---
        current_data = data.get("current", {})
        location_data = data.get("location", {})
        condition_data = current_data.get("condition", {})
        timestamp_local_str = location_data.get("localtime")

        weather_info = {
            "temp_c": current_data.get("temp_c"), "feelslike_c": current_data.get("feelslike_c"),
            "humidity": current_data.get("humidity"), "pressure_mb": current_data.get("pressure_mb"),
            "condition_text": condition_data.get("text"), "condition_icon": condition_data.get("icon"),
            "wind_kph": current_data.get("wind_kph"), "wind_dir": current_data.get("wind_dir"),
            "uv_index": current_data.get("uv"), "city": location_data.get("name"),
            "region": location_data.get("region"), "country": location_data.get("country"),
            "last_updated": current_data.get("last_updated"), "localtime": timestamp_local_str
        }
        log.info(f"Processed weather info for {city_name}: Temp={weather_info.get('temp_c')}C, Condition={weather_info.get('condition_text')}")
        return weather_info

    except requests.exceptions.Timeout as e:
        msg = f"Request to WeatherAPI timed out for '{city_name}'."
        log.error(msg)
        raise APITimeoutError(msg, service="WeatherAPI") from e
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code
        msg = f"WeatherAPI HTTP error: {http_err} - Status: {status_code}"
        log.error(msg)
        if status_code == 401:
             raise APIKeyError("Authorization failed (HTTP 401). Check WEATHERAPI_API_KEY.", service="WeatherAPI") from http_err
        elif status_code == 403:
             raise APIError("Forbidden (HTTP 403). Check API key status/plan limits.", status_code=status_code, service="WeatherAPI") from http_err
        elif status_code == 404: # Endpoint itself not found
             raise APINotFoundError(f"WeatherAPI endpoint not found ({weatherapi_base_url}).", service="WeatherAPI") from http_err
        else: # Other 4xx/5xx errors
             raise APIError(f"HTTP error {status_code}", status_code=status_code, service="WeatherAPI") from http_err
    except requests.exceptions.RequestException as req_err:
        msg = f"WeatherAPI network error: {req_err}"
        log.error(msg)
        raise APIError(msg, service="WeatherAPI") from req_err # Wrap network errors
    except (ValueError, json.JSONDecodeError) as json_err:
        msg = f"WeatherAPI JSON decoding error: {json_err}"
        log.error(msg)
        raise ValueError(msg) from json_err # Re-raise standard ValueError

# --- Example Usage Block (Adjusted to catch exceptions) ---
if __name__ == "__main__":
    # Logging configured by importing CONFIG above
    print("\n" + "="*30)
    print(" Running weather_client.py Tests (using WeatherAPI.com) ")
    print("="*30 + "\n")

    # Import exceptions needed for testing here
    try:
         from src.exceptions import APIKeyError, APITimeoutError, APINotFoundError, APIError, ConfigError
    except ImportError:
         # Define dummies if needed, though config_loader import should handle path
         class APIKeyError(Exception): pass
         class APITimeoutError(Exception): pass
         class APINotFoundError(Exception): pass
         class APIError(Exception): pass
         class ConfigError(Exception): pass


    test_cities = ["Delhi, India", "London", "Atlantisxyz"]

    for city in test_cities:
        print(f"--- Test: Fetching weather for '{city}' ---")
        try:
            weather_data = get_current_weather(city) # This might raise exceptions

            if weather_data:
                print(f"Success! Received weather data for '{city}':")
                import pprint
                pprint.pprint(weather_data)
            else:
                # This specific case is for API code 1006 (city not found in JSON response)
                print(f"API reported city not found for '{city}'.")

        # Catch specific custom exceptions first
        except APIKeyError as e:
             print(f"Failure! API Key Error occurred for '{city}': {e}")
        except APITimeoutError as e:
             print(f"Failure! Timeout Error occurred for '{city}': {e}")
        except APINotFoundError as e: # For 404 errors
             print(f"Failure! API Endpoint Not Found Error occurred for '{city}': {e}")
        except APIError as e: # Catch other API/HTTP errors (like 400, 403, 5xx)
             print(f"Failure! API Error occurred for '{city}': {e}")
        except ConfigError as e:
             print(f"Failure! Configuration Error occurred for '{city}': {e}")
        except ValueError as e: # Catch JSON errors
             print(f"Failure! Value Error (likely JSON) occurred for '{city}': {e}")
        except requests.exceptions.RequestException as e: # Catch other network errors
             print(f"Failure! Network Request Error occurred for '{city}': {e}")
        except Exception as e: # Catch any other unexpected error
             print(f"Failure! An unexpected error occurred for '{city}': {e}")
             log.error(f"Unexpected error during weather client test for {city}", exc_info=True)

        print("-" * 20) # Separator

    # Test API key missing (Manual Simulation needed - would now raise APIKeyError)
    print("\n--- Test: API Key Missing Check ---")
    print("(Requires manually removing/commenting WeatherAPI key in .env or code - should raise APIKeyError)")

    print("\n" + "="*30)
    print(" weather_client.py (WeatherAPI.com) Tests Finished ")
    print("="*30 + "\n")