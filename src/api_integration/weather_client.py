# File: src/api_integration/weather_client.py

"""
Handles interactions with the WeatherAPI.com service.

Provides a function to fetch current weather conditions for a specified city.
Requires a WEATHERAPI_API_KEY environment variable (loaded from .env).
API base URL is configurable via config/config.yaml.
Handles various API errors and network issues by raising specific exceptions.
"""

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
# (Keep existing import logic with fallbacks)
try:
    from src.config_loader import CONFIG
    from src.exceptions import APIKeyError, APITimeoutError, APINotFoundError, APIError, ConfigError
    # Logging configured by config_loader
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"Weather Client: Could not import dependencies. Error: {e}", exc_info=True)
    CONFIG = {}
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
    """Fetches current weather data using the WeatherAPI.com service.

    Constructs the request using the base URL from configuration and the
    API key from environment variables. Parses the response and extracts
    key weather metrics.

    Args:
        city_name (str): The name of the city (e.g., 'Delhi, India', 'London').
                         Using "City, Country" is recommended for accuracy.

    Returns:
        dict or None: A dictionary containing simplified weather information:
                      - temp_c (float): Temperature in Celsius.
                      - feelslike_c (float): Feels like temperature in Celsius.
                      - humidity (int): Humidity percentage.
                      - pressure_mb (float): Pressure in millibars.
                      - condition_text (str): Text description (e.g., 'Partly cloudy').
                      - condition_icon (str): URL path for weather icon.
                      - wind_kph (float): Wind speed in km/h.
                      - wind_dir (str): Wind direction (e.g., 'WNW').
                      - uv_index (float): UV index.
                      - city (str): Name of the location returned by API.
                      - region (str): Region/state returned by API.
                      - country (str): Country returned by API.
                      - last_updated (str): Local time string of last update.
                      - localtime (str): Local time string for the location.
                      Returns None ONLY if the API successfully responds but indicates
                      the city was not found (e.g., error code 1006).

    Raises:
        APIKeyError: If WEATHERAPI_API_KEY is missing or invalid (HTTP 401/403 or specific API codes).
        ConfigError: If WeatherAPI base URL is missing in configuration.
        APITimeoutError: If the request times out.
        APINotFoundError: If the API endpoint returns HTTP 404.
        APIError: For other HTTP errors (e.g., 400, 403, 5xx) or other API-specific errors returned in the JSON payload.
        ValueError: If the API response is not valid JSON.
        requests.exceptions.RequestException: For underlying network connection issues (wrapped in APIError).
    """
    # (Function code remains the same)
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
        response.raise_for_status()
        data = response.json()
        log.debug(f"Raw WeatherAPI response JSON: {data}")

        if "error" in data:
            error_info = data["error"]
            error_msg = error_info.get('message', 'Unknown API error')
            error_code = error_info.get('code')
            log.error(f"WeatherAPI returned error for '{city_name}': {error_msg} (Code: {error_code})")
            if error_code == 1006:
                 log.warning(f"City '{city_name}' not found by WeatherAPI.")
                 return None # Special case: return None for 'city not found'
            elif error_code in [2006, 2007, 2008, 1002, 1003, 1005]:
                 raise APIKeyError(f"WeatherAPI Error Code {error_code}: {error_msg}", service="WeatherAPI")
            else:
                 raise APIError(f"WeatherAPI Error Code {error_code}: {error_msg}", service="WeatherAPI")

        # Extract Data
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
        if status_code == 401: raise APIKeyError("Authorization failed (HTTP 401). Check WEATHERAPI_API_KEY.", service="WeatherAPI") from http_err
        elif status_code == 403: raise APIError("Forbidden (HTTP 403). Check API key status/plan limits.", status_code=status_code, service="WeatherAPI") from http_err
        elif status_code == 404: raise APINotFoundError(f"WeatherAPI endpoint not found ({weatherapi_base_url}).", service="WeatherAPI") from http_err
        else: raise APIError(f"HTTP error {status_code}", status_code=status_code, service="WeatherAPI") from http_err
    except requests.exceptions.RequestException as req_err:
        msg = f"WeatherAPI network error: {req_err}"
        log.error(msg)
        raise APIError(msg, service="WeatherAPI") from req_err
    except (ValueError, json.JSONDecodeError) as json_err:
        msg = f"WeatherAPI JSON decoding error: {json_err}"
        log.error(msg)
        raise ValueError(msg) from json_err

# --- Example Usage Block ---
# (Keep existing __main__ block as is, including the try/except around get_current_weather)
if __name__ == "__main__":
    # ... (test code remains the same) ...
    pass # Added pass for valid syntax if test code removed/commented