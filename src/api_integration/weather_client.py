# File: src/api_integration/weather_client.py

import requests
import os
import logging # Standard logging import
from dotenv import load_dotenv
from datetime import datetime
import sys

# --- Setup Project Root Path ---
# (Keep existing PROJECT_ROOT logic)
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')):
         PROJECT_ROOT = os.path.abspath('.')
    logging.warning(f"Weather Client: __file__ not defined. Assuming project root: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)
     logging.info(f"Weather Client: Added project root to sys.path: {PROJECT_ROOT}")

# --- Import Configuration ---
# This import will also trigger the centralized logging setup in config_loader
try:
    from src.config_loader import CONFIG
    logging.info("Weather Client: Successfully imported config.")
except ImportError as e:
    # Temporary basic config if imports fail
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"Weather Client: Could not import CONFIG. Error: {e}", exc_info=True)
    CONFIG = {}
except Exception as e:
     logging.basicConfig(level=logging.WARNING)
     logging.error(f"Weather Client: Error importing config: {e}", exc_info=True)
     CONFIG = {}

# --- Get Logger ---
# Get the logger instance for this module
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
    """
    # Get Base URL from config, with a fallback default
    weatherapi_base_url = CONFIG.get('apis', {}).get('weatherapi', {}).get('base_url', "http://api.weatherapi.com/v1/current.json")

    if not WEATHERAPI_KEY:
        log.error("WEATHERAPI_API_KEY not found in environment variables. Check .env file.")
        return None
    if not weatherapi_base_url:
         log.error("WeatherAPI base URL not found in configuration.")
         return None

    params = {'key': WEATHERAPI_KEY, 'q': city_name, 'aqi': 'no'}
    log.info(f"Requesting weather from WeatherAPI for: {city_name} using URL: {weatherapi_base_url}")

    try:
        response = requests.get(weatherapi_base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        log.debug(f"Raw WeatherAPI response JSON: {data}")

        if "error" in data:
            error_info = data["error"]
            log.error(f"WeatherAPI error for '{city_name}': {error_info.get('message')} (Code: {error_info.get('code')})")
            if error_info.get('code') == 1006: log.warning(f"City '{city_name}' not found by WeatherAPI.")
            return None

        # Extract Data
        current_data = data.get("current", {})
        location_data = data.get("location", {})
        condition_data = current_data.get("condition", {})
        timestamp_local_str = location_data.get("localtime")

        weather_info = {
            "temp_c": current_data.get("temp_c"),
            "feelslike_c": current_data.get("feelslike_c"),
            "humidity": current_data.get("humidity"),
            "pressure_mb": current_data.get("pressure_mb"),
            "condition_text": condition_data.get("text"),
            "condition_icon": condition_data.get("icon"),
            "wind_kph": current_data.get("wind_kph"),
            "wind_dir": current_data.get("wind_dir"),
            "uv_index": current_data.get("uv"),
            "city": location_data.get("name"),
            "region": location_data.get("region"),
            "country": location_data.get("country"),
            "last_updated": current_data.get("last_updated"),
            "localtime": timestamp_local_str
        }
        log.info(f"Processed weather info for {city_name}: Temp={weather_info['temp_c']}C, Condition={weather_info['condition_text']}")
        return weather_info

    # (Keep existing except blocks)
    except requests.exceptions.Timeout:
        log.error(f"Request to WeatherAPI timed out for '{city_name}'.")
        return None
    except requests.exceptions.HTTPError as http_err:
        log.error(f"WeatherAPI HTTP error: {http_err} - Status: {http_err.response.status_code}")
        if http_err.response.status_code == 401: log.error("Check WEATHERAPI_API_KEY.")
        elif http_err.response.status_code == 403: log.error("Check WeatherAPI key status/limits.")
        return None
    except requests.exceptions.RequestException as req_err:
        log.error(f"WeatherAPI network error: {req_err}")
        return None
    except ValueError as json_err:
        log.error(f"WeatherAPI JSON decoding error: {json_err}")
        return None
    except Exception as e:
        log.error(f"Unexpected error in get_current_weather (WeatherAPI): {e}", exc_info=True)
        return None

# --- Example Usage Block (No changes needed here) ---
if __name__ == "__main__":
    # Logging configured by importing CONFIG above
    print("\n" + "="*30)
    print(" Running weather_client.py Tests (using WeatherAPI.com) ")
    print("="*30 + "\n")
    # (Keep the existing test block code exactly as it was)
    test_cities = ["Delhi, India", "London", "Atlantisxyz"]
    for city in test_cities:
        print(f"--- Test: Fetching weather for '{city}' ---")
        weather_data = get_current_weather(city)
        if weather_data:
            print(f"Success! Received weather data for '{city}':")
            import pprint
            pprint.pprint(weather_data)
        else: print(f"Failure or city not found for '{city}'. Check logs.")
        print("-" * 20)
    print("\n--- Test: API Key Missing Check ---")
    print("(Requires manually removing/commenting WeatherAPI key in .env or code)")
    print("\n" + "="*30)
    print(" weather_client.py (WeatherAPI.com) Tests Finished ")
    print("="*30 + "\n")