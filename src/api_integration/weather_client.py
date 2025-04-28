# File: src/api_integration/weather_client.py (Updated for Forecast)

import requests
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
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
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

# --- Import Configuration & Exceptions ---
try:
    from src.config_loader import CONFIG
    from src.exceptions import APIKeyError, APITimeoutError, APINotFoundError, APIError, ConfigError
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

log = logging.getLogger(__name__)

# --- Load API Key ---
try:
    dotenv_path = os.path.join(PROJECT_ROOT, '.env')
    loaded = load_dotenv(dotenv_path=dotenv_path)
    # Logging handled by root logger setup via CONFIG import
except Exception as e:
    log.error(f"Weather Client: Error loading .env file: {e}", exc_info=True)

WEATHERAPI_KEY = os.getenv('WEATHERAPI_API_KEY')

# --- API Endpoints ---
WEATHERAPI_CURRENT_URL = CONFIG.get('apis', {}).get('weatherapi', {}).get('base_url', "http://api.weatherapi.com/v1/current.json")
WEATHERAPI_FORECAST_URL = CONFIG.get('apis', {}).get('weatherapi', {}).get('forecast_url', "http://api.weatherapi.com/v1/forecast.json")


# --- Current Weather Function ---
def get_current_weather(city_name):
    """Fetches current weather data using WeatherAPI.com service."""
    # (Keep exact code from previous version - no changes needed here)
    if not WEATHERAPI_KEY: raise APIKeyError("WEATHERAPI_API_KEY missing.", service="WeatherAPI")
    if not WEATHERAPI_CURRENT_URL: raise ConfigError("WeatherAPI base URL missing.")
    params = {'key': WEATHERAPI_KEY, 'q': city_name, 'aqi': 'no'}
    log.info(f"Requesting current weather from WeatherAPI for: {city_name} using URL: {WEATHERAPI_CURRENT_URL}")
    try:
        response = requests.get(WEATHERAPI_CURRENT_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            error_info = data["error"]
            error_msg = error_info.get('message', 'Unknown API error')
            error_code = error_info.get('code')
            log.error(f"WeatherAPI error for '{city_name}': {error_msg} (Code: {error_code})")
            if error_code == 1006: return None
            elif error_code in [2006, 2007, 2008, 1002, 1003, 1005]: raise APIKeyError(f"WeatherAPI Error Code {error_code}: {error_msg}", service="WeatherAPI")
            else: raise APIError(f"WeatherAPI Error Code {error_code}: {error_msg}", service="WeatherAPI")
        current_data = data.get("current", {}); location_data = data.get("location", {}); condition_data = current_data.get("condition", {})
        weather_info = {
            "temp_c": current_data.get("temp_c"), "feelslike_c": current_data.get("feelslike_c"),
            "humidity": current_data.get("humidity"), "pressure_mb": current_data.get("pressure_mb"),
            "condition_text": condition_data.get("text"), "condition_icon": condition_data.get("icon"),
            "wind_kph": current_data.get("wind_kph"), "wind_dir": current_data.get("wind_dir"),
            "uv_index": current_data.get("uv"), "city": location_data.get("name"),
            "region": location_data.get("region"), "country": location_data.get("country"),
            "last_updated": current_data.get("last_updated"), "localtime": location_data.get("localtime")}
        log.info(f"Processed current weather for {city_name}: Temp={weather_info.get('temp_c')}C, Cond={weather_info.get('condition_text')}")
        return weather_info
    except requests.exceptions.Timeout as e: raise APITimeoutError(f"WeatherAPI current timed out for '{city_name}'.", service="WeatherAPI") from e
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code; log.error(f"WeatherAPI current HTTP error: {http_err} - Status: {status_code}")
        if status_code == 401: raise APIKeyError("Auth failed (401). Check WEATHERAPI_API_KEY.", service="WeatherAPI") from http_err
        elif status_code == 403: raise APIError("Forbidden (403). Check key/plan.", status_code=status_code, service="WeatherAPI") from http_err
        elif status_code == 404: raise APINotFoundError(f"WeatherAPI current endpoint not found ({WEATHERAPI_CURRENT_URL}).", service="WeatherAPI") from http_err
        else: raise APIError(f"HTTP error {status_code}", status_code=status_code, service="WeatherAPI") from http_err
    except requests.exceptions.RequestException as req_err: raise APIError(f"WeatherAPI current network error: {req_err}", service="WeatherAPI") from req_err
    except (ValueError, json.JSONDecodeError) as json_err: raise ValueError(f"WeatherAPI current JSON decoding error: {json_err}") from json_err
    except Exception as e: raise APIError(f"Unexpected error in get_current_weather: {e}", service="WeatherAPI") from e


# --- NEW Function: Get Weather Forecast ---
def get_weather_forecast(city_name, days=3):
    """
    Fetches weather forecast data for a specific city using WeatherAPI.com service.

    Args:
        city_name (str): The name of the city (e.g., 'Delhi, India').
        days (int): Number of forecast days required (1-3 for free tier usually).

    Returns:
        list[dict] or None: A list of dictionaries, one for each forecast day,
                            containing key daily metrics. Returns None on failure
                            or if city not found (API error 1006).

    Raises:
        APIKeyError: If key is missing/invalid.
        ConfigError: If forecast URL is missing in config.
        APITimeoutError: If request times out.
        APINotFoundError: If forecast endpoint not found (HTTP 404).
        APIError: For other HTTP errors or API issues.
        ValueError: If response is not valid JSON.
        requests.exceptions.RequestException: For network issues.
    """
    if not WEATHERAPI_KEY: raise APIKeyError("WEATHERAPI_API_KEY missing.", service="WeatherAPI")
    if not WEATHERAPI_FORECAST_URL: raise ConfigError("WeatherAPI forecast URL missing.")

    # Clamp days to a reasonable range based on typical API limits
    if not 1 <= days <= 14:
         log.warning(f"Requested forecast days ({days}) outside typical range 1-14. Clamping to 3.")
         days = 3

    params = {'key': WEATHERAPI_KEY, 'q': city_name, 'days': days, 'aqi': 'no', 'alerts': 'no'}
    log.info(f"Requesting {days}-day weather forecast from WeatherAPI for: {city_name} using URL: {WEATHERAPI_FORECAST_URL}")

    try:
        response = requests.get(WEATHERAPI_FORECAST_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        log.debug(f"Raw WeatherAPI Forecast response JSON: {data}")

        if "error" in data:
            error_info = data["error"]
            error_msg = error_info.get('message', 'Unknown API error')
            error_code = error_info.get('code')
            log.error(f"WeatherAPI forecast error for '{city_name}': {error_msg} (Code: {error_code})")
            if error_code == 1006: return None # City not found
            elif error_code in [2006, 2007, 2008, 1002, 1003, 1005]: raise APIKeyError(f"WeatherAPI Error Code {error_code}: {error_msg}", service="WeatherAPI")
            else: raise APIError(f"WeatherAPI Error Code {error_code}: {error_msg}", service="WeatherAPI")

        # --- Extract Forecast Data ---
        forecast_days_data = data.get("forecast", {}).get("forecastday", [])
        processed_forecast = []

        if not forecast_days_data:
             log.warning(f"No 'forecastday' data found in WeatherAPI response for {city_name}")
             return None # Or maybe return empty list []? None signals failure better.

        for day_data in forecast_days_data:
            date_str = day_data.get("date")
            day_info = day_data.get("day", {})
            condition_info = day_info.get("condition", {})

            # Extract values needed for regressors (using names from weatherapi response)
            daily_summary = {
                "date": date_str,
                "avgtemp_c": day_info.get("avgtemp_c"),           # -> temperature_2m?
                "avghumidity": day_info.get("avghumidity"),       # -> relative_humidity_2m?
                "totalprecip_mm": day_info.get("totalprecip_mm"), # -> rain? (Note unit mm)
                "maxwind_kph": day_info.get("maxwind_kph"),       # -> wind_speed_10m? / wind_gusts_10m?
                "avgvis_km": day_info.get("avgvis_km"),           # -> cloud_cover? (Visibility is related but not the same)
                "uv": day_info.get("uv"),                         # -> (Not directly pressure/cloud)
                # Add other potentially useful fields for mapping
                "maxtemp_c": day_info.get("maxtemp_c"),
                "mintemp_c": day_info.get("mintemp_c"),
                "daily_chance_of_rain": day_info.get("daily_chance_of_rain"),
                # Missing direct forecast equivalents for pressure, specific cloud levels
                # We will need to map these available fields to your regressor columns in predictor.py
                "condition_text": condition_info.get("text"), # Keep for potential future use/display
                "condition_icon": condition_info.get("icon"), # Keep for potential future use/display
            }
            processed_forecast.append(daily_summary)

        log.info(f"Successfully processed {len(processed_forecast)}-day weather forecast for {city_name}.")
        return processed_forecast

    # (Keep existing except blocks, update service name if desired)
    except requests.exceptions.Timeout as e: raise APITimeoutError(f"WeatherAPI forecast timed out for '{city_name}'.", service="WeatherAPI") from e
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code; log.error(f"WeatherAPI forecast HTTP error: {http_err} - Status: {status_code}")
        if status_code == 401: raise APIKeyError("Auth failed (401). Check WEATHERAPI_API_KEY.", service="WeatherAPI") from http_err
        elif status_code == 403: raise APIError("Forbidden (403). Check key/plan.", status_code=status_code, service="WeatherAPI") from http_err
        elif status_code == 404: raise APINotFoundError(f"WeatherAPI forecast endpoint not found ({WEATHERAPI_FORECAST_URL}).", service="WeatherAPI") from http_err
        else: raise APIError(f"HTTP error {status_code}", status_code=status_code, service="WeatherAPI") from http_err
    except requests.exceptions.RequestException as req_err: raise APIError(f"WeatherAPI forecast network error: {req_err}", service="WeatherAPI") from req_err
    except (ValueError, json.JSONDecodeError) as json_err: raise ValueError(f"WeatherAPI forecast JSON decoding error: {json_err}") from json_err
    except Exception as e: raise APIError(f"Unexpected error in get_weather_forecast: {e}", service="WeatherAPI") from e


# --- Update Example Usage Block ---
if __name__ == "__main__":
    # Configure logging if needed (should be done by CONFIG import)
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s')

    print("\n" + "="*30)
    print(" Running weather_client.py Tests (using WeatherAPI.com) ")
    print("="*30 + "\n")

    # --- Test Current Weather ---
    # (Keep existing loop for get_current_weather)
    test_cities_current = ["Delhi, India", "London", "Atlantisxyz"]
    for city in test_cities_current:
        print(f"--- Test Current: Fetching weather for '{city}' ---")
        try: weather_data = get_current_weather(city)
        except Exception as e: weather_data = None; print(f"ERROR Caught: {e}")
        if weather_data:
            print("Success! Current Weather Data:"); import pprint; pprint.pprint(weather_data)
        else: print(f"Failure or city not found.")
        print("-" * 20)

    # --- Add Test for Forecast ---
    print("\n" + "="*30)
    print(" Testing Weather Forecast Function ")
    print("="*30 + "\n")
    forecast_city = "Delhi, India"
    forecast_days = CONFIG.get('modeling', {}).get('forecast_days', 3) # Use config value
    print(f"--- Test Forecast: Fetching {forecast_days}-day forecast for '{forecast_city}' ---")
    try:
         forecast_data = get_weather_forecast(forecast_city, days=forecast_days)
         if forecast_data:
             print(f"Success! Received {len(forecast_data)}-day forecast:")
             import pprint
             pprint.pprint(forecast_data)
         else:
             print(f"Failure or city not found for forecast.")
    except Exception as e:
         print(f"Failure! An error occurred fetching forecast: {e}")
         log.error(f"Error during forecast test for {forecast_city}", exc_info=True)


    print("\n" + "="*30)
    print(" weather_client.py Tests Finished ")
    print("="*30 + "\n")