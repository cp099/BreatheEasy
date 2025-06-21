# File: src/api_integration/weather_client.py

"""
Handles interactions with the WeatherAPI.com service.

Provides functions to fetch current weather conditions and multi-day weather
forecasts. This module includes built-in, configurable retry logic for handling
transient network or server-side errors.

Dependencies:
- WEATHERAPI_API_KEY environment variable (loaded from .env).
- API base URLs and retry settings are configured via config/config.yaml.
"""

import requests
import os
import logging
from dotenv import load_dotenv
import sys
import json
import time

# --- Setup Project Root Path ---
# This allows the script to be run from anywhere and still find the project root.
try:
    SCRIPT_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    # Fallback for environments where __file__ is not defined.
    PROJECT_ROOT = os.path.abspath('.')
    if 'src' not in os.listdir(PROJECT_ROOT):
        alt_root = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
        if 'src' in os.listdir(alt_root): PROJECT_ROOT = alt_root
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

# --- Import Project Modules & Dependencies ---
# The config_loader import also triggers the centralized logging setup.
try:
    from src.config_loader import CONFIG
    from src.exceptions import APIKeyError, APITimeoutError, APINotFoundError, APIError, ConfigError
except ImportError as e:
    # Fallback if dependencies or project structure are not found.
    # Defines dummy classes to allow the module to be imported without crashing.
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"Weather Client: Could not import dependencies. Error: {e}", exc_info=True)
    CONFIG = {'apis': {'weatherapi': {}}, 'api_retries': {}, 'api_retry_delay_seconds': {}}
    class APIKeyError(Exception):
        def __init__(self, message, service=None):
            super().__init__(message)
            self.service = service
            self.status_code = 401
    class APITimeoutError(Exception):
        def __init__(self, message, service=None):
            super().__init__(message)
            self.service = service
    class APINotFoundError(Exception):
        def __init__(self, message, service=None):
            super().__init__(message)
            self.service = service
            self.status_code = 404
    class APIError(Exception):
        def __init__(self, message, status_code=None, service=None):
            super().__init__(message)
            self.status_code = status_code
            self.service = service
    class ConfigError(Exception):
        pass
except Exception as e:
    # Broad exception for other potential import errors.
    logging.basicConfig(level=logging.WARNING)
    logging.error(f"Weather Client: Critical error importing dependencies: {e}", exc_info=True)
    # Re-raising might be appropriate here if the app cannot function at all.
    CONFIG = {'apis': {'weatherapi': {}}, 'api_retries': {}, 'api_retry_delay_seconds': {}}
    class APIKeyError(Exception): 
        def __init__(self, message, service=None): super().__init__(message); self.service=service; self.status_code=401
    class APITimeoutError(Exception): 
        def __init__(self, message, service=None): super().__init__(message); self.service=service
    class APINotFoundError(Exception): 
        def __init__(self, message, service=None): super().__init__(message); self.service=service; self.status_code=404
    class APIError(Exception): 
        def __init__(self, message, status_code=None, service=None): super().__init__(message); self.status_code=status_code; self.service=service
    class ConfigError(Exception): 
        pass

log = logging.getLogger(__name__)

# --- Load API Token from .env file ---
try:
    # Looks for .env in the project root directory (e.g., BREATHEEASY/.env).
    dotenv_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(dotenv_path): load_dotenv(dotenv_path=dotenv_path)
except Exception: 
    # Silently pass if there's an issue loading the .env file.
    # The os.getenv call below is the canonical check.
    pass

# --- Module Configuration ---
WEATHERAPI_API_KEY = os.getenv('WEATHERAPI_API_KEY')
WEATHERAPI_CURRENT_URL_CFG = CONFIG.get('apis', {}).get('weatherapi', {}).get('base_url', "http://api.weatherapi.com/v1/current.json")
WEATHERAPI_FORECAST_URL_CFG = CONFIG.get('apis', {}).get('weatherapi', {}).get('forecast_url', "http://api.weatherapi.com/v1/forecast.json")
API_TIMEOUT_CFG = CONFIG.get('api_timeout_seconds', 10)
DEFAULT_RETRIES = CONFIG.get('api_retries', {}).get('default', 2)
DEFAULT_RETRY_DELAY = CONFIG.get('api_retry_delay_seconds', {}).get('default', 1)

def _make_weatherapi_request(url, params, city_name_for_log, max_retries_cfg, retry_delay_cfg, context="request"):
    """
    Internal function to make a request to the WeatherAPI with retry logic.
    Handles various exceptions and retries on 5xx server errors and timeouts.
    """
    last_exception = None
    for attempt in range(max_retries_cfg + 1):
        try:
            log.debug(f"Attempt {attempt + 1}/{max_retries_cfg+1} for {context} ({city_name_for_log}) to URL: {url}")
            response = requests.get(url, params=params, timeout=API_TIMEOUT_CFG)
            response.raise_for_status()
            data = response.json()
            # Handle cases where the API returns a 200 OK but includes an error in the JSON payload.
            if "error" in data:
                err_info = data["error"]; err_msg = err_info.get('message', f'Unknown API error for {context}'); err_code = err_info.get('code')
                log.error(f"WeatherAPI JSON error ({context}) for '{city_name_for_log}' (Code: {err_code}): {err_msg}")
                # Specific error code for 'No location found'. A valid, non-exceptional case.
                if err_code == 1006: 
                    return None
                # Codes related to invalid API keys.
                elif err_code in [1002,1003,1005,2006,2007,2008]: raise APIKeyError(f"Code {err_code}: {err_msg}", service="WeatherAPI")
                else: 
                    base_msg_for_exception = f"HTTP error {status_code} encountered for {context} query on '{city_name_for_log}'."
                    log.error(f"WeatherAPI: {base_msg_for_exception} Original HTTPError detail: {str(http_err)}. Resp: {response_text_snippet}")
                    raise APIError(base_msg_for_exception, status_code=status_code, service="WeatherAPI") from http_err
            return data
        except requests.exceptions.HTTPError as http_err:
            status = http_err.response.status_code if http_err.response else 0
            resp_text = http_err.response.text[:100] if http_err.response and hasattr(http_err.response, 'text') else "N/A"
            error_lines = str(http_err).splitlines()
            detail = error_lines[0] if error_lines else str(http_err)
            core_msg = f"HTTP error {status} for {context} query: '{city_name_for_log}'. Detail: {detail}"

            # Retry on 5xx server-side errors.
            if 500 <= status <= 599 and attempt < max_retries_cfg:
                log.warning(f"{core_msg} (attempt {attempt+1}). Retrying in {retry_delay_cfg}s. Resp: {resp_text}")
                last_exception = APIError(core_msg, status_code=status, service="WeatherAPI")
                time.sleep(retry_delay_cfg); continue
            else:
                 # Handle non-retriable errors (4xx) or final failed retry (5xx).
                log.error(f"Final/Non-retry {core_msg}. Resp: {resp_text}")
                if status == 401: raise APIKeyError(f"Auth (401) for {context} query: '{city_name_for_log}'. Check key.", service="WeatherAPI") from http_err
                elif status == 403: raise APIError(f"Forbidden (403) for {context} query: '{city_name_for_log}'.", status_code=403, service="WeatherAPI") from http_err
                elif status == 400 and "no matching location found" in resp_text.lower(): return None
                elif status == 404: raise APINotFoundError(f"Endpoint not found (404) for URL: {url}", service="WeatherAPI") from http_err
                else: 
                     # Generic case for unhandled 4xx or 5xx errors after retries.
                    clean_core_msg = f"HTTP error {status} for {context} query: '{city_name_for_log}'." 
                    log.error(f"WeatherAPI: Raising APIError with: {clean_core_msg} Original HTTPError: {str(http_err)}. Resp: {resp_text}") 
                    raise APIError(clean_core_msg, status_code=status, service="WeatherAPI") from http_err 
        except requests.exceptions.Timeout as e_timeout:
            core_msg = f"Request timed out for {context}: '{city_name_for_log}'"
            if attempt < max_retries_cfg: log.warning(f"{core_msg} (attempt {attempt+1}). Retrying..."); last_exception = APITimeoutError(core_msg, service="WeatherAPI"); time.sleep(retry_delay_cfg); continue
            else: log.error(f"Final timeout. {core_msg}"); raise APITimeoutError(core_msg, service="WeatherAPI") from e_timeout
        except requests.exceptions.RequestException as e_req:
            core_msg = f"Network error for {context}: {e_req}"
            if attempt < max_retries_cfg: log.warning(f"{core_msg} (attempt {attempt+1}). Retrying..."); last_exception = APIError(core_msg, service="WeatherAPI"); time.sleep(retry_delay_cfg); continue
            else: log.error(f"Final network error. {core_msg}"); raise APIError(core_msg, service="WeatherAPI") from e_req
        except (json.JSONDecodeError, ValueError) as json_e:
            response_text = response.text[:100] if 'response' in locals() and hasattr(response, 'text') else 'N/A'
            msg = f"Invalid data format from WeatherAPI ({context}) for '{city_name_for_log}': {str(json_e)}. Response snippet: {response_text}"
            log.error(msg); raise ValueError(msg) from json_e
        except (APIKeyError, APINotFoundError, APIError) as specific_error: 
             # Re-raise specific, already handled exceptions immediately.
            raise specific_error
        except Exception as e_general:
            core_msg = f"Unexpected error for {context}: {e_general}"
            if attempt < max_retries_cfg: log.error(f"{core_msg} (attempt {attempt+1}). Retrying...", exc_info=True); last_exception = APIError(core_msg, service="WeatherAPI"); time.sleep(retry_delay_cfg); continue
            else: log.error(f"Final unexpected error. {core_msg}", exc_info=True); raise APIError(core_msg, service="WeatherAPI") from e_general
    if last_exception: 
        # If the loop finished due to exhausted retries, raise the last captured exception.
        raise last_exception
    return None

def get_current_weather(city_name):
    """
    Fetches the current weather for a specified city.

    Args:
        city_name (str): The name of the city to query (e.g., "Delhi, India").

    Returns:
        dict | None: A dictionary of processed weather data if successful,
                     or None if the city is not found by the API.
    """
    if not WEATHERAPI_API_KEY: raise APIKeyError("WEATHERAPI_API_KEY missing.", service="WeatherAPI")
    if not WEATHERAPI_CURRENT_URL_CFG: raise ConfigError("WeatherAPI current base URL missing.")
    current_retries = CONFIG.get('api_retries', {}).get('weather_api_current', DEFAULT_RETRIES)
    current_delay = CONFIG.get('api_retry_delay_seconds', {}).get('weather_api_current', DEFAULT_RETRY_DELAY)
    params = {'key': WEATHERAPI_API_KEY, 'q': city_name, 'aqi': 'no'}
    data = _make_weatherapi_request(WEATHERAPI_CURRENT_URL_CFG, params, city_name, current_retries, current_delay, "current weather")
    if data is None: return None
    current = data.get("current", {}); loc = data.get("location", {}); cond = current.get("condition", {}) 
    if not current or not loc: raise APIError(f"Missing 'current'/'location' for {city_name}", service="WeatherAPI")
    return {"temp_c": current.get("temp_c"), "feelslike_c": current.get("feelslike_c"), "humidity": current.get("humidity"), "pressure_mb": current.get("pressure_mb"), "condition_text": cond.get("text"), "condition_icon": cond.get("icon"), "wind_kph": current.get("wind_kph"), "wind_dir": current.get("wind_dir"), "uv_index": current.get("uv"), "city": loc.get("name"), "region": loc.get("region"), "country": loc.get("country"), "last_updated": current.get("last_updated"), "localtime": loc.get("localtime")}

def get_weather_forecast(city_name, days=3):
    """
    Fetches a multi-day weather forecast for a specified city.

    Args:
        city_name (str): The name of the city to query.
        days (int): The number of forecast days to retrieve (1-14).

    Returns:
        list[dict] | None: A list of dictionaries, where each dictionary
                           represents one day's forecast. Returns None if
                           the city is not found or no forecast data is available.
    """
    if not WEATHERAPI_API_KEY: raise APIKeyError("WEATHERAPI_API_KEY missing.", service="WeatherAPI")
    if not WEATHERAPI_FORECAST_URL_CFG: raise ConfigError("WeatherAPI forecast URL missing.")
    forecast_retries = CONFIG.get('api_retries', {}).get('weather_api_forecast', DEFAULT_RETRIES)
    forecast_delay = CONFIG.get('api_retry_delay_seconds', {}).get('weather_api_forecast', DEFAULT_RETRY_DELAY)

     # Ensure 'days' is within the API's accepted range (1-14).
    days = max(1, min(days, 14))
    params = {'key': WEATHERAPI_API_KEY, 'q': city_name, 'days': days, 'aqi': 'no', 'alerts': 'no'}
    data = _make_weatherapi_request(WEATHERAPI_FORECAST_URL_CFG, params, city_name, forecast_retries, forecast_delay, "weather forecast")
    if data is None: return None
    fc_days_data = data.get("forecast", {}).get("forecastday", [])
    if not fc_days_data: log.warning(f"No 'forecastday' data for {city_name}"); return None
    processed_fc = []
    for day_data in fc_days_data:
        day_info = day_data.get("day", {}); cond_info = day_info.get("condition", {}) # condition is under day_info
        processed_fc.append({"date": day_data.get("date"), "avgtemp_c": day_info.get("avgtemp_c"), "avghumidity": day_info.get("avghumidity"), "maxwind_kph": day_info.get("maxwind_kph"), "totalprecip_mm": day_info.get("totalprecip_mm"), "uv": day_info.get("uv"), "condition_text": cond_info.get("text")})
    return processed_fc


# --- Example Usage / Direct Execution ---
if __name__ == "__main__":
    # This block runs only when the script is executed directly.
    # It serves as a quick test and demonstration of the module's functions.
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                            format='%(asctime)s - [%(levelname)s] - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
         log.info("Configured fallback logging for direct script run of weather_client.py.")

    print("\n" + "="*30); print(" Running weather_client.py Tests "); print("="*30 + "\n")
    
    test_cities_current = ["Delhi, India", "London", "Atlantisxyz123", "Paris"] 
    for city in test_cities_current:
        print(f"--- Test Current Weather: Fetching for '{city}' ---")
        try:
            weather_data = get_current_weather(city)
            if weather_data:
                print(f"  SUCCESS: Temp={weather_data.get('temp_c')}C, Cond={weather_data.get('condition_text')}")
            elif weather_data is None and city == "Atlantisxyz123":
                # For this specific test city, we expect to get None back.
                print(f"  SUCCESS (Expected): City '{city}' not found by WeatherAPI, returned None.")
            else:
                print(f"  WARNING/UNEXPECTED: No weather data returned for '{city}'. Response: {weather_data}")
        except Exception as e:
            print(f"  ERROR Caught: {type(e).__name__} - {e}")
        print("-" * 20)

    print("\n" + "="*30); print(" Testing Weather Forecast Function "); print("="*30 + "\n")
    
    test_cities_forecast = ["Delhi, India", "Paris", "Atlantisxyz123"] 
    forecast_days_cfg = CONFIG.get('modeling', {}).get('forecast_days', 3) 
    
    for city_fc in test_cities_forecast:
        print(f"--- Test Forecast: Fetching {forecast_days_cfg}-day forecast for '{city_fc}' ---")
        try:
             forecast_data = get_weather_forecast(city_fc, days=forecast_days_cfg)
             if forecast_data:
                 print(f"  SUCCESS: Received {len(forecast_data)}-day forecast. First day temp: {forecast_data[0].get('avgtemp_c') if forecast_data else 'N/A'}")
             elif forecast_data is None and city_fc == "Atlantisxyz123":
                 print(f"  SUCCESS (Expected): City '{city_fc}' not found by WeatherAPI for forecast, returned None.")
             else:
                 print(f"  WARNING/UNEXPECTED: No forecast data returned for '{city_fc}'. Response: {forecast_data}")
        except Exception as e:
             print(f"  ERROR Caught: {type(e).__name__} - {e}")
        print("-" * 20)

    print("\n" + "="*30); print(" weather_client.py Tests Finished "); print("="*30 + "\n")