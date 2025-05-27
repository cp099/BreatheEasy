# File: tests/api_integration/test_weather_client.py (Corrected Assertions)

import pytest
import requests
import sys
import os
import json
from unittest.mock import MagicMock, patch
import time

# --- Path Setup ---
try:
    TEST_DIR = os.path.dirname(__file__); PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')): PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')): PROJECT_ROOT = os.path.abspath(os.getcwd())
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from src.api_integration.weather_client import get_current_weather, get_weather_forecast
from src.exceptions import APIKeyError, APITimeoutError, APINotFoundError, APIError
from src.config_loader import CONFIG

# --- Mock Data (Keep as is) ---
MOCK_SUCCESS_RESPONSE_DELHI_WEATHER = {"location": {"name": "Delhi", "country": "India"}, "current": {"temp_c": 27.2, "condition": {"text": "Mist"}}}
MOCK_ERROR_CITY_NOT_FOUND_JSON = {"error": {"code": 1006, "message": "No location found matching parameter q"}}
MOCK_ERROR_BAD_KEY_JSON = {"error": {"code": 2006, "message": "API key provided is invalid."}}
MOCK_FORECAST_SUCCESS_DELHI = {
    "location": {},
    "current": {},
    "forecast": {
        "forecastday": [
            {"date": "2025-05-13", "day": {"avgtemp_c": 30.0, "avghumidity": 60, "maxwind_kph": 10, "totalprecip_mm": 0, "uv": 7, "condition": {"text": "Sunny"}}},
            {"date": "2025-05-14", "day": {"avgtemp_c": 31.0, "avghumidity": 62, "maxwind_kph": 12, "totalprecip_mm": 0.1, "uv": 8, "condition": {"text": "Partly cloudy"}}},
            {"date": "2025-05-15", "day": {"avgtemp_c": 29.5, "avghumidity": 65, "maxwind_kph": 15, "totalprecip_mm": 0.5, "uv": 6, "condition": {"text": "Patchy rain possible"}}}
        ]
    }
}

# --- Mock Helper (Keep as is) ---
def mock_requests_get_setup(mocker, status_code=200, json_data=None, text_data=None, side_effect=None):
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data if json_data is not None else {}
    mock_resp.text = text_data if text_data is not None else (json.dumps(json_data) if json_data else "")
    if 200 <= status_code < 300 and not (side_effect and isinstance(side_effect, requests.exceptions.HTTPError)): mock_resp.raise_for_status.return_value = None
    elif not (side_effect and isinstance(side_effect, requests.exceptions.HTTPError)):
        http_error = requests.exceptions.HTTPError(response=mock_resp); http_error.response = mock_resp
        mock_resp.raise_for_status.side_effect = http_error
    if side_effect: return mocker.patch('requests.get', side_effect=side_effect)
    else: return mocker.patch('requests.get', return_value=mock_resp)

# --- Tests for get_current_weather ---
@patch('src.api_integration.weather_client.time.sleep', return_value=None)
def test_get_current_weather_success(mock_sleep, mocker):
    mock_requests_get_setup(mocker, json_data=MOCK_SUCCESS_RESPONSE_DELHI_WEATHER)
    result = get_current_weather("Delhi, India")
    assert result["temp_c"] == 27.2

@patch('src.api_integration.weather_client.time.sleep', return_value=None)
def test_get_current_weather_city_not_found_json_error(mock_sleep, mocker): # API returns "error" in JSON
    mock_requests_get_setup(mocker, json_data=MOCK_ERROR_CITY_NOT_FOUND_JSON)
    result = get_current_weather("Atlantisxyz")
    assert result is None # Client handles 1006 by returning None

@patch('src.api_integration.weather_client.time.sleep', return_value=None)
def test_get_current_weather_bad_api_key_json_error(mock_sleep, mocker): # API returns "error" in JSON
    mock_requests_get_setup(mocker, json_data=MOCK_ERROR_BAD_KEY_JSON)
    with pytest.raises(APIKeyError) as excinfo:
        get_current_weather("Delhi, India")
    expected_msg_content = f"Code {MOCK_ERROR_BAD_KEY_JSON['error']['code']}: {MOCK_ERROR_BAD_KEY_JSON['error']['message']}"
    assert str(excinfo.value) == f"WeatherAPI API Error: {expected_msg_content} (Status: 401)"

@patch('src.api_integration.weather_client.time.sleep', return_value=None)
def test_get_current_weather_http_error_401(mock_sleep, mocker): # requests.get raises HTTPError 401
    mock_requests_get_setup(mocker, status_code=401, text_data="Unauthorized")
    with pytest.raises(APIKeyError) as excinfo:
        get_current_weather("Delhi, India")
    expected_msg_content = "Auth (401) for current weather query: 'Delhi, India'. Check key."
    assert str(excinfo.value) == f"WeatherAPI API Error: {expected_msg_content} (Status: 401)"

@patch('src.api_integration.weather_client.time.sleep', return_value=None)
def test_get_current_weather_http_error_400_generic(mock_sleep, mocker):
    mock_resp = MagicMock(status_code=400, reason="Bad Request", text='{"error":{"code":9999,"message":"Some other generic bad request."}}')
    http_error = requests.exceptions.HTTPError(response=mock_resp)
    http_error.response = mock_resp 
    mocker.patch('requests.get', side_effect=http_error)
    
    test_city = "BadQuery"
    context_in_client = "current weather" # Match the context string used in client.py
    with pytest.raises(APIError) as excinfo:
        get_current_weather(test_city)

    # This is the message passed to APIError constructor in the MODIFIED client.py for this case
    base_message_from_client = f"HTTP error 400 for {context_in_client} query: '{test_city}'." # No "Detail:" part
    
    # This is the fully formatted message that APIError (from exceptions.py) will have in its args[0]
    expected_full_exception_message = f"WeatherAPI API Error: {base_message_from_client} (Status: 400)"
    
    print(f"ACTUAL  : {repr(excinfo.value.args[0])}")
    print(f"EXPECTED: {repr(expected_full_exception_message)}")

    assert excinfo.value.args[0] == expected_full_exception_message
    assert excinfo.value.service == "WeatherAPI"
    assert excinfo.value.status_code == 400
    assert isinstance(excinfo.value.__cause__, requests.exceptions.HTTPError)
    
@patch('src.api_integration.weather_client.time.sleep', return_value=None)
def test_get_current_weather_http_error_502_with_retries(mock_sleep, mocker):
    retries = CONFIG.get('api_retries',{}).get('weather_api_current', CONFIG.get('api_retries',{}).get('default',2))
    mock_resp = MagicMock(status_code=502); http_error = requests.exceptions.HTTPError(response=mock_resp); http_error.response = mock_resp
    mock_get = mock_requests_get_setup(mocker, side_effect=[http_error] * (retries + 1))
    with pytest.raises(APIError) as excinfo: get_current_weather("Delhi, India")
    expected_msg_content = f"HTTP error 502 for current weather query: 'Delhi, India'."
    assert str(excinfo.value) == f"WeatherAPI API Error: {expected_msg_content} (Status: 502)"
    assert mock_get.call_count == retries + 1

@patch('src.api_integration.weather_client.time.sleep', return_value=None)
def test_get_current_weather_timeout_with_retries(mock_sleep, mocker):
    retries = CONFIG.get('api_retries',{}).get('weather_api_current', CONFIG.get('api_retries',{}).get('default',2))
    mock_get = mock_requests_get_setup(mocker, side_effect=[requests.exceptions.Timeout("Simulated")] * (retries + 1))
    with pytest.raises(APITimeoutError) as excinfo: get_current_weather("Delhi, India")
    expected_msg = "Request timed out for current weather: 'Delhi, India'"
    assert str(excinfo.value) == f"WeatherAPI API Error: {expected_msg}"
    assert mock_get.call_count == retries + 1

@patch('src.api_integration.weather_client.time.sleep', return_value=None)
def test_get_current_weather_network_error_with_retries(mock_sleep, mocker):
    retries = CONFIG.get('api_retries',{}).get('weather_api_current', CONFIG.get('api_retries',{}).get('default',2))
    err = requests.exceptions.ConnectionError("Simulated Network Down")
    mock_get = mock_requests_get_setup(mocker, side_effect=[err] * (retries + 1))
    with pytest.raises(APIError) as excinfo: get_current_weather("Delhi, India")
    expected_msg = f"Network error for current weather: {err}"
    assert str(excinfo.value) == f"WeatherAPI API Error: {expected_msg}"
    assert mock_get.call_count == retries + 1

@patch('src.api_integration.weather_client.time.sleep', return_value=None)
def test_get_current_weather_invalid_json(mock_sleep, mocker):
    mock_resp = MagicMock(status_code=200, text="Not JSON")
    json_err = json.JSONDecodeError("Expecting value", "Not JSON", 0)
    mock_resp.json.side_effect = json_err
    mock_resp.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_resp)
    test_city = "Delhi, India"
    with pytest.raises(ValueError) as excinfo: get_current_weather(test_city)
    expected_msg = f"Invalid data format from WeatherAPI (current weather) for '{test_city}': {str(json_err)}. Response snippet: Not JSON"
    assert str(excinfo.value) == expected_msg

# --- Tests for get_weather_forecast ---
@patch('src.api_integration.weather_client.time.sleep', return_value=None)
def test_get_weather_forecast_success(mock_sleep, mocker):
    mock_requests_get_setup(mocker, json_data=MOCK_FORECAST_SUCCESS_DELHI)
    result = get_weather_forecast("Delhi, India", days=3)
    assert len(result) == 3 and result[0]['avgtemp_c'] == 30.0

@patch('src.api_integration.weather_client.time.sleep', return_value=None)
def test_get_weather_forecast_city_not_found_json_error(mock_sleep, mocker):
    mock_requests_get_setup(mocker, json_data=MOCK_ERROR_CITY_NOT_FOUND_JSON)
    assert get_weather_forecast("Wakanda", days=3) is None

@patch('src.api_integration.weather_client.time.sleep', return_value=None)
def test_get_weather_forecast_bad_api_key_json_error(mock_sleep, mocker):
    mock_requests_get_setup(mocker, json_data=MOCK_ERROR_BAD_KEY_JSON)
    with pytest.raises(APIKeyError) as excinfo: get_weather_forecast("Gotham", days=3)
    base_msg = f"Code {MOCK_ERROR_BAD_KEY_JSON['error']['code']}: {MOCK_ERROR_BAD_KEY_JSON['error']['message']}"
    assert str(excinfo.value) == f"WeatherAPI API Error: {base_msg} (Status: 401)"

@patch('src.api_integration.weather_client.time.sleep', return_value=None)
def test_get_weather_forecast_http_503_with_retries(mock_sleep, mocker):
    retries = CONFIG.get('api_retries',{}).get('weather_api_forecast', CONFIG.get('api_retries',{}).get('default',2))
    mock_resp = MagicMock(status_code=503); http_err = requests.exceptions.HTTPError(response=mock_resp); http_err.response = mock_resp
    mock_get = mock_requests_get_setup(mocker, side_effect=[http_err] * (retries + 1))
    test_city = "London, UK"
    with pytest.raises(APIError) as excinfo: get_weather_forecast(test_city, days=3)
    base_msg = f"HTTP error 503 for weather forecast query: '{test_city}'."
    assert str(excinfo.value) == f"WeatherAPI API Error: {base_msg} (Status: 503)"
    assert mock_get.call_count == retries + 1