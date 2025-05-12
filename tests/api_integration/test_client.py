# File: tests/api_integration/test_client.py

import pytest
import requests
import sys
import os
import json
from unittest.mock import MagicMock, patch # Import patch for cleaner mocking

# --- Add project root to sys.path for imports ---
try:
    TEST_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..', '..'))
except NameError: # Fallback for environments where __file__ might not be defined
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..')) # Assumes tests run from tests/api_integration
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')): # If not, try one level up from cwd
        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')): # If still not, assume cwd is project root
        PROJECT_ROOT = os.path.abspath(os.getcwd())

if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import the functions and exceptions to be tested ---
from src.api_integration.client import (
    get_city_aqi_data, 
    get_current_aqi_for_city,
    # get_current_pollutant_risks_for_city # Add if you have tests for this
)
from src.exceptions import APIError, APINotFoundError, APIKeyError, APITimeoutError, ConfigError

# --- Test Data Samples (Simulated API Responses) ---
# (Keep your MOCK_... dictionaries as they are, they seem fine for simulating responses)
MOCK_SUCCESS_RESPONSE_DELHI = { "status": "ok", "data": { "aqi": 178, "idx": 1437, "city": {"geo": [28.6139, 77.2090], "name": "Major Dhyan Chand National Stadium, Delhi", "url": "..."}, "dominentpol": "pm25", "iaqi": { "co": {"v": 1.2}, "pm25": {"v": 161}}, "time": {"s": "2025-04-21 12:00:00", "tz": "+05:30", "v": 1618996800}}}
MOCK_ERROR_UNKNOWN_STATION = {"status": "error", "data": "Unknown station"}
MOCK_ERROR_OTHER = {"status": "error", "data": "Invalid API key or request."}
MOCK_OK_NO_DATA_FIELD = {"status": "ok"} # Renamed for clarity
MOCK_OK_NO_AQI_VALUE = {"status": "ok", "data": {"city": {"name": "Test City"}, "time": {"s": "2024-01-01 10:00:00"}, "iaqi": {"pm25": {"v": 50}}}} # Renamed
MOCK_OK_AQI_IS_NONE_OR_DASH = {"status": "ok", "data": {"aqi": None, "city": {"name": "Test City Dash AQI"}, "time": {"s": "2024-01-01 10:00:00"}}}


# --- Helper for Mocking requests.get ---
def mock_requests_get(mocker, status_code=200, json_data=None, text_data=None, raise_for_status_effect=None, side_effect=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    if json_data is not None:
        mock_resp.json.return_value = json_data
    if text_data is not None: # For non-JSON responses or error context
        mock_resp.text = text_data
    if raise_for_status_effect:
        mock_resp.raise_for_status.side_effect = raise_for_status_effect
    else:
        mock_resp.raise_for_status.return_value = None # Default: no error
    
    if side_effect: # For simulating network errors etc. directly on requests.get
        return mocker.patch('requests.get', side_effect=side_effect)
    else:
        return mocker.patch('requests.get', return_value=mock_resp)

# --- Tests for get_city_aqi_data ---

def test_get_city_aqi_data_success(mocker):
    mock_requests_get(mocker, json_data=MOCK_SUCCESS_RESPONSE_DELHI)
    result = get_city_aqi_data("Delhi")
    assert result == MOCK_SUCCESS_RESPONSE_DELHI # Expect the full data dict back

def test_get_city_aqi_data_unknown_station(mocker):
    mock_requests_get(mocker, json_data=MOCK_ERROR_UNKNOWN_STATION)
    result = get_city_aqi_data("Atlantis")
    assert result is None # This function returns None for "Unknown station"

def test_get_city_aqi_data_other_api_error_payload(mocker): # Renamed for clarity
    mock_requests_get(mocker, json_data=MOCK_ERROR_OTHER)
    with pytest.raises(APIError) as excinfo:
        get_city_aqi_data("SomeCity")
    assert "AQICN API error: Invalid API key or request." in str(excinfo.value)
    assert excinfo.value.service == "AQICN"

def test_get_city_aqi_data_http_error_404(mocker):
    mock_requests_get(mocker, status_code=404, text_data="Not Found Text", 
                      raise_for_status_effect=requests.exceptions.HTTPError(response=MagicMock(status_code=404, reason="Not Found", text="Not Found Text")))
    with pytest.raises(APINotFoundError) as excinfo:
        get_city_aqi_data("NonExistentPlace")
    assert "AQICN endpoint or city query 'NonExistentPlace' not found (404)." in str(excinfo.value)

def test_get_city_aqi_data_http_error_401(mocker):
    # Your mock setup for requests.get to simulate a 401 response
    mock_resp = MagicMock(status_code=401, reason="Unauthorized", text="Client Error: Unauthorized")
    http_error_instance = requests.exceptions.HTTPError(response=mock_resp)
    # It's important that the mock_response is attached to the error instance, 
    # as client.py might try to access http_err.response.status_code
    http_error_instance.response = mock_resp 
                                     
    mocker.patch('requests.get', 
                 side_effect=http_error_instance) # Make requests.get raise this error
    
    city_query_in_test = "Delhi" # The city name used in the get_city_aqi_data call

    with pytest.raises(APIKeyError) as excinfo:
        get_city_aqi_data(city_query_in_test) # This call will trigger the raise in client.py

    # This is the message passed by client.py to the APIKeyError constructor
    base_message_from_client = f"AQICN Authorization failed (401) for query '{city_query_in_test}'. Check token."
    
    # This is the fully formatted message that APIKeyError (via APIError) will have in its args[0]
    expected_full_exception_message = f"AQICN API Error: {base_message_from_client} (Status: 401)"
    
    assert excinfo.value.args[0] == expected_full_exception_message
    
    # Also assert other attributes of your custom exception
    assert excinfo.value.service == "AQICN"
    assert excinfo.value.status_code == 401

def test_get_city_aqi_data_network_error(mocker):
    mock_requests_get(mocker, side_effect=requests.exceptions.ConnectionError("Simulated Failed to connect"))
    with pytest.raises(APIError) as excinfo: # Wrapped as general APIError
        get_city_aqi_data("Delhi")
    assert "AQICN request error for 'Delhi': Simulated Failed to connect" in str(excinfo.value)

def test_get_city_aqi_data_timeout(mocker):
    mock_requests_get(mocker, side_effect=requests.exceptions.Timeout("Simulated Request timed out"))
    with pytest.raises(APITimeoutError) as excinfo:
        get_city_aqi_data("Delhi")
    assert "Request to AQICN API timed out for 'Delhi'" in str(excinfo.value) # Message includes city

def test_get_city_aqi_data_invalid_json(mocker):
    mock_resp = MagicMock(status_code=200, text="Invalid JSON String")
    mock_resp.json.side_effect = json.JSONDecodeError("Simulated Decoding JSON has failed", "invalid json string", 0)
    mock_resp.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_resp)
    with pytest.raises(ValueError) as excinfo: # Wrapped as ValueError
        get_city_aqi_data("Delhi")
    assert "AQICN JSON decoding error for 'Delhi'" in str(excinfo.value)
    assert "Simulated Decoding JSON has failed" in str(excinfo.value)


# --- Tests for get_current_aqi_for_city ---
# For these tests, we mock get_city_aqi_data itself, as its own unit tests cover its behavior.

@patch('src.api_integration.client.get_city_aqi_data') # Path to the function to mock
def test_get_current_aqi_success_wrapper(mock_get_city_aqi_data):
    mock_get_city_aqi_data.return_value = MOCK_SUCCESS_RESPONSE_DELHI
    
    # The wrapper expects "City, Country" format from Dash app
    result = get_current_aqi_for_city("Delhi, India") 
    
    assert isinstance(result, dict)
    assert result.get('error') is None # No error key on success
    assert result['city'] == 'Delhi' # Extracted simple city name
    assert result['aqi'] == 178
    assert result['station'] == "Major Dhyan Chand National Stadium, Delhi"
    assert result['time'] == "2025-04-21 12:00:00"
    mock_get_city_aqi_data.assert_called_once_with("Delhi") # Check it called underlying with simple name

@patch('src.api_integration.client.get_city_aqi_data')
def test_get_current_aqi_unknown_station_wrapper(mock_get_city_aqi_data):
    mock_get_city_aqi_data.return_value = None # Simulate "Unknown station" from underlying
    
    result = get_current_aqi_for_city("Atlantis, Nowhere")
    
    assert isinstance(result, dict)
    assert result.get('aqi') is None
    assert 'error' in result
    assert result['error'] == "Station not found by AQICN."
    assert result['station'] == "Unknown station"
    mock_get_city_aqi_data.assert_called_once_with("Atlantis")

@patch('src.api_integration.client.get_city_aqi_data')
def test_get_current_aqi_api_error_wrapper(mock_get_city_aqi_data):
    # Simulate an APIError being raised by get_city_aqi_data
    mock_get_city_aqi_data.side_effect = APIError("Simulated underlying API failure", service="AQICN_Test")
    
    result = get_current_aqi_for_city("ErrorCity, Test")
    
    assert isinstance(result, dict)
    assert result.get('aqi') is None
    assert 'error' in result
    assert "API Error: Simulated underlying API failure" in result['error'] # The str(e) part
    mock_get_city_aqi_data.assert_called_once_with("ErrorCity")

@patch('src.api_integration.client.get_city_aqi_data')
def test_get_current_aqi_missing_data_field_wrapper(mock_get_city_aqi_data): # Test for missing "data" field in API response
    mock_get_city_aqi_data.return_value = MOCK_OK_NO_DATA_FIELD 
    
    result = get_current_aqi_for_city("NoDataCity, Test")
    
    assert isinstance(result, dict)
    assert result.get('aqi') is None
    assert 'error' in result
    # The current client.py logic for missing 'aqi' key (when 'data' itself is missing)
    # might fall into a generic error or a specific check. Let's assume it should indicate missing AQI.
    # Based on client.py: `aqi_raw = api_data.get("aqi")` where `api_data` is `{}`. `aqi_raw` is None.
    assert result['error'] == "AQI value not reported by station."
    mock_get_city_aqi_data.assert_called_once_with("NoDataCity")

@patch('src.api_integration.client.get_city_aqi_data')
def test_get_current_aqi_missing_aqi_value_wrapper(mock_get_city_aqi_data): # 'data' exists, 'aqi' key missing
    mock_get_city_aqi_data.return_value = MOCK_OK_NO_AQI_VALUE
    
    result = get_current_aqi_for_city("MissingAQICity, Test")
    
    assert isinstance(result, dict)
    assert result.get('aqi') is None
    assert 'error' in result
    assert result['error'] == "AQI value not reported by station."
    mock_get_city_aqi_data.assert_called_once_with("MissingAQICity")

@patch('src.api_integration.client.get_city_aqi_data')
def test_get_current_aqi_aqi_is_none_or_dash_wrapper(mock_get_city_aqi_data):
    mock_get_city_aqi_data.return_value = MOCK_OK_AQI_IS_NONE_OR_DASH # AQI field is explicitly None or '-'
    
    result = get_current_aqi_for_city("DashAQICity, Test")
    
    assert isinstance(result, dict)
    assert result.get('aqi') is None
    assert 'error' in result
    assert result['error'] == "AQI value not reported by station."
    mock_get_city_aqi_data.assert_called_once_with("DashAQICity")