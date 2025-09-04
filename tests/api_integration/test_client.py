
# File: tests/api_integration/test_client.py
"""
Unit and integration tests for the AQICN API client (`src/api_integration/client.py`).

This test suite covers two main areas:
1.  Low-level tests for `get_city_aqi_data`, mocking `requests.get` to simulate
    various API responses (success, errors, network failures).
2.  Higher-level tests for wrapper functions like `get_current_aqi_for_city`,
    mocking `get_city_aqi_data` itself to isolate the wrapper's logic.
"""

import pytest
import requests
import sys
import os
import json
from unittest.mock import MagicMock, patch 

# --- Setup Project Root Path ---
# This allows the script to be run from anywhere and still find the project root.
try:
    TEST_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..', '..'))
except NameError: 
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..')) 
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')): 
        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src')): 
        PROJECT_ROOT = os.path.abspath(os.getcwd())

if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import Functions and Exceptions to be Tested ---
from src.api_integration.client import (
    get_city_aqi_data, 
    get_current_aqi_for_city,
    get_current_pollutant_risks_for_city
)
from src.exceptions import APIError, APINotFoundError, APIKeyError, APITimeoutError, ConfigError

# --- Mock Data Samples (Simulated API Responses) ---
MOCK_SUCCESS_RESPONSE_DELHI = { "status": "ok", "data": { "aqi": 178, "idx": 1437, "city": {"geo": [28.6139, 77.2090], "name": "Major Dhyan Chand National Stadium, Delhi", "url": "..."}, "dominentpol": "pm25", "iaqi": { "co": {"v": 1.2}, "pm25": {"v": 161}}, "time": {"s": "2025-04-21 12:00:00", "tz": "+05:30", "v": 1618996800}}}
MOCK_ERROR_UNKNOWN_STATION = {"status": "error", "data": "Unknown station"}
MOCK_ERROR_OTHER = {"status": "error", "data": "Invalid API key or request."}
MOCK_OK_NO_DATA_FIELD = {"status": "ok"} 
MOCK_OK_NO_AQI_VALUE = {"status": "ok", "data": {"city": {"name": "Test City"}, "time": {"s": "2024-01-01 10:00:00"}, "iaqi": {"pm25": {"v": 50}}}} 
MOCK_OK_AQI_IS_NONE_OR_DASH = {"status": "ok", "data": {"aqi": None, "city": {"name": "Test City Dash AQI"}, "time": {"s": "2024-01-01 10:00:00"}}}


def mock_requests_get(mocker, status_code=200, json_data=None, text_data=None, raise_for_status_effect=None, side_effect=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    if json_data is not None:
        mock_resp.json.return_value = json_data
    if text_data is not None: 
        mock_resp.text = text_data
    if raise_for_status_effect:
        mock_resp.raise_for_status.side_effect = raise_for_status_effect
    else:
        mock_resp.raise_for_status.return_value = None 
    
    if side_effect: 
        return mocker.patch('requests.get', side_effect=side_effect)
    else:
        return mocker.patch('requests.get', return_value=mock_resp)

# --- Tests for get_city_aqi_data (Low-Level API Fetcher) ---

def test_get_city_aqi_data_success(mocker):
    """Test a successful API call returns the full JSON data."""
    mock_requests_get(mocker, json_data=MOCK_SUCCESS_RESPONSE_DELHI)
    result = get_city_aqi_data("Delhi")
    assert result == MOCK_SUCCESS_RESPONSE_DELHI 

def test_get_city_aqi_data_unknown_station(mocker):
    """Test that the function correctly handles the 'Unknown station' error by returning None."""
    mock_requests_get(mocker, json_data=MOCK_ERROR_UNKNOWN_STATION)
    result = get_city_aqi_data("Atlantis")
    assert result is None 

def test_get_city_aqi_data_other_api_error_payload(mocker):
    """Test that other 'error' statuses in the JSON payload raise a generic APIError."""
    mock_requests_get(mocker, json_data=MOCK_ERROR_OTHER)
    with pytest.raises(APIError) as excinfo:
        get_city_aqi_data("SomeCity")
    assert "AQICN API error: Invalid API key or request." in str(excinfo.value)
    assert excinfo.value.service == "AQICN"

def test_get_city_aqi_data_http_error_404(mocker):
    """Test that a 404 HTTP error is caught and raised as an APINotFoundError."""
    mock_requests_get(mocker, status_code=404, text_data="Not Found Text", 
                      raise_for_status_effect=requests.exceptions.HTTPError(response=MagicMock(status_code=404, reason="Not Found", text="Not Found Text")))
    with pytest.raises(APINotFoundError) as excinfo:
        get_city_aqi_data("NonExistentPlace")
    assert "AQICN endpoint or city query 'NonExistentPlace' not found (404)." in str(excinfo.value)

def test_get_city_aqi_data_http_error_401(mocker):
    """Test that a 401 HTTP error is caught and raised as an APIKeyError."""
    mock_resp = MagicMock(status_code=401, reason="Unauthorized", text="Client Error: Unauthorized")
    http_error_instance = requests.exceptions.HTTPError(response=mock_resp)
    http_error_instance.response = mock_resp 
                                     
    mocker.patch('requests.get', 
                 side_effect=http_error_instance)
    
    city_query_in_test = "Delhi" 

    with pytest.raises(APIKeyError) as excinfo:
        get_city_aqi_data(city_query_in_test) 

    
    base_message_from_client = f"AQICN Authorization failed (401) for query '{city_query_in_test}'. Check token."
    
    
    expected_full_exception_message = f"AQICN API Error: {base_message_from_client} (Status: 401)"
    
    # Assert that the correct custom exception was raised with the expected message.
    assert excinfo.value.args[0] == expected_full_exception_message
    assert excinfo.value.service == "AQICN"
    assert excinfo.value.status_code == 401

def test_get_city_aqi_data_network_error(mocker):
    """Test that a network connection error is wrapped in a generic APIError."""
    mock_requests_get(mocker, side_effect=requests.exceptions.ConnectionError("Simulated Failed to connect"))
    with pytest.raises(APIError) as excinfo: 
        get_city_aqi_data("Delhi")
    assert "AQICN request error for 'Delhi': Simulated Failed to connect" in str(excinfo.value)

def test_get_city_aqi_data_timeout(mocker):
    """Test that a request timeout is caught and raised as a specific APITimeoutError."""
    mock_requests_get(mocker, side_effect=requests.exceptions.Timeout("Simulated Request timed out"))
    with pytest.raises(APITimeoutError) as excinfo:
        get_city_aqi_data("Delhi")
    assert "Request to AQICN API timed out for 'Delhi'" in str(excinfo.value) 

def test_get_city_aqi_data_invalid_json(mocker):
    """Test that an invalid JSON response raises a ValueError."""
    mock_resp = MagicMock(status_code=200, text="Invalid JSON String")
    mock_resp.json.side_effect = json.JSONDecodeError("Simulated Decoding JSON has failed", "invalid json string", 0)
    mock_resp.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_resp)
    with pytest.raises(ValueError) as excinfo: 
        get_city_aqi_data("Delhi")
    assert "AQICN JSON decoding error for 'Delhi'" in str(excinfo.value)
    assert "Simulated Decoding JSON has failed" in str(excinfo.value)

# --- Tests for get_current_aqi_for_city (High-Level Wrapper) ---

@patch('src.api_integration.client.get_city_aqi_data') 
def test_get_current_aqi_success_wrapper(mock_get_city_aqi_data):
    """Test the wrapper correctly parses a successful response from the underlying function."""
    mock_get_city_aqi_data.return_value = MOCK_SUCCESS_RESPONSE_DELHI
    result = get_current_aqi_for_city("Delhi, India") 
    
    # The wrapper should extract and format the key data points.
    assert isinstance(result, dict)
    assert result.get('error') is None 
    assert result['city'] == 'Delhi' 
    assert result['aqi'] == 178
    assert result['station'] == "Major Dhyan Chand National Stadium, Delhi"
    assert result['time'] == "2025-04-21 12:00:00"
    mock_get_city_aqi_data.assert_called_once_with("Delhi") # Check it called with the simple city name.

@patch('src.api_integration.client.get_city_aqi_data')
def test_get_current_aqi_unknown_station_wrapper(mock_get_city_aqi_data):
    """Test the wrapper correctly handles the 'Unknown station' case (None from underlying)."""
    mock_get_city_aqi_data.return_value = None 
    
    result = get_current_aqi_for_city("Atlantis, Nowhere")
    
    assert isinstance(result, dict)
    assert result.get('aqi') is None
    assert 'error' in result
    assert result['error'] == "Station not found by AQICN."
    assert result['station'] == "Unknown station"
    mock_get_city_aqi_data.assert_called_once_with("Atlantis")

@patch('src.api_integration.client.get_city_aqi_data')
def test_get_current_aqi_api_error_wrapper(mock_get_city_aqi_data):
    """Test the wrapper catches exceptions from the underlying function and returns an error dict."""
    mock_get_city_aqi_data.side_effect = APIError("Simulated underlying API failure", service="AQICN_Test")
    
    result = get_current_aqi_for_city("ErrorCity, Test")
    
    assert isinstance(result, dict)
    assert result.get('aqi') is None
    assert 'error' in result
    assert "API Error: Simulated underlying API failure" in result['error'] 
    mock_get_city_aqi_data.assert_called_once_with("ErrorCity")

@patch('src.api_integration.client.get_city_aqi_data')
def test_get_current_aqi_missing_data_field_wrapper(mock_get_city_aqi_data):
    """Test the wrapper handles a valid 'ok' response that is missing the 'data' field."""
    mock_get_city_aqi_data.return_value = MOCK_OK_NO_DATA_FIELD 
    
    result = get_current_aqi_for_city("NoDataCity, Test")
    
    assert isinstance(result, dict)
    assert result.get('aqi') is None
    assert 'error' in result
    assert result['error'] == "AQI value not reported by station."
    mock_get_city_aqi_data.assert_called_once_with("NoDataCity")

@patch('src.api_integration.client.get_city_aqi_data')
def test_get_current_aqi_missing_aqi_value_wrapper(mock_get_city_aqi_data): 
    """Test the wrapper handles a response where the 'aqi' key itself is missing."""
    mock_get_city_aqi_data.return_value = MOCK_OK_NO_AQI_VALUE
    
    result = get_current_aqi_for_city("MissingAQICity, Test")
    
    assert isinstance(result, dict)
    assert result.get('aqi') is None
    assert 'error' in result
    assert result['error'] == "AQI value not reported by station."
    mock_get_city_aqi_data.assert_called_once_with("MissingAQICity")

@patch('src.api_integration.client.get_city_aqi_data')
def test_get_current_aqi_aqi_is_none_or_dash_wrapper(mock_get_city_aqi_data):
    """Test the wrapper handles a response where the 'aqi' value is explicitly None or a dash."""
    mock_get_city_aqi_data.return_value = MOCK_OK_AQI_IS_NONE_OR_DASH
    
    result = get_current_aqi_for_city("DashAQICity, Test")
    
    assert isinstance(result, dict)
    assert result.get('aqi') is None
    assert 'error' in result
    assert result['error'] == "AQI value not reported by station."
    mock_get_city_aqi_data.assert_called_once_with("DashAQICity")

# --- Tests for get_current_pollutant_risks_for_city (High-Level Wrapper) ---

@patch('src.api_integration.client.get_city_aqi_data')
def test_get_pollutant_risks_success_wrapper(mock_get_city_aqi_data, mocker):
    """
    Tests that the pollutant risks wrapper correctly parses a successful response
    and calls the interpreter function.
    """
    # We also need to mock the interpreter function to isolate this test.
    mocker.patch(
        'src.api_integration.client.interpret_pollutant_risks', 
        return_value=["Mocked PM2.5 Risk"]
    )
    mock_get_city_aqi_data.return_value = MOCK_SUCCESS_RESPONSE_DELHI
    
    result = get_current_pollutant_risks_for_city("Delhi, India")
    
    assert isinstance(result, dict)
    assert 'error' not in result
    assert result['city'] == 'Delhi'
    assert result['pollutants'] == MOCK_SUCCESS_RESPONSE_DELHI['data']['iaqi']
    assert result['risks'] == ["Mocked PM2.5 Risk"]
    mock_get_city_aqi_data.assert_called_once_with("Delhi")

@patch('src.api_integration.client.get_city_aqi_data')
def test_get_pollutant_risks_unknown_station_wrapper(mock_get_city_aqi_data):
    """
    Tests that the wrapper returns a standard error dict for an unknown station.
    """
    mock_get_city_aqi_data.return_value = None # This simulates "Unknown station"
    
    result = get_current_pollutant_risks_for_city("Atlantis, Nowhere")
    
    assert isinstance(result, dict)
    assert 'error' in result
    assert result['error'] == "Station not found by AQICN."

@patch('src.api_integration.client.get_city_aqi_data')
def test_get_pollutant_risks_missing_iaqi_data_wrapper(mock_get_city_aqi_data):
    """
    Tests that the wrapper handles a successful response that is missing the
    'iaqi' (pollutants) data field.
    """
    # Create a mock response without the 'iaqi' key
    mock_response = {"status": "ok", "data": {"aqi": 100, "time": {"s": "..."}}}
    mock_get_city_aqi_data.return_value = mock_response
    
    result = get_current_pollutant_risks_for_city("NoIaqiCity, Test")
    
    assert isinstance(result, dict)
    assert 'error' in result
    assert result['error'] == 'Pollutant data or timestamp missing.'