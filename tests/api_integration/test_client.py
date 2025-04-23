# File: tests/api_integration/test_client.py

import pytest
import requests
import sys
import os
import json
from unittest.mock import MagicMock

# --- Add project root to sys.path for imports ---
try:
    TEST_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import the functions to be tested ---
from src.api_integration.client import get_city_aqi_data, get_current_aqi_for_city
# --- ADD Custom Exception Imports ---
from src.exceptions import APIError, APINotFoundError, APIKeyError, APITimeoutError

# --- Test Data Samples (Simulated API Responses) ---

# Simulate a successful response for Delhi from AQICN API
MOCK_SUCCESS_RESPONSE_DELHI = {
    "status": "ok",
    "data": {
        "aqi": 178,
        "idx": 1437,
        "attributions": [], # Simplified
        "city": {"geo": [28.6139, 77.2090], "name": "Major Dhyan Chand National Stadium, Delhi", "url": "..."},
        "dominentpol": "pm25",
        "iaqi": { # Individual Air Quality Index values
            "co": {"v": 1.2}, "dew": {"v": 5}, "h": {"v": 27.3}, "no2": {"v": 15.8},
            "o3": {"v": 41.2}, "p": {"v": 983.2}, "pm10": {"v": 178},
            "pm25": {"v": 161}, "so2": {"v": 7.6}, "t": {"v": 36.7},
            "w": {"v": 0.5}, "wd": {"v": 329.7}, "wg": {"v": 8.2}
        },
        "time": {"s": "2025-04-21 12:00:00", "tz": "+05:30", "v": 1618996800},
        "forecast": {}, # Simplified
        "debug": {"sync": "..."}
    }
}

# Simulate an "Unknown station" error response
MOCK_ERROR_UNKNOWN_STATION = {
    "status": "error",
    "data": "Unknown station"
}

# Simulate an error response with a different message
MOCK_ERROR_OTHER = {
    "status": "error",
    "data": "Invalid API key or request."
}

# Simulate a response where status is 'ok' but 'data' key is missing
MOCK_OK_NO_DATA = {
    "status": "ok"
    # Missing 'data' field
}

# Simulate a response where 'data' exists but 'aqi' is missing
MOCK_OK_NO_AQI = {
     "status": "ok",
     "data": {
         # Missing 'aqi' field
         "city": {"name": "Test City"},
         "time": {"s": "2024-01-01 10:00:00"},
         "iaqi": {"pm25": {"v": 50}}
     }
}


# --- Tests for get_city_aqi_data ---

def test_get_city_aqi_data_success(mocker):
    """Tests successful API call and data retrieval."""
    # 1. Configure the mock for requests.get
    mock_response = MagicMock()
    mock_response.status_code = 200 # Simulate HTTP 200 OK
    mock_response.json.return_value = MOCK_SUCCESS_RESPONSE_DELHI # Return our sample success data
    # raise_for_status() should do nothing on 200 OK
    mock_response.raise_for_status.return_value = None

    # Patch requests.get: When requests.get is called anywhere, return mock_response instead
    mocker.patch('requests.get', return_value=mock_response)

    # 2. Call the function under test
    result = get_city_aqi_data("Delhi")

    # 3. Assertions
    assert result is not None # Should return data, not None
    assert isinstance(result, dict)
    assert result["status"] == "ok"
    assert "data" in result
    assert result["data"]["aqi"] == 178 # Check specific data points
    assert result["data"]["city"]["name"] == "Major Dhyan Chand National Stadium, Delhi"
    # Check that requests.get was called (optional but good)
    requests.get.assert_called_once()

def test_get_city_aqi_data_unknown_station(mocker):
    """Tests handling of 'Unknown station' error from API."""
    mock_response = MagicMock()
    mock_response.status_code = 200 # API returns 200 OK but JSON contains error status
    mock_response.json.return_value = MOCK_ERROR_UNKNOWN_STATION
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_response)

    result = get_city_aqi_data("Atlantis")

    assert result is None # Function should return None on known API error

def test_get_city_aqi_data_other_api_error(mocker):
    """Tests handling of other 'error' statuses from API."""
    # (Keep mock setup as before)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_ERROR_OTHER
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_response)

    # Expect an APIError to be raised
    with pytest.raises(APIError) as excinfo:
        get_city_aqi_data("SomeCity")

    # Optional: Check exception details
    assert "Invalid API key or request" in str(excinfo.value)
    assert excinfo.value.service == "AQICN"

def test_get_city_aqi_data_http_error_404(mocker):
    """Tests handling of HTTP 404 error."""
    # (Keep mock setup as before)
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mocker.patch('requests.get', return_value=mock_response)

    # Expect APINotFoundError
    with pytest.raises(APINotFoundError) as excinfo:
        get_city_aqi_data("NonExistentPlace")

    # Optional checks
    assert "endpoint or city 'NonExistentPlace' not found" in str(excinfo.value)
    assert excinfo.value.service == "AQICN"

def test_get_city_aqi_data_http_error_401(mocker):
    """Tests handling of HTTP 401 Unauthorized error."""
    # (Keep mock setup as before)
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mocker.patch('requests.get', return_value=mock_response)

    # Expect APIKeyError
    with pytest.raises(APIKeyError) as excinfo:
        get_city_aqi_data("Delhi")

    assert "Authorization failed" in str(excinfo.value)
    assert excinfo.value.service == "AQICN"

def test_get_city_aqi_data_network_error(mocker):
    """Tests handling of network errors like connection errors."""
    # (Keep mock setup as before)
    mocker.patch('requests.get', side_effect=requests.exceptions.ConnectionError("Failed to connect"))

    # Expect APIError (as network errors are wrapped)
    with pytest.raises(APIError) as excinfo:
        get_city_aqi_data("Delhi")

    assert "request error: Failed to connect" in str(excinfo.value)
    assert excinfo.value.service == "AQICN"

def test_get_city_aqi_data_timeout(mocker):
    """Tests handling of request timeout."""
    # (Keep mock setup as before)
    mocker.patch('requests.get', side_effect=requests.exceptions.Timeout("Request timed out"))

    # Expect APITimeoutError
    with pytest.raises(APITimeoutError) as excinfo:
        get_city_aqi_data("Delhi")

    assert "timed out for 'Delhi'" in str(excinfo.value)
    assert excinfo.value.service == "AQICN"

def test_get_city_aqi_data_invalid_json(mocker):
    """Tests handling of invalid JSON response from server."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Configure json() method to raise JSONDecodeError explicitly
    # Requires dummy args for JSONDecodeError: msg, doc, pos
    mock_response.json.side_effect = json.JSONDecodeError("Decoding JSON has failed", "invalid json string", 0)
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_response)

    # Expect ValueError (since client.py catches ValueError which includes JSONDecodeError)
    with pytest.raises(ValueError) as excinfo:
        get_city_aqi_data("Delhi")

    assert "JSON decoding error" in str(excinfo.value)


# --- Tests for get_current_aqi_for_city ---

# We can reuse the mocks by calling get_city_aqi_data inside the test,
# or mock get_city_aqi_data directly if preferred. Let's mock requests.get again.

def test_get_current_aqi_success(mocker):
    """Tests successful extraction of current AQI."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_SUCCESS_RESPONSE_DELHI
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_response) # Mock the underlying call

    result = get_current_aqi_for_city("Delhi")

    assert result is not None
    assert isinstance(result, dict)
    assert result['city'] == 'Delhi'
    assert result['aqi'] == 178
    assert result['station'] == "Major Dhyan Chand National Stadium, Delhi"
    assert result['time'] == "2025-04-21 12:00:00"

def test_get_current_aqi_api_error(mocker):
    """Tests that current AQI func returns None if underlying API call fails."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_ERROR_UNKNOWN_STATION
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_response)

    result = get_current_aqi_for_city("Atlantis")

    assert result is None

def test_get_current_aqi_missing_data_key(mocker):
    """Tests case where API status is ok but 'data' key is missing."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_OK_NO_DATA
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_response)

    result = get_current_aqi_for_city("TestCity")

    assert result is None

def test_get_current_aqi_missing_aqi_value(mocker):
    """Tests case where 'data' exists but 'aqi' value within it is missing."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_OK_NO_AQI
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_response)

    result = get_current_aqi_for_city("TestCity")

    assert result is None # Should return None if 'aqi' cannot be extracted