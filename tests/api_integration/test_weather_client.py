# File: tests/api_integration/test_weather_client.py

import pytest
import requests # For exception types
import sys
import os
from unittest.mock import MagicMock

# --- Add project root to sys.path for imports ---
try:
    TEST_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import the function to be tested ---
from src.api_integration.weather_client import get_current_weather

# --- Test Data Samples (Simulated WeatherAPI.com Responses) ---

# Simulate a successful response for Delhi, India
MOCK_SUCCESS_RESPONSE_DELHI_WEATHER = {
    "location": {
        "name": "Delhi",
        "region": "Delhi",
        "country": "India",
        "lat": 28.67,
        "lon": 77.22,
        "tz_id": "Asia/Kolkata",
        "localtime_epoch": 1713798600, # Example timestamp
        "localtime": "2025-04-22 23:10"
    },
    "current": {
        "last_updated_epoch": 1713798000,
        "last_updated": "2025-04-22 23:00",
        "temp_c": 27.2,
        "temp_f": 81.0,
        "is_day": 0, # 0 = Night, 1 = Day
        "condition": {
            "text": "Mist",
            "icon": "//cdn.weatherapi.com/weather/64x64/night/143.png",
            "code": 1030
        },
        "wind_mph": 8.7,
        "wind_kph": 14.0,
        "wind_degree": 290,
        "wind_dir": "WNW",
        "pressure_mb": 1009.0,
        "pressure_in": 29.8,
        "precip_mm": 0.0,
        "precip_in": 0.0,
        "humidity": 17,
        "cloud": 0,
        "feelslike_c": 25.4,
        "feelslike_f": 77.7,
        "vis_km": 3.5,
        "vis_miles": 2.0,
        "uv": 1.0 # UV Index might be low at night
    }
}

# Simulate an error response (e.g., city not found - code 1006)
MOCK_ERROR_CITY_NOT_FOUND = {
    "error": {
        "code": 1006,
        "message": "No location found matching parameter 'q'"
    }
}

# Simulate an error response (e.g., invalid API key - code 2006/1002?)
MOCK_ERROR_BAD_KEY = {
     "error": {
         "code": 2006, # Example code for invalid key
         "message": "API key provided is invalid."
     }
}


# --- Tests for get_current_weather ---

def test_get_current_weather_success(mocker):
    """Tests successful API call and data parsing for WeatherAPI."""
    # 1. Configure Mock Response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_SUCCESS_RESPONSE_DELHI_WEATHER
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_response)

    # 2. Call Function
    result = get_current_weather("Delhi, India") # Use specific query

    # 3. Assertions
    assert result is not None
    assert isinstance(result, dict)
    assert result["city"] == "Delhi"
    assert result["country"] == "India"
    assert result["temp_c"] == 27.2
    assert result["condition_text"] == "Mist"
    assert result["humidity"] == 17
    assert "condition_icon" in result # Check key presence
    requests.get.assert_called_once()

def test_get_current_weather_city_not_found(mocker):
    """Tests handling of 'city not found' error from WeatherAPI."""
    mock_response = MagicMock()
    mock_response.status_code = 200 # API returns 200 OK but JSON contains error
    mock_response.json.return_value = MOCK_ERROR_CITY_NOT_FOUND
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_response)

    result = get_current_weather("Atlantisxyz")

    assert result is None # Should return None when JSON indicates error

def test_get_current_weather_bad_api_key(mocker):
    """Tests handling of bad API key error from WeatherAPI JSON."""
    mock_response = MagicMock()
    mock_response.status_code = 200 # API might return 200 but with error in JSON
    mock_response.json.return_value = MOCK_ERROR_BAD_KEY
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_response)

    result = get_current_weather("Delhi")

    assert result is None # Should return None

def test_get_current_weather_http_error_401(mocker):
    """Tests handling of HTTP 401 error (e.g., key truly invalid/missing)."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mocker.patch('requests.get', return_value=mock_response)

    result = get_current_weather("Delhi")

    assert result is None # Should return None on 401

def test_get_current_weather_http_error_400(mocker):
    """Tests handling of HTTP 400 error (sometimes used for bad query)."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
    mocker.patch('requests.get', return_value=mock_response)

    # Simulate the scenario seen in previous test run
    result = get_current_weather("Atlantisxyz")

    assert result is None # Should return None on 400

def test_get_current_weather_network_error(mocker):
    """Tests handling of network connection errors."""
    mocker.patch('requests.get', side_effect=requests.exceptions.ConnectionError("Network unavailable"))

    result = get_current_weather("Delhi")

    assert result is None # Should return None

def test_get_current_weather_timeout(mocker):
    """Tests handling of request timeout."""
    mocker.patch('requests.get', side_effect=requests.exceptions.Timeout("Request timed out"))

    result = get_current_weather("Delhi")

    assert result is None # Should return None

def test_get_current_weather_invalid_json(mocker):
    """Tests handling of invalid JSON response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("Invalid JSON received")
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.get', return_value=mock_response)

    result = get_current_weather("Delhi")

    assert result is None # Should return None if JSON parsing fails