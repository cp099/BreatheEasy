# File: tests/modeling/test_predictor.py

"""
Integration tests for the prediction script `src/modeling/predictor.py`.
"""
import pytest
import pandas as pd
import os
import sys
import requests
from unittest.mock import MagicMock

# --- Setup Project Root Path ---
try:
    TEST_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..', '..'))
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

from src.modeling.predictor import get_daily_summary_forecast
from src.exceptions import ModelFileNotFoundError, PredictionError

# --- Mock Data and Fixtures ---

@pytest.fixture
def mock_dependencies(mocker):
    """
    This fixture uses pytest-mock to "hijack" all the external dependencies
    that our predictor function relies on.
    """
    # 1. Mock the LightGBM model object
    mock_model = MagicMock()
    mock_model.predict.return_value = [150.0]
    mock_model.feature_name_ = [
        'temperature_2m_mean', 'temperature_2m_min', 'temperature_2m_max',
        'relative_humidity_2m_mean', 'precipitation_sum', 'wind_speed_10m_mean',
        'day_of_week', 'month', 'year', 'AQI_lag_1_day', 'AQI_lag_7_day'
    ]
    mocker.patch('src.modeling.predictor.load_lgbm_model', return_value=mock_model)

    # 2. Mock the historical data file read
    history_data = {
        'Date': pd.to_datetime(pd.date_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=7)),
        'City': ['TestCity'] * 7,
        'AQI': [100, 105, 110, 115, 120, 125, 130],
        'latitude': [10.0] * 7,
        'longitude': [10.0] * 7
    }
    mock_df = pd.DataFrame(history_data)
    mocker.patch('src.modeling.predictor.pd.read_csv', return_value=mock_df)

    # 3. Mock the live AQI API call
    mocker.patch('src.modeling.predictor.get_current_aqi_for_city', return_value={'aqi': 135})

    # 4. Mock the weather forecast API call (requests.get)
    future_dates = pd.to_datetime(pd.date_range(start=pd.Timestamp.now().normalize(), periods=4))
    weather_data = {
        'time': [d.strftime('%Y-%m-%d') for d in future_dates],
        'temperature_2m_mean': [25.0] * 4,
        'temperature_2m_min': [20.0] * 4,
        'temperature_2m_max': [30.0] * 4,
        'relative_humidity_2m_mean': [70.0] * 4,
        'precipitation_sum': [0.0] * 4,
        'wind_speed_10m_mean': [10.0] * 4
    }
    mock_response = MagicMock(spec=requests.Response)
    mock_response.json.return_value = {'daily': weather_data}
    mock_response.raise_for_status.return_value = None
    mocker.patch('src.modeling.predictor.requests.get', return_value=mock_response)


# --- Test Cases ---

def test_get_daily_summary_forecast_success(mock_dependencies):
    """
    Tests the full, successful execution of the predictor function.
    """
    city = "TestCity"
    forecast = get_daily_summary_forecast(city, days_ahead=3)

    # 1. Assert the output structure
    assert isinstance(forecast, list)
    assert len(forecast) == 3
    assert "predicted_aqi" in forecast[0]
    assert "level" in forecast[0]
    
    # 2. Assert the "Anchor and Trend" logic
    assert forecast[0]['predicted_aqi'] == 135
    assert forecast[1]['predicted_aqi'] == 135
    assert forecast[2]['predicted_aqi'] == 135

def test_get_daily_summary_forecast_model_not_found(mocker):
    """
    Tests that the function returns an empty list if the model file is not found.
    """
    mocker.patch('src.modeling.predictor.load_lgbm_model', side_effect=ModelFileNotFoundError)
    
    forecast = get_daily_summary_forecast("UnknownCity")
    assert forecast == []

def test_get_daily_summary_forecast_live_aqi_fails(mocker):
    """
    Tests that the function returns an empty list if the live AQI call fails.
    """

    mocker.patch('src.modeling.predictor.get_current_aqi_for_city', return_value={'error': 'API down'})
    mocker.patch('src.modeling.predictor.load_lgbm_model')
    mocker.patch('src.modeling.predictor.pd.read_csv')

    forecast = get_daily_summary_forecast("AnyCity")
    assert forecast == []