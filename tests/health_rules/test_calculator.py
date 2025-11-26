# File: tests/health_rules/test_calculator.py

"""
Unit tests for the CPCB AQI calculation logic in `src/health_rules/calculator.py`.
"""
import pytest
import sys
import os
import pandas as pd

# --- Setup Project Root Path ---
try:
    TEST_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

from src.health_rules.calculator import calculate_sub_index, calculate_aqi_from_pollutants

# --- Tests for calculate_sub_index ---

@pytest.mark.parametrize("value, pollutant, expected_sub_index", [
    # Test cases for PM2.5
    (15, 'pm25', 25),      # Good
    (45, 'pm25', 75),      # Satisfactory
    (75, 'pm25', 149),     # Moderate (mid-range)
    (90, 'pm25', 200),     # Moderate (boundary)
    (105, 'pm25', 249),    # Poor (mid-range)
    (250, 'pm25', 400),    # Very Poor (boundary)
    (300, 'pm25', 401),    # Severe (value > 251)
    
    # Test cases for PM10
    (75, 'pm10', 75),      # Satisfactory
    (175, 'pm10', 150),    # Moderate (mid-range)
    
    # Test cases for CO (has float breakpoints)
    (1.5, 'co', 73),       # Satisfactory
    (13.5, 'co', 250),     # Poor (mid-range)

    # Test cases for invalid inputs
    (None, 'pm25', None),     # None value
    (100, 'xyz', None),      # Invalid pollutant name
    (-10, 'pm10', None),     # Negative value (should not find a range)
])
def test_calculate_sub_index(value, pollutant, expected_sub_index):
    """
    Tests the linear interpolation for various pollutants and values.
    """
    assert calculate_sub_index(value, pollutant) == expected_sub_index

# --- Tests for calculate_aqi_from_pollutants ---

def test_calculate_aqi_from_pollutants_pm25_dominant():
    """Tests that the final AQI is the max of the sub-indices (PM2.5 is highest)."""
    # PM2.5 = 75 -> Sub-index 150 (Moderate)
    # CO = 1.5 -> Sub-index 75 (Satisfactory)
    # The final AQI should be 150.
    data_row = pd.Series({'PM2.5': 75, 'CO': 1.5, 'O3': 30})
    assert calculate_aqi_from_pollutants(data_row) == 149

def test_calculate_aqi_from_pollutants_co_dominant():
    """Tests that the final AQI is the max of the sub-indices (CO is highest)."""
    # PM2.5 = 45 -> Sub-index 75 (Satisfactory)
    # CO = 13.5 -> Sub-index 250 (Poor)
    # The final AQI should be 250.
    data_row = pd.Series({'PM2.5': 45, 'CO': 13.5})
    assert calculate_aqi_from_pollutants(data_row) == 250

def test_calculate_aqi_from_pollutants_with_missing_values():
    """Tests that missing pollutant values are ignored in the calculation."""
    data_row = pd.Series({'PM2.5': None, 'CO': 1.5})
    assert calculate_aqi_from_pollutants(data_row) == 73

def test_calculate_aqi_from_pollutants_no_valid_pollutants():
    """Tests that None is returned if no known pollutants are in the data row."""
    data_row = pd.Series({'XYZ': 100, 'ABC': 200})
    assert calculate_aqi_from_pollutants(data_row) is None