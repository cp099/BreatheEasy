# File: tests/health_rules/test_info.py

import pytest # Import the pytest framework
import sys
import os

# --- Add project root to sys.path for imports ---
# This ensures src modules can be imported in tests
try:
    # Assumes tests/health_rules/test_info.py
    TEST_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..', '..'))
except NameError:
    # Fallback if __file__ is not defined
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..')) # Assumes running pytest from root
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import the function to be tested ---
from src.health_rules.info import get_aqi_info, AQI_SCALE # Import function and scale data

# --- Test Cases for get_aqi_info ---

# Use @pytest.mark.parametrize to run the same test function with different inputs/outputs
@pytest.mark.parametrize("aqi_value, expected_level", [
    (0,    "Good"),
    (25,   "Good"),
    (50,   "Good"),
    (51,   "Satisfactory"),
    (100,  "Satisfactory"),
    (101,  "Moderate"),
    (200,  "Moderate"),
    (201,  "Poor"),
    (300,  "Poor"),
    (301,  "Very Poor"),
    (400,  "Very Poor"),
    (401,  "Severe"),
    (500,  "Severe"),
    (550,  "Severe"), # Test value above highest defined range
    (1000, "Severe"), # Test very high value
])
def test_get_aqi_info_levels(aqi_value, expected_level):
    """Tests if get_aqi_info returns the correct AQI level for various values."""
    result = get_aqi_info(aqi_value)
    assert result is not None # Ensure a result dictionary was returned
    assert result['level'] == expected_level # Assert the 'level' field matches

def test_get_aqi_info_structure():
    """Tests if the dictionary returned has the expected keys."""
    result = get_aqi_info(75) # Use a typical valid value
    assert result is not None
    assert isinstance(result, dict)
    # Check if all expected keys are present
    expected_keys = {"range", "level", "color", "implications"}
    assert expected_keys == result.keys() # Checks for exact match of keys

def test_get_aqi_info_invalid_input():
    """Tests invalid inputs like negative numbers, None, and strings."""
    assert get_aqi_info(-10) is None
    assert get_aqi_info(-0.1) is None # Test negative float
    assert get_aqi_info(None) is None
    assert get_aqi_info("abc") is None
    # assert get_aqi_info([10]) is None # Test incorrect type

def test_aqi_scale_consistency():
    """Checks if the ranges in AQI_SCALE are somewhat consistent (basic checks)."""
    last_high = -1
    for category in AQI_SCALE:
        assert isinstance(category, dict)
        assert "range" in category
        range_str = category["range"]
        try:
            if '-' in range_str:
                low, high = map(int, range_str.split('-'))
                assert low == last_high + 1 # Check if ranges are contiguous
                assert low <= high
                last_high = high
            # Add checks for '+' notation if used later
        except ValueError:
            pytest.fail(f"Could not parse range string: {range_str}")
        except AssertionError as e:
             pytest.fail(f"Range consistency error for '{range_str}': {e}")