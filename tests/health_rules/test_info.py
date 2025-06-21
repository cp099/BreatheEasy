
# File: tests/health_rules/test_info.py
"""
Unit tests for the AQI scale definitions and utilities in `src/health_rules/info.py`.

This suite verifies that the `get_aqi_info` function correctly classifies
AQI values into the appropriate categories and handles various edge cases
and invalid inputs gracefully. It also includes a sanity check on the
`AQI_SCALE` data structure itself.
"""

import pytest 
import sys
import os

# --- Setup Project Root Path ---
# This allows the script to be run from anywhere and still find the project root.
try:
    TEST_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..', '..'))
except NameError:
    # Fallback for environments where __file__ is not defined.
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..')) 
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import the function and data to be tested ---
from src.health_rules.info import get_aqi_info, AQI_SCALE # Import function and scale data


# --- Test Cases for get_aqi_info ---

@pytest.mark.parametrize("aqi_value, expected_level", [
    # Test cases for the "Good" category, including boundaries
    (0,    "Good"),
    (25,   "Good"),
    (50,   "Good"),
    # Test cases for the "Satisfactory" category
    (51,   "Satisfactory"),
    (100,  "Satisfactory"),
    # Test cases for the "Moderate" category
    (101,  "Moderate"),
    (200,  "Moderate"),
    # Test cases for the "Poor" category
    (201,  "Poor"),
    (300,  "Poor"),
    # Test cases for the "Very Poor" category    
    (301,  "Very Poor"),
    (400,  "Very Poor"),
    # Test cases for the "Severe" category
    (401,  "Severe"),
    (500,  "Severe"),
    # Test cases for values above the highest defined range
    (550,  "Severe"), 
    (1000, "Severe"), 
])
def test_get_aqi_info_levels(aqi_value, expected_level):
    """
    Verifies that `get_aqi_info` returns the correct AQI level for a wide
    range of valid numerical inputs, including boundary values.
    """
    result = get_aqi_info(aqi_value)
    assert result is not None 
    assert result['level'] == expected_level 

def test_get_aqi_info_structure():
    """
    Ensures the dictionary returned by `get_aqi_info` has the expected structure and keys.
    """
    # Use a typical valid value for the structure check.
    result = get_aqi_info(75) 
    assert result is not None
    assert isinstance(result, dict)

    # Verify that the result contains exactly the required keys.
    expected_keys = {"range", "level", "color", "implications"}
    assert expected_keys == result.keys() 

def test_get_aqi_info_invalid_input():
    """
    Checks that `get_aqi_info` handles various invalid inputs by correctly returning None.
    """
    # Test with negative numbers (integer and float)
    assert get_aqi_info(-10) is None, "Should return None for negative integer."
    assert get_aqi_info(-0.1) is None, "Should return None for negative float."
    # Test with None 
    assert get_aqi_info(None) is None, "Should return None for None input."
    # Test with incorrect data types
    assert get_aqi_info("abc") is None, "Should return None for a string."
    assert get_aqi_info([10]) is None, "Should return None for a list."


# --- Sanity Check for AQI_SCALE Data Structure ---


def test_aqi_scale_consistency():
    """
    Performs a basic sanity check on the AQI_SCALE data constant to ensure
    the ranges are contiguous and properly formatted.
    """
    last_high = -1
    for category in AQI_SCALE:
        assert isinstance(category, dict)
        assert "range" in category
        range_str = category["range"]
        try:
            # Assumes format is "low-high".
            if '-' in range_str:
                low, high = map(int, range_str.split('-'))
                # Check that the current range starts one after the previous one ended.
                assert low == last_high + 1, f"Gap or overlap found at range '{range_str}'."
                assert low <= high, f"Low value '{low}' is greater than high value '{high}'."
                last_high = high
            
        except ValueError:
            pytest.fail(f"Could not parse range string: {range_str}")
        except AssertionError as e:
             pytest.fail(f"Range consistency error for '{range_str}': {e}")