# File: tests/health_rules/test_interpreter.py

"""
Unit tests for the pollutant risk interpretation logic in `src/health_rules/interpreter.py`.
"""
import pytest
import sys
import os

# --- Setup Project Root Path ---
try:
    TEST_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

from src.health_rules.interpreter import interpret_pollutant_risks

# --- Basic Functionality Tests ---

def test_interpret_risks_clean_air():
    """Tests that no risks are triggered when all pollutant values are below thresholds."""
    iaqi_data = {'pm25': {'v': 10.5}, 'pm10': {'v': 45}, 'o3': {'v': 30}}
    assert interpret_pollutant_risks(iaqi_data) == []

def test_interpret_risks_single_pollutant_moderate():
    """Tests when a single pollutant (PM2.5) crosses the 'Moderate' threshold."""
    iaqi_data = {'pm25': {'v': 70}, 'pm10': {'v': 90}}
    risks = interpret_pollutant_risks(iaqi_data)
    assert len(risks) == 1
    assert "PM25 (Moderate):" in risks[0]

def test_interpret_risks_multiple_pollutants():
    """Tests when multiple pollutants trigger risks at different severity levels."""
    iaqi_data = {'pm10': {'v': 260}, 'o3': {'v': 115}} 
    risks = interpret_pollutant_risks(iaqi_data)
    assert len(risks) == 2
    assert any("PM10 (Poor):" in risk for risk in risks)
    assert any("O3 (Moderate):" in risk for risk in risks)

def test_interpret_risks_highest_severity_is_chosen():
    """Ensures that only the risk for the highest threshold exceeded is reported per pollutant."""
    iaqi_data = {'pm25': {'v': 130}} 
    risks = interpret_pollutant_risks(iaqi_data)
    assert len(risks) == 1
    assert "PM25 (Very Poor):" in risks[0]

# --- Edge Case and Invalid Input Tests ---

def test_interpret_risks_boundary_values():
    """Tests values that fall exactly on a threshold boundary, and just below it."""
    assert len(interpret_pollutant_risks({'pm25': {'v': 61}})) == 1
    assert len(interpret_pollutant_risks({'pm25': {'v': 60.9}})) == 0

def test_interpret_risks_with_unknown_pollutant():
    """Tests that pollutants not defined in the thresholds dictionary are ignored."""
    iaqi_data = {'pm25': {'v': 200}, 'xyz': {'v': 100}} 
    risks = interpret_pollutant_risks(iaqi_data)
    assert len(risks) == 1
    assert "PM25 (Very Poor):" in risks[0]

@pytest.mark.parametrize("malformed_data, description", [
    ({'pm25': {'v': "high"}}, "Value is a string"),
    ({'pm25': {'value': 100}}, "Missing the 'v' key"),
    ({'pm25': 100}, "Value is not a dictionary"),
])
def test_interpret_risks_malformed_input_data(malformed_data, description):
    """
    Tests that various forms of malformed input data are handled gracefully
    and result in an empty list of risks.
    """
    risks = interpret_pollutant_risks(malformed_data)
    assert risks == [], f"Test failed for malformed data case: {description}"

def test_interpret_risks_empty_or_none_input():
    """Tests that an empty dictionary or a None input returns an empty list."""
    assert interpret_pollutant_risks({}) == []
    assert interpret_pollutant_risks(None) == []