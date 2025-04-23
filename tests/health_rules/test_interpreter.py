# File: tests/health_rules/test_interpreter.py

import pytest
import sys
import os

# --- Add project root to sys.path for imports ---
try:
    TEST_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname('.'), '..'))
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Import the function to be tested ---
# Also import the thresholds dict if you want to directly reference levels in asserts
from src.health_rules.interpreter import interpret_pollutant_risks, POLLUTANT_HEALTH_THRESHOLDS

# --- Test Cases for interpret_pollutant_risks ---

def test_interpret_risks_clean_air():
    """Tests with pollutant values below all defined thresholds."""
    iaqi_data = {
        'pm25': {'v': 10.5}, 'pm10': {'v': 45}, 'o3': {'v': 30},
        'no2': {'v': 20}, 'so2': {'v': 5}, 'co': {'v': 0.8}
    }
    risks = interpret_pollutant_risks(iaqi_data)
    assert isinstance(risks, list)
    assert len(risks) == 0 # Expect no risks triggered

def test_interpret_risks_moderate_pm25():
    """Tests when only PM2.5 crosses the 'Moderate' threshold."""
    # PM2.5 Moderate threshold starts at 61
    iaqi_data = {'pm25': {'v': 70}, 'pm10': {'v': 90}}
    risks = interpret_pollutant_risks(iaqi_data)
    assert len(risks) == 1
    assert "PM25 (Moderate):" in risks[0]
    assert "Breathing discomfort" in risks[0] # Check substring of the message

def test_interpret_risks_poor_pm10_moderate_o3():
    """Tests when PM10 is 'Poor' and O3 is 'Moderate'."""
    # PM10 Poor starts at 251, O3 Moderate starts at 101
    iaqi_data = {'pm10': {'v': 260}, 'o3': {'v': 115}, 'pm25': {'v': 50}}
    risks = interpret_pollutant_risks(iaqi_data)
    assert len(risks) == 2
    # The order might vary depending on dictionary iteration, so check presence
    assert any("PM10 (Poor):" in risk for risk in risks)
    assert any("O3 (Moderate):" in risk for risk in risks)

def test_interpret_risks_severe_multiple():
    """Tests when multiple pollutants are in 'Severe' range."""
    # PM2.5 Severe >= 251, NO2 Severe >= 401
    iaqi_data = {'pm25': {'v': 300}, 'no2': {'v': 410}, 'so2': {'v': 100}}
    risks = interpret_pollutant_risks(iaqi_data)
    assert len(risks) == 3
    assert any("PM25 (Severe):" in risk for risk in risks)
    assert any("NO2 (Severe):" in risk for risk in risks)
    assert any("SO2 (Moderate):" in risk for risk in risks) 
    assert any("Serious respiratory impact" in risk for risk in risks) # Check part of severe message

def test_interpret_risks_highest_severity_chosen():
    """Tests that only the highest severity risk is reported per pollutant."""
    # PM2.5 = 130 -> Should trigger Very Poor (>=121), not Poor (>=91) or Moderate (>=61)
    iaqi_data = {'pm25': {'v': 130}}
    risks = interpret_pollutant_risks(iaqi_data)
    assert len(risks) == 1
    assert "PM25 (Very Poor):" in risks[0]
    assert "PM25 (Poor):" not in risks[0] # Ensure lower severity isn't included implicitly
    assert "Respiratory illness on prolonged exposure" in risks[0] # Check V.Poor message

def test_interpret_risks_boundary_values():
    """Tests values exactly matching thresholds."""
    # PM2.5 Moderate threshold starts at 61
    iaqi_data_pm25_mod = {'pm25': {'v': 61}}
    risks_pm25_mod = interpret_pollutant_risks(iaqi_data_pm25_mod)
    assert len(risks_pm25_mod) == 1
    assert "PM25 (Moderate):" in risks_pm25_mod[0]

    # PM10 Poor threshold starts at 251
    iaqi_data_pm10_poor = {'pm10': {'v': 251}}
    risks_pm10_poor = interpret_pollutant_risks(iaqi_data_pm10_poor)
    assert len(risks_pm10_poor) == 1
    assert "PM10 (Poor):" in risks_pm10_poor[0]

    # Test just below a threshold
    iaqi_data_below_mod = {'pm25': {'v': 60.9}}
    risks_below_mod = interpret_pollutant_risks(iaqi_data_below_mod)
    assert len(risks_below_mod) == 0 # Should not trigger Moderate

def test_interpret_risks_missing_pollutant():
    """Tests input data missing some expected pollutants."""
    iaqi_data = {'pm10': {'v': 150}} # Missing pm25, o3 etc.
    risks = interpret_pollutant_risks(iaqi_data)
    assert len(risks) == 1
    assert "PM10 (Moderate):" in risks[0]

def test_interpret_risks_pollutant_not_in_thresholds():
    """Tests input data with a pollutant not defined in our thresholds dict."""
    iaqi_data = {'pm25': {'v': 200}, 'xyz': {'v': 100}} # 'xyz' is not defined
    risks = interpret_pollutant_risks(iaqi_data)
    assert len(risks) == 1 # Should only find risk for pm25
    assert "PM25 (Very Poor):" in risks[0] # 200 falls into Very Poor for PM2.5 (>=121)

def test_interpret_risks_invalid_value_format():
    """Tests input data with incorrect format for pollutant value."""
    iaqi_data_str = {'pm25': {'v': "high"}} # Value is string
    risks_str = interpret_pollutant_risks(iaqi_data_str)
    assert len(risks_str) == 0

    iaqi_data_no_v = {'pm25': {'value': 100}} # Missing 'v' key
    risks_no_v = interpret_pollutant_risks(iaqi_data_no_v)
    assert len(risks_no_v) == 0

    iaqi_data_not_dict = {'pm25': 100} # Pollutant value is not a dict
    risks_not_dict = interpret_pollutant_risks(iaqi_data_not_dict)
    assert len(risks_not_dict) == 0

def test_interpret_risks_empty_or_none_input():
    """Tests behavior with empty dictionary or None as input."""
    assert interpret_pollutant_risks({}) == []
    assert interpret_pollutant_risks(None) == []