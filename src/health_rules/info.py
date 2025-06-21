# File: src/health_rules/info.py

"""
Defines the CPCB Air Quality Index (AQI) scale and provides related utilities.

This module contains the definitions for the Indian CPCB NAQI standard,
including health implications and color codes. It provides a lookup function
to classify a numerical AQI value into its corresponding category.
"""

import logging 
import pandas as pd # Used for the robust pd.isna check


log = logging.getLogger(__name__)


# --- AQI Definition ---
# A general-purpose description of the Air Quality Index for educational display.
AQI_DEFINITION = """
The Air Quality Index (AQI) is a tool used by government agencies to communicate how polluted the air currently is or how polluted it is forecast to become.
It helps you understand the potential health effects associated with different levels of air quality.
"""

# --- AQI Scale and Health Implications (India CPCB NAQI Standard) ---
# This list defines the official CPCB AQI categories, ranges, health impacts,
# and standard colors.
# Source: Central Pollution Control Board (CPCB), India.
AQI_SCALE = [
    {"range": "0-50", "level": "Good", "color": "#228B22", "implications": "Minimal Impact. Air quality is considered satisfactory, and air pollution poses little or no risk."},
    {"range": "51-100", "level": "Satisfactory", "color": "#90EE90", "implications": "Minor breathing discomfort to sensitive people. Air quality is acceptable."},
    {"range": "101-200", "level": "Moderate", "color": "#FFD700", "implications": "Breathing discomfort to people with lung disease such as asthma, and discomfort to people with heart disease, children and older adults."},
    {"range": "201-300", "level": "Poor", "color": "#FFA500", "implications": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease."},
    {"range": "301-400", "level": "Very Poor", "color": "#FF0000", "implications": "Respiratory illness on prolonged exposure. Effect may be more pronounced in people with lung and heart diseases."},
    {"range": "401-500", "level": "Severe", "color": "#800000", "implications": "Affects healthy people and seriously impacts those with existing diseases. May cause respiratory impact even on light physical activity."}
]


def get_aqi_info(aqi_value):
    """
    Finds the CPCB AQI category details for a given numerical AQI value.

    Args:
        aqi_value (int | float | None): The numerical AQI value to classify.

    Returns:
        dict | None: A dictionary containing the category details ('range',
                     'level', 'color', 'implications') if a match is found.
                     Returns the 'Severe' category for any value above the
                     highest defined range. Returns None for invalid inputs
                     (e.g., negative, non-numeric, or None).
    """
    # Validate input: return None for None, non-numeric, or negative values.
    if pd.isna(aqi_value) or not isinstance(aqi_value, (int, float)) or aqi_value < 0:
        log.warning(f"Invalid AQI value received: {aqi_value}. Returning None.")
        return None

    # Find the matching category in the defined scale.
    for category in AQI_SCALE:
        try:
            if '-' in category['range']:
                low, high = map(int, category['range'].split('-'))
                if low <= aqi_value <= high:
                    return category
        except ValueError:
            log.error(f"Could not parse AQI range string: {category['range']}")
            continue

    # Handle cases where the AQI value is above the highest defined range.
    try:
         if AQI_SCALE and '-' in AQI_SCALE[-1]['range']:
             last_high = int(AQI_SCALE[-1]['range'].split('-')[1])
             if aqi_value > last_high:
                  log.info(f"AQI value {aqi_value} > {last_high}, classifying as '{AQI_SCALE[-1]['level']}'.")
                  return AQI_SCALE[-1]
         else:
              log.warning("Cannot determine upper bound from AQI_SCALE configuration.")
    except (ValueError, IndexError) as e:
         log.error(f"Error processing upper bound of AQI_SCALE: {e}")

    # This line should ideally not be reached with valid non-negative numbers.
    log.warning(f"Could not find matching AQI category for value: {aqi_value}")
    return None

# --- Example Usage / Direct Execution ---
if __name__ == "__main__":
    # This block runs only when the script is executed directly.
    # It serves as a quick test and demonstration of the module's functions.
    # The comprehensive, automated tests for this module are in:
    # tests/health_rules/test_info.py

    print("\n" + "="*40)
    print(" Running info.py Self-Test ")
    print("="*40 + "\n")

    print("--- Testing AQI_DEFINITION Constant ---")
    print(AQI_DEFINITION)

    print("\n" + "-"*40)
    print("--- Testing get_aqi_info with valid values ---")

    # A list of test values covering all categories and boundaries
    valid_test_values = [
        0, 25, 50,          # Good
        51, 75, 100,         # Satisfactory
        101, 150, 200,       # Moderate
        201, 250, 300,       # Poor
        301, 350, 400,       # Very Poor
        401, 450, 500,       # Severe
        550, 1000            # Above range (should be Severe)
    ]

    for aqi in valid_test_values:
        info = get_aqi_info(aqi)
        if info:
            # Using f-string alignment to make the output neat
            print(f"AQI: {aqi:<4} -> Level: {info['level']:<12} | Color: {info['color']:<8} | Implications: {info['implications']}")
        else:
            print(f"AQI: {aqi:<4} -> FAILED to get info, returned None unexpectedly.")

    print("\n" + "-"*40)
    print("--- Testing get_aqi_info with invalid values ---")

    invalid_test_values = [
        None,
        -10,
        "not a number"
    ]

    for aqi in invalid_test_values:
        info = get_aqi_info(aqi)
        # For invalid inputs, the expected result is None
        if info is None:
            print(f"Input: {str(aqi):<15} -> Correctly returned None as expected.")
        else:
            print(f"Input: {str(aqi):<15} -> FAILED, expected None but got a result: {info}")

    print("\n" + "="*40)
    print(" info.py Self-Test Finished ")
    print("="*40 + "\n")