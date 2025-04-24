# File: src/health_rules/info.py

"""
Provides definitions and utility functions related to the Air Quality Index (AQI),
specifically following the Indian CPCB NAQI standard.

Contains:
- A definition string for AQI.
- A list representing the CPCB AQI scale (ranges, levels, colors, implications).
- A function to look up the AQI category details based on a numerical value.
"""

import logging # Standard logging import
import pandas as pd # Added for pd.isna check

# --- Get Logger ---
# Get the logger instance for this module. It inherits root config.
log = logging.getLogger(__name__)
# Note: CONFIG is not explicitly needed here, logging config comes via root logger

# --- AQI Definition ---
AQI_DEFINITION = """
The Air Quality Index (AQI) is a tool used by government agencies to communicate how polluted the air currently is or how polluted it is forecast to become.
It helps you understand the potential health effects associated with different levels of air quality.
"""

# --- AQI Scale and Health Implications (Based on India CPCB NAQI Standard) ---
# This list defines the CPCB AQI categories, ranges, associated health impacts, and standard colours.
AQI_SCALE = [
    {"range": "0-50", "level": "Good", "color": "#228B22", "implications": "Minimal Impact. Air quality is considered satisfactory, and air pollution poses little or no risk."},
    {"range": "51-100", "level": "Satisfactory", "color": "#90EE90", "implications": "Minor breathing discomfort to sensitive people. Air quality is acceptable."},
    {"range": "101-200", "level": "Moderate", "color": "#FFD700", "implications": "Breathing discomfort to people with lung disease such as asthma, and discomfort to people with heart disease, children and older adults."},
    {"range": "201-300", "level": "Poor", "color": "#FFA500", "implications": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease."},
    {"range": "301-400", "level": "Very Poor", "color": "#FF0000", "implications": "Respiratory illness on prolonged exposure. Effect may be more pronounced in people with lung and heart diseases."},
    {"range": "401-500", "level": "Severe", "color": "#800000", "implications": "Affects healthy people and seriously impacts those with existing diseases. May cause respiratory impact even on light physical activity."}
]

# --- Function to easily get info by AQI value ---
def get_aqi_info(aqi_value):
    """Finds the CPCB AQI category details for a given numerical AQI value.

    Iterates through the AQI_SCALE list and returns the dictionary corresponding
    to the range the input value falls into. Handles values above the highest
    defined range by assigning them to the 'Severe' category.

    Args:
        aqi_value (int or float or None): The numerical AQI value to classify.
                                          Handles None or non-numeric types gracefully.

    Returns:
        dict or None: A dictionary containing the category details:
                      - 'range' (str): The numerical range (e.g., "101-200").
                      - 'level' (str): The category name (e.g., "Moderate").
                      - 'color' (str): The associated hex color code.
                      - 'implications' (str): The health implication text.
                      Returns None if the input aqi_value is invalid (e.g., negative,
                      non-numeric, None) or if no matching range is found (which
                      should not happen for valid non-negative numbers given the scale).
    """
    # (Function code remains the same)
    if pd.isna(aqi_value) or not isinstance(aqi_value, (int, float)) or aqi_value < 0:
        log.warning(f"Invalid AQI value received: {aqi_value}. Returning None.")
        return None

    for category in AQI_SCALE:
        try:
            if '-' in category['range']:
                low, high = map(int, category['range'].split('-'))
                if low <= aqi_value <= high:
                    return category
        except ValueError:
            log.error(f"Could not parse AQI range string: {category['range']}")
            continue

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

    log.warning(f"Could not find matching AQI category for value: {aqi_value}")
    return None

# --- Example Usage Block ---
# (Keep existing __main__ block as is)
if __name__ == "__main__":
    # ... (test code remains the same) ...
    pass # Added pass for valid syntax if test code removed/commented