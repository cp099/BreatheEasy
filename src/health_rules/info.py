# File: src/health_rules/info.py

import logging # Standard logging import
import pandas as pd # Added for pd.isna check

# --- Get Logger ---
# Get the logger instance for this module. It inherits root config.
log = logging.getLogger(__name__)
# Note: CONFIG is not explicitly needed here, logging config comes via root logger

# --- AQI Definition ---
# (Keep AQI_DEFINITION exactly as it was)
AQI_DEFINITION = """
The Air Quality Index (AQI) is a tool used by government agencies to communicate how polluted the air currently is or how polluted it is forecast to become.
It helps you understand the potential health effects associated with different levels of air quality.
"""

# --- AQI Scale and Health Implications (Based on India CPCB NAQI Standard) ---
# (Keep AQI_SCALE list exactly as it was)
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
    """
    Finds the corresponding CPCB AQI category information for a given numerical AQI value.
    """
    # Validate input using pandas isna for broader check
    if pd.isna(aqi_value) or not isinstance(aqi_value, (int, float)) or aqi_value < 0:
        # Use the module logger instance
        log.warning(f"Invalid AQI value received: {aqi_value}. Returning None.")
        return None

    for category in AQI_SCALE:
        try:
            if '-' in category['range']:
                low, high = map(int, category['range'].split('-'))
                if low <= aqi_value <= high:
                    return category
            # Handle '+' notation if added later
            # elif category['range'].endswith('+'): ...
        except ValueError:
            log.error(f"Could not parse AQI range string: {category['range']}")
            continue

    # Handle values above the highest defined range
    try:
         # Ensure AQI_SCALE is not empty and the last range is parseable
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

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Logging is configured when src.config_loader is imported by any module,
    # including this one when run directly (due to imports in other modules).
    # If running this file *completely* standalone, uncommenting basicConfig below might be needed.
    # if not logging.getLogger().hasHandlers():
    #      logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

    print("\n" + "="*30)
    print(" Testing AQI Info Module (CPCB) ")
    print("="*30 + "\n")
    # (Keep the rest of the test block exactly as it was)
    print(f"--- AQI Definition ---\n{AQI_DEFINITION}")
    print("\n--- CPCB AQI Scale ---")
    for cat in AQI_SCALE:
        print(f"Range: {cat['range']:<7} | Level: {cat['level']:<12} | Color: {cat['color']:<7} | Implications: {cat['implications']}")
    print("\n--- Testing get_aqi_info() Function ---")
    test_values = [0, 25, 50, 51, 75, 100, 101, 150, 200, 201, 250, 300, 301, 350, 400, 401, 450, 500, 550, -10, None, "abc"]
    for val in test_values:
        info = get_aqi_info(val)
        if info:
            print(f"AQI Value {str(val):>3} -> Level: {info['level']:<12} (Color: {info['color']:<7})")
        else:
            print(f"AQI Value {str(val):>3} -> Invalid input or no category found.")
    print("\n" + "="*30)
    print(" AQI Info Module Tests Finished ")
    print("="*30 + "\n")