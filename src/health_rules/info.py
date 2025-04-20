# File: src/health_rules/info.py

import logging # Using logging for potential errors during range parsing

# --- AQI Definition ---
# Concise definition based on common understanding and CPCB context.
AQI_DEFINITION = """
The Air Quality Index (AQI) is a tool used by government agencies to communicate how polluted the air currently is or how polluted it is forecast to become. 
It helps you understand the potential health effects associated with different levels of air quality.
"""

# --- AQI Scale and Health Implications (Based on India CPCB NAQI Standard) ---
# This list defines the CPCB AQI categories, ranges, associated health impacts, and standard colours.
AQI_SCALE = [
    {
        "range": "0-50",
        "level": "Good",
        "color": "#228B22", # Forest Green (representing Dark Green used by CPCB)
        "implications": "Minimal Impact. Air quality is considered satisfactory, and air pollution poses little or no risk."
    },
    {
        "range": "51-100",
        "level": "Satisfactory", # CPCB uses 'Satisfactory' here
        "color": "#90EE90", # Light Green
        "implications": "Minor breathing discomfort to sensitive people. Air quality is acceptable."
    },
    {
        "range": "101-200",
        "level": "Moderate", # CPCB uses 'Moderate' here
        "color": "#FFD700", # Gold (representing Yellow used by CPCB)
        "implications": "Breathing discomfort to people with lung disease such as asthma, and discomfort to people with heart disease, children and older adults."
    },
    {
        "range": "201-300",
        "level": "Poor", # CPCB uses 'Poor' here
        "color": "#FFA500", # Orange
        "implications": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease."
    },
    {
        "range": "301-400",
        "level": "Very Poor", # CPCB uses 'Very Poor' here
        "color": "#FF0000", # Red
        "implications": "Respiratory illness on prolonged exposure. Effect may be more pronounced in people with lung and heart diseases."
    },
    {
        "range": "401-500", # CPCB scale usually caps the 'Severe' implications description here
        "level": "Severe", # CPCB uses 'Severe' here
        "color": "#800000", # Maroon (representing Dark Red used by CPCB)
        "implications": "Affects healthy people and seriously impacts those with existing diseases. May cause respiratory impact even on light physical activity."
    }
    # Note: CPCB sometimes refers to >500 as 'Severe+' or 'Emergency', often with similar health warnings as Severe.
    # The function below handles values > 500 by mapping them to the 'Severe' category.
]


# --- Function to easily get info by AQI value ---
def get_aqi_info(aqi_value):
    """
    Finds the corresponding CPCB AQI category information for a given numerical AQI value.

    Args:
        aqi_value (int or float): The numerical AQI value.

    Returns:
        dict: A dictionary containing the 'range', 'level', 'color', and 'implications'
              for the matching category, or None if the value is invalid (< 0 or non-numeric).
    """
    # Validate input
    if aqi_value is None or not isinstance(aqi_value, (int, float)) or aqi_value < 0:
        logging.warning(f"Invalid AQI value received: {aqi_value}. Returning None.")
        return None # Handle invalid input

    for category in AQI_SCALE:
        try:
            # Parse the range string (e.g., "101-200")
            if '-' in category['range']:
                low, high = map(int, category['range'].split('-'))
                # Check if the value falls within the range (inclusive)
                if low <= aqi_value <= high:
                    return category
            # Add handling for potential future single-value ranges or '+' notation if needed
            # elif category['range'].endswith('+'): ...
            # else: ...
        except ValueError:
            # Log error if range format is unexpected (shouldn't happen with current scale)
            logging.error(f"Could not parse AQI range string: {category['range']}")
            continue # Skip this category if parsing fails

    # Handle values potentially above the highest defined range (e.g., > 500)
    # Return the information for the highest category ('Severe' in this case)
    if AQI_SCALE and aqi_value > int(AQI_SCALE[-1]['range'].split('-')[1]): # Check against upper bound of last range
        logging.info(f"AQI value {aqi_value} is above highest defined range, classifying as '{AQI_SCALE[-1]['level']}'.")
        return AQI_SCALE[-1]

    # Should ideally not be reached if scale covers 0+ and logic above works
    logging.warning(f"Could not find matching AQI category for value: {aqi_value}")
    return None


# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Setup logger for testing output if not already configured
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

    print("\n" + "="*30)
    print(" Testing AQI Info Module (CPCB) ")
    print("="*30 + "\n")

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