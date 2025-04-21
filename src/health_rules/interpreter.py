# File: src/health_rules/interpreter.py

import logging

# --- Pollutant Thresholds and Associated Risks (Derived from CPCB NAQI Breakpoints) ---
# Defines thresholds based on the *start* of CPCB concentration ranges for different
# AQI categories (Moderate, Poor, Very Poor, Severe). The associated risks use language
# adapted from the health impacts described for those *overall AQI categories*.
#
# !!! KEY ASSUMPTIONS & LIMITATIONS !!!
# 1.  AVERAGING TIME MISMATCH: CPCB health impacts relate to 24hr/8hr averages.
#     This code applies them to near real-time API values as an APPROXIMATION.
# 2.  API UNITS ('v'): Assumes the 'v' value from AQICN API for each pollutant is
#     roughly comparable to the standard units used by CPCB (µg/m³ for most, mg/m³ for CO).
#     Verify this assumption based on API documentation or data patterns.
# 3.  INDICATIVE ONLY: This provides an *indication* of potential risk based on individual
#     pollutant levels, not the official calculated AQI health statement or medical advice.

# CPCB NAQI Breakpoints (Concentration ranges defining AQI categories)
# PM2.5 (24hr, µg/m³): 61(Mod), 91(Poor), 121(V.Poor), 251(Severe)
# PM10  (24hr, µg/m³): 101(Mod), 251(Poor), 351(V.Poor), 431(Severe)
# O3    (8hr, µg/m³):  101(Mod), 169(Poor), 209(V.Poor), 749(Severe) - Note: >748 is severe break.
# NO2   (24hr, µg/m³): 81(Mod), 181(Poor), 281(V.Poor), 401(Severe)
# SO2   (24hr, µg/m³): 81(Mod), 381(Poor), 801(V.Poor), 1601(Severe)
# CO    (8hr, mg/m³):  2.1(Mod), 10.1(Poor), 17.1(V.Poor), 34.1(Severe) -> Need unit check for API 'v'

POLLUTANT_HEALTH_THRESHOLDS = {
    "pm25": [ # PM2.5 (µg/m³) - Approximation based on CPCB 24hr breakpoints
        {"threshold": 251, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"},
        {"threshold": 121, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 91,  "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 61,  "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
        # Optional: Add 'Satisfactory' level if needed, threshold: 31
        # {"threshold": 31,  "risk": "Minor breathing discomfort to sensitive people.", "severity": "Satisfactory"},
    ],
    "pm10": [ # PM10 (µg/m³) - Approximation based on CPCB 24hr breakpoints
        {"threshold": 431, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"},
        {"threshold": 351, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 251, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 101, "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
        # Optional: Add 'Satisfactory' level if needed, threshold: 51
        # {"threshold": 51,  "risk": "Minor breathing discomfort to sensitive people.", "severity": "Satisfactory"},
    ],
    "o3": [ # Ozone (µg/m³) - Approximation based on CPCB 8hr breakpoints
        {"threshold": 749, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"}, # Note: CPCB uses >748
        {"threshold": 209, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 169, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 101, "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
        # Optional: Add 'Satisfactory' level if needed, threshold: 51
        # {"threshold": 51,  "risk": "Minor breathing discomfort to sensitive people.", "severity": "Satisfactory"},
    ],
    "no2": [ # Nitrogen Dioxide (µg/m³) - Approximation based on CPCB 24hr breakpoints
        {"threshold": 401, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"},
        {"threshold": 281, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 181, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 81,  "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
        # Optional: Add 'Satisfactory' level if needed, threshold: 41
        # {"threshold": 41,  "risk": "Minor breathing discomfort to sensitive people.", "severity": "Satisfactory"},
    ],
    "so2": [ # Sulfur Dioxide (µg/m³) - Approximation based on CPCB 24hr breakpoints
        {"threshold": 1601, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"},
        {"threshold": 801, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 381, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 81,  "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
        # Optional: Add 'Satisfactory' level if needed, threshold: 41
        # {"threshold": 41,  "risk": "Minor breathing discomfort to sensitive people.", "severity": "Satisfactory"},
    ],
    "co": [ # Carbon Monoxide (mg/m³) - Approximation based on CPCB 8hr breakpoints
           # !!! Crucial: Check if API 'v' value for CO aligns with mg/m³ or needs conversion/different interpretation !!!
        {"threshold": 34.1, "risk": "Serious aggravation of heart or lung disease; may cause respiratory effects even during light activity.", "severity": "Severe"},
        {"threshold": 17.1, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 10.1, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 2.1,  "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
        # Optional: Add 'Satisfactory' level if needed, threshold: 1.1
        # {"threshold": 1.1,  "risk": "Minor breathing discomfort to sensitive people.", "severity": "Satisfactory"},
    ]
}

def interpret_pollutant_risks(iaqi_data):
    """
    Analyzes individual pollutant levels from AQICN data and identifies potential
    respiratory health risks based on thresholds derived from CPCB NAQI breakpoints.

    Args:
        iaqi_data (dict): The 'iaqi' part of the AQICN API response, containing
                          pollutant keys (e.g., 'pm25', 'o3') mapped to dictionaries
                          containing the value under the 'v' key (e.g., {'v': 150}).

    Returns:
        list: A list of strings, each describing a potential health risk identified
              based on the pollutant levels exceeding CPCB-derived thresholds.
              Returns an empty list if no thresholds are exceeded or input is invalid.
              Example: ["PM2.5 (Poor): Breathing discomfort to people on prolonged exposure..."]
    """
    triggered_risks = []
    if not iaqi_data or not isinstance(iaqi_data, dict):
        logging.warning("Invalid or empty iaqi_data received for interpretation.")
        return triggered_risks

    logging.info(f"Interpreting risks using CPCB-derived thresholds for iaqi data: {iaqi_data}")

    for pollutant, thresholds in POLLUTANT_HEALTH_THRESHOLDS.items():
        if pollutant in iaqi_data and isinstance(iaqi_data[pollutant], dict) and 'v' in iaqi_data[pollutant]:
            try:
                # Attempt to convert the 'v' value to a float for comparison
                value = float(iaqi_data[pollutant]['v'])
                logging.debug(f"Checking {pollutant.upper()} with value {value}")

                # Check thresholds from highest severity to lowest
                highest_risk_found = None
                # Sort thresholds by 'threshold' value descending to check most severe first
                for level_info in sorted(thresholds, key=lambda x: x['threshold'], reverse=True):
                    if value >= level_info["threshold"]:
                        # Format the risk string including pollutant name and severity level
                        highest_risk_found = f"{pollutant.upper()} ({level_info['severity']}): {level_info['risk']}"
                        logging.info(f"Threshold exceeded for {pollutant.upper()} at value {value} (>= {level_info['threshold']}). Risk: {level_info['risk']}")
                        break # Found highest triggered risk for this pollutant, stop checking lower ones

                if highest_risk_found:
                    triggered_risks.append(highest_risk_found)

            except (ValueError, TypeError) as e:
                # Log warning if the 'v' value isn't a valid number
                logging.warning(f"Could not parse value for pollutant '{pollutant}': {iaqi_data[pollutant].get('v')}. Error: {e}")
                continue # Skip to the next pollutant
        else:
            # Log debug message if pollutant not found or data format is wrong
            logging.debug(f"Pollutant '{pollutant}' not found in iaqi_data or format invalid: {iaqi_data.get(pollutant)}")

    if not triggered_risks:
        logging.info("No significant pollutant thresholds exceeded based on CPCB-derived rules.")

    # Return the list of identified risk strings
    return triggered_risks

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Setup logger if running directly
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

    print("\n" + "="*40)
    print(" Testing Health Risk Interpreter (CPCB Based)")
    print("="*40 + "\n")

    # Example 1: Data similar to the previous Delhi test output
    # PM2.5=161 (Very Poor), PM10=178 (Moderate), others likely Moderate/Satisfactory/Good
    test_data_delhi = {
        'co': {'v': 1.2}, 'h': {'v': 27.3}, 'no2': {'v': 15.8}, # NO2 < 81
        'o3': {'v': 41.2}, 'p': {'v': 983.2}, 'pm10': {'v': 178}, # PM10 >= 101
        'pm25': {'v': 161}, 'so2': {'v': 7.6}, 't': {'v': 36.7}, # PM2.5 >= 121
        'w': {'v': 0.5}, 'wd': {'v': 329.7}, 'wg': {'v': 8.2}  # CO < 2.1, SO2 < 81
    }
    print(f"--- Testing with Delhi-like data: {test_data_delhi} ---")
    risks_delhi = interpret_pollutant_risks(test_data_delhi)
    print("Identified Risks:")
    if risks_delhi:
        for risk in risks_delhi:
            print(f"- {risk}")
    else:
        print("- None")

    # Example 2: Hypothetical Very Poor PM10, Moderate O3
    test_data_vpoor_pm10 = {
        'pm10': {'v': 360}, 'o3': {'v': 110}, 'no2': {'v': 50} # PM10 >= 351, O3 >= 101
    }
    print(f"\n--- Testing with Very Poor PM10 data: {test_data_vpoor_pm10} ---")
    risks_vpoor_pm10 = interpret_pollutant_risks(test_data_vpoor_pm10)
    print("Identified Risks:")
    if risks_vpoor_pm10:
        for risk in risks_vpoor_pm10:
            print(f"- {risk}")
    else:
        print("- None")

    # Example 3: Clean air data
    test_data_clean = {
        'pm25': {'v': 10}, 'pm10': {'v': 20}, 'o3': {'v': 30}, 'no2': {'v': 15}, 'co': {'v': 0.5}
    }
    print(f"\n--- Testing with Clean Air data: {test_data_clean} ---")
    risks_clean = interpret_pollutant_risks(test_data_clean)
    print("Identified Risks:")
    if risks_clean:
        for risk in risks_clean:
            print(f"- {risk}")
    else:
        print("- None (Expected)")

    # Example 4: Invalid/Empty data (Should remain the same)
    print("\n--- Testing with Invalid/Empty data ---")
    risks_invalid = interpret_pollutant_risks(None)
    print(f"Invalid data risks: {risks_invalid} (Expected: [])")
    risks_empty = interpret_pollutant_risks({})
    print(f"Empty data risks: {risks_empty} (Expected: [])")
    risks_bad_format = interpret_pollutant_risks({'pm25': 150}) # Incorrect format
    print(f"Bad format data risks: {risks_bad_format} (Expected: [])")


    print("\n" + "="*40)
    print(" Health Risk Interpreter (CPCB Based) Tests Finished ")
    print("="*40 + "\n")