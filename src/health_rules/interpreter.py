# File: src/health_rules/interpreter.py

import logging
import pandas as pd # Needed for pd.isna check if used, though not strictly needed now

# --- Logging Configuration ---
# Configure logger specifically for this module if needed, or rely on root logger
log = logging.getLogger(__name__)

# --- Pollutant Thresholds and Associated Risks (Derived from CPCB NAQI Breakpoints) ---
# (Keep the POLLUTANT_HEALTH_THRESHOLDS dictionary exactly as it was in the previous correct version)
# ... (Paste the full POLLUTANT_HEALTH_THRESHOLDS dictionary here) ...
POLLUTANT_HEALTH_THRESHOLDS = {
    "pm25": [ # PM2.5 (µg/m³) - Approximation based on CPCB 24hr breakpoints
        {"threshold": 251, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"},
        {"threshold": 121, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 91,  "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 61,  "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
    ],
    "pm10": [ # PM10 (µg/m³) - Approximation based on CPCB 24hr breakpoints
        {"threshold": 431, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"},
        {"threshold": 351, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 251, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 101, "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
    ],
    "o3": [ # Ozone (µg/m³) - Approximation based on CPCB 8hr breakpoints
        {"threshold": 749, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"}, # Note: CPCB uses >748
        {"threshold": 209, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 169, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 101, "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
    ],
    "no2": [ # Nitrogen Dioxide (µg/m³) - Approximation based on CPCB 24hr breakpoints
        {"threshold": 401, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"},
        {"threshold": 281, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 181, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 81,  "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
    ],
    "so2": [ # Sulfur Dioxide (µg/m³) - Approximation based on CPCB 24hr breakpoints
        {"threshold": 1601, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"},
        {"threshold": 801, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 381, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 81,  "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
    ],
    "co": [ # Carbon Monoxide (mg/m³) - Approximation based on CPCB 8hr breakpoints
        {"threshold": 34.1, "risk": "Serious aggravation of heart or lung disease; may cause respiratory effects even during light activity.", "severity": "Severe"},
        {"threshold": 17.1, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 10.1, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 2.1,  "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
    ]
}


def interpret_pollutant_risks(iaqi_data):
    """
    Analyzes individual pollutant levels from AQICN data and identifies potential
    respiratory health risks based on thresholds derived from CPCB NAQI breakpoints.
    (Function definition remains exactly the same as the previous correct version)

    Args:
        iaqi_data (dict): The 'iaqi' part of the AQICN API response.

    Returns:
        list: A list of strings describing potential health risks.
    """
    triggered_risks = []
    if not iaqi_data or not isinstance(iaqi_data, dict):
        log.warning("Invalid or empty iaqi_data received for interpretation.")
        return triggered_risks

    log.info(f"Interpreting risks using CPCB-derived thresholds for iaqi data: {iaqi_data}")

    for pollutant, thresholds in POLLUTANT_HEALTH_THRESHOLDS.items():
        if pollutant in iaqi_data and isinstance(iaqi_data[pollutant], dict) and 'v' in iaqi_data[pollutant]:
            try:
                value = float(iaqi_data[pollutant]['v'])
                log.debug(f"Checking {pollutant.upper()} with value {value}")
                highest_risk_found = None
                for level_info in sorted(thresholds, key=lambda x: x['threshold'], reverse=True):
                    if value >= level_info["threshold"]:
                        highest_risk_found = f"{pollutant.upper()} ({level_info['severity']}): {level_info['risk']}"
                        log.info(f"Threshold exceeded for {pollutant.upper()} at value {value} (>= {level_info['threshold']}). Risk: {level_info['risk']}")
                        break
                if highest_risk_found:
                    triggered_risks.append(highest_risk_found)
            except (ValueError, TypeError) as e:
                log.warning(f"Could not parse value for pollutant '{pollutant}': {iaqi_data[pollutant].get('v')}. Error: {e}")
                continue
        else:
            log.debug(f"Pollutant '{pollutant}' not found or format invalid: {iaqi_data.get(pollutant)}")

    if not triggered_risks:
        log.info("No significant pollutant thresholds exceeded based on CPCB-derived rules.")
    return triggered_risks


# --- Example Usage (for testing THIS module - interpret_pollutant_risks only) ---
if __name__ == "__main__":
    # Setup logger if running directly
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

    print("\n" + "="*40)
    print(" Testing Health Risk Interpreter (CPCB Based) - Current Pollutants")
    print("="*40 + "\n")

    # --- Keep only the tests for interpret_pollutant_risks ---
    test_data_delhi = {
        'co': {'v': 1.2}, 'h': {'v': 27.3}, 'no2': {'v': 15.8},
        'o3': {'v': 41.2}, 'p': {'v': 983.2}, 'pm10': {'v': 178},
        'pm25': {'v': 161}, 'so2': {'v': 7.6}, 't': {'v': 36.7},
        'w': {'v': 0.5}, 'wd': {'v': 329.7}, 'wg': {'v': 8.2}
    }
    # ... (keep the other test data examples: test_data_vpoor_pm10, test_data_clean, test_data_invalid etc.) ...
    # ... (keep the print statements and calls to interpret_pollutant_risks for these tests) ...
    print(f"--- Testing with Delhi-like data: {test_data_delhi} ---")
    risks_delhi = interpret_pollutant_risks(test_data_delhi)
    print("Identified Risks:")
    if risks_delhi:
        for risk in risks_delhi:
            print(f"- {risk}")
    else:
        print("- None")

    test_data_vpoor_pm10 = {
        'pm10': {'v': 360}, 'o3': {'v': 110}, 'no2': {'v': 50}
    }
    print(f"\n--- Testing with Very Poor PM10 data: {test_data_vpoor_pm10} ---")
    risks_vpoor_pm10 = interpret_pollutant_risks(test_data_vpoor_pm10)
    print("Identified Risks:")
    if risks_vpoor_pm10:
        for risk in risks_vpoor_pm10:
            print(f"- {risk}")
    else:
        print("- None")

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

    print("\n--- Testing with Invalid/Empty data ---")
    risks_invalid = interpret_pollutant_risks(None)
    print(f"Invalid data risks: {risks_invalid} (Expected: [])")
    risks_empty = interpret_pollutant_risks({})
    print(f"Empty data risks: {risks_empty} (Expected: [])")
    risks_bad_format = interpret_pollutant_risks({'pm25': 150}) # Incorrect format
    print(f"Bad format data risks: {risks_bad_format} (Expected: [])")

    print("\n" + "="*40)
    print(" Health Risk Interpreter Tests Finished ")
    print("="*40 + "\n")