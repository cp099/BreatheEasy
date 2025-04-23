# File: src/health_rules/interpreter.py

import logging # Standard logging import
# import pandas as pd # Not strictly needed in this module's functions anymore

# --- Get Logger ---
# Get the logger instance for this module. Inherits root config.
log = logging.getLogger(__name__)
# NOTE: CONFIG is not needed here unless thresholds move to config later

# --- Pollutant Thresholds and Associated Risks (Derived from CPCB NAQI Breakpoints) ---
# (Keep POLLUTANT_HEALTH_THRESHOLDS dictionary exactly as it was)
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
    """
    # (Function logic remains exactly the same)
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

# --- Example Usage Block (No change needed here) ---
if __name__ == "__main__":
    # Logging configured when config_loader is imported by any module run
    # (Keep the existing test block code exactly as it was)
    print("\n" + "="*40)
    print(" Testing Health Risk Interpreter (CPCB Based) - Current Pollutants")
    print("="*40 + "\n")
    test_data_delhi = {'co': {'v': 1.2}, 'h': {'v': 27.3}, 'no2': {'v': 15.8}, 'o3': {'v': 41.2}, 'p': {'v': 983.2}, 'pm10': {'v': 178}, 'pm25': {'v': 161}, 'so2': {'v': 7.6}, 't': {'v': 36.7}, 'w': {'v': 0.5}, 'wd': {'v': 329.7}, 'wg': {'v': 8.2}}
    print(f"--- Testing with Delhi-like data: {test_data_delhi} ---")
    risks_delhi = interpret_pollutant_risks(test_data_delhi)
    print("Identified Risks:")
    if risks_delhi:
        for risk in risks_delhi: print(f"- {risk}")
    else: print("- None")
    test_data_vpoor_pm10 = {'pm10': {'v': 360}, 'o3': {'v': 110}, 'no2': {'v': 50}}
    print(f"\n--- Testing with Very Poor PM10 data: {test_data_vpoor_pm10} ---")
    risks_vpoor_pm10 = interpret_pollutant_risks(test_data_vpoor_pm10)
    print("Identified Risks:")
    if risks_vpoor_pm10:
        for risk in risks_vpoor_pm10: print(f"- {risk}")
    else: print("- None")
    test_data_clean = {'pm25': {'v': 10}, 'pm10': {'v': 20}, 'o3': {'v': 30}, 'no2': {'v': 15}, 'co': {'v': 0.5}}
    print(f"\n--- Testing with Clean Air data: {test_data_clean} ---")
    risks_clean = interpret_pollutant_risks(test_data_clean)
    print("Identified Risks:")
    if risks_clean:
        for risk in risks_clean: print(f"- {risk}")
    else: print("- None (Expected)")
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