# File: src/health_rules/interpreter.py

"""
Interprets potential health risks based on current pollutant concentration levels.

This module defines thresholds for various air pollutants (based on CPCB NAQI
breakpoints) and provides a function to compare real-time pollutant data
(typically from an API) against these thresholds to generate human-readable
health risk warnings.
"""

import logging # Standard logging import

# --- Get Logger ---
log = logging.getLogger(__name__)
# Note: No direct dependency on CONFIG needed here.

# --- Pollutant Thresholds and Associated Risks (Derived from CPCB NAQI Breakpoints) ---
# Defines thresholds based on the *start* of CPCB concentration ranges for different
# AQI categories (Moderate, Poor, Very Poor, Severe). The associated risks use language
# adapted from the health impacts described for those *overall AQI categories*.
# (Keep POLLUTANT_HEALTH_THRESHOLDS dictionary exactly as it was)
POLLUTANT_HEALTH_THRESHOLDS = {
    "pm25": [ # PM2.5 (µg/m³)
        {"threshold": 251, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"},
        {"threshold": 121, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 91,  "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 61,  "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
    ],
    "pm10": [ # PM10 (µg/m³)
        {"threshold": 431, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"},
        {"threshold": 351, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 251, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 101, "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
    ],
    "o3": [ # Ozone (µg/m³)
        {"threshold": 749, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"},
        {"threshold": 209, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 169, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 101, "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
    ],
    "no2": [ # Nitrogen Dioxide (µg/m³)
        {"threshold": 401, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"},
        {"threshold": 281, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 181, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 81,  "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
    ],
    "so2": [ # Sulfur Dioxide (µg/m³)
        {"threshold": 1601, "risk": "Serious respiratory impact on healthy people. Serious aggravation of heart or lung disease.", "severity": "Severe"},
        {"threshold": 801, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 381, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 81,  "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
    ],
    "co": [ # Carbon Monoxide (mg/m³)
        {"threshold": 34.1, "risk": "Serious aggravation of heart or lung disease; may cause respiratory effects even during light activity.", "severity": "Severe"},
        {"threshold": 17.1, "risk": "Respiratory illness on prolonged exposure. Effect may be pronounced in people with heart/lung diseases.", "severity": "Very Poor"},
        {"threshold": 10.1, "risk": "Breathing discomfort to people on prolonged exposure, and discomfort to people with heart disease.", "severity": "Poor"},
        {"threshold": 2.1,  "risk": "Breathing discomfort to people with lung disease (e.g., asthma) and heart disease, children, older adults.", "severity": "Moderate"},
    ]
}

def interpret_pollutant_risks(iaqi_data):
    """Analyzes individual pollutant levels and identifies potential health risks.

    Compares pollutant values from the input dictionary (expected to be in the
    format provided by the AQICN API's 'iaqi' field) against predefined
    thresholds based on CPCB NAQI breakpoints. Returns a list of human-readable
    warnings for pollutants exceeding their respective thresholds. Only the
    warning corresponding to the highest severity threshold exceeded for each
    pollutant is included.

    Args:
        iaqi_data (dict or None): A dictionary where keys are pollutant codes
                                  (e.g., 'pm25', 'o3') and values are dictionaries
                                  containing at least a 'v' key with the numerical
                                  pollutant reading (e.g., {'v': 161}).
                                  Handles None or empty dict as input.

    Returns:
        list[str]: A list of strings, each describing a potential health risk.
                   The format is typically "{POLLUTANT} ({Severity}): {Risk Description}".
                   Returns an empty list if no thresholds are met or if input is invalid.
    """
    # (Function code remains the same)
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

# --- Example Usage Block ---
# (Keep existing __main__ block as is)
if __name__ == "__main__":
    # ... (test code remains the same) ...
    pass # Added pass for valid syntax if test code removed/commented