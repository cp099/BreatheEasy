
# File: src/health_rules/interpreter.py

"""
Interprets health risks based on individual air pollutant concentrations.

This module defines health risk thresholds for various air pollutants, based
on the Indian CPCB NAQI standard. It provides a function to compare real-time
pollutant data against these thresholds to generate human-readable health
risk advisories.
"""

import logging 

log = logging.getLogger(__name__)

# --- Pollutant Health Risk Thresholds ---
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
    """
    Analyzes individual pollutant levels to identify the highest potential health risk for each.

    Compares pollutant values from the input dictionary against the predefined
    thresholds. For each pollutant present, it identifies the most severe risk
    level exceeded and generates a corresponding advisory.

    Args:
        iaqi_data (dict | None): A dictionary of pollutant data, typically from the
                                 AQICN API's 'iaqi' field.
                                 Example: {'pm25': {'v': 161}, 'o3': {'v': 45}}

    Returns:
        list[str]: A list of human-readable risk advisories for any pollutant
                   exceeding a defined threshold. The format is:
                   "{POLLUTANT} ({Severity}): {Risk Description}".
                   Returns an empty list if no thresholds are met or input is invalid.
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

# --- Example Usage / Direct Execution ---
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    print("\n" + "="*40)
    print(" Running interpreter.py Self-Test ")
    print("="*40 + "\n")

    # --- Test Cases ---
    test_cases = {
        "Clean Air": {'pm25': {'v': 20}, 'o3': {'v': 30}},
        "Moderate PM2.5": {'pm25': {'v': 75.5}, 'co': {'v': 1.0}},
        "Poor PM10 & Moderate O3": {'pm10': {'v': 255}, 'o3': {'v': 110}},
        "Very Poor Multiple": {'pm25': {'v': 130}, 'no2': {'v': 300}},
        "Severe SO2": {'so2': {'v': 1700}},
        "Missing Pollutant Value": {'pm25': {'w': 100}},
        "Invalid Pollutant Value": {'co': {'v': 'high'}},
        "Empty Input": {},
        "None Input": None,
    }

    for name, data in test_cases.items():
        print(f"--- Testing Case: {name} ---")
        print(f"Input Data: {data}")
        risks = interpret_pollutant_risks(data)
        if risks:
            print("  --> Identified Risks:")
            for risk in risks:
                print(f"    - {risk}")
        else:
            print("  --> No significant risks identified.")
        print("-" * 20)

    print("\n" + "="*40)
    print(" interpreter.py Self-Test Finished ")
    print("="*40 + "\n")