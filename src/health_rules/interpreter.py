# File: src/health_rules/interpreter.py

import logging

# --- Pollutant Thresholds and Associated Risks ---
# Define thresholds for key pollutants and the potential respiratory risks
# associated with exceeding them.
# NOTE: These are EXAMPLES. Thresholds and specific risk language should be
# verified against reliable sources (WHO, CPCB, EPA, medical guidelines).
# Concentrations are typically in µg/m³ (except CO usually ppm). AQICN API
# usually provides values that can be roughly compared to these scales.

POLLUTANT_HEALTH_THRESHOLDS = {
    "pm25": [ # PM2.5 (µg/m³) - Often linked to deeper lung/heart issues
        {"threshold": 150, "risk": "Increased risk of respiratory symptoms, aggravation of asthma or lung disease.", "severity": "High"},
        {"threshold": 100, "risk": "Potential breathing discomfort for sensitive groups (asthma, COPD).", "severity": "Moderate"},
        {"threshold": 55, "risk": "Possible minor breathing discomfort for very sensitive individuals.", "severity": "Low"} # Based roughly on Moderate AQI category start
    ],
    "pm10": [ # PM10 (µg/m³) - Larger particles, more upper airway irritation
        {"threshold": 250, "risk": "Increased likelihood of respiratory symptoms and aggravation of lung disease.", "severity": "High"},
        {"threshold": 150, "risk": "Potential breathing discomfort for sensitive groups.", "severity": "Moderate"},
        {"threshold": 100, "risk": "Possible aggravation of respiratory conditions for sensitive individuals.", "severity": "Low"} # Based roughly on Moderate AQI category start
     ],
    "o3": [ # Ozone (µg/m³ - Note: AQICN might use ppb, conversion may be needed if scales differ) - Irritant, triggers asthma
        {"threshold": 160, "risk": "Significant likelihood of breathing difficulty, asthma attacks, reduced lung function.", "severity": "High"},
        {"threshold": 100, "risk": "Breathing discomfort, potential asthma aggravation, especially during exercise.", "severity": "Moderate"},
    ],
    "no2": [ # Nitrogen Dioxide (µg/m³) - Airway inflammation, asthma aggravation
        {"threshold": 200, "risk": "Increased likelihood of respiratory symptoms, reduced lung function, asthma aggravation.", "severity": "High"},
        {"threshold": 100, "risk": "Potential for airway inflammation and asthma aggravation for sensitive groups.", "severity": "Moderate"},
    ],
    "so2": [ # Sulfur Dioxide (µg/m³) - Bronchoconstriction, asthma trigger
        {"threshold": 350, "risk": "Significant risk of bronchoconstriction (airway narrowing), asthma attacks.", "severity": "High"},
        {"threshold": 125, "risk": "Potential for bronchoconstriction, particularly for asthmatics during exercise.", "severity": "Moderate"},
    ],
    "co": [ # Carbon Monoxide (Value often in ppm, AQICN usually provides index-like value? Check API docs) - Reduces oxygen delivery
          # Thresholds here are highly dependent on units and exposure time.
          # Example based on general high levels, may need adjustment based on AQICN 'v' unit for CO.
        {"threshold": 10, "risk": "Potential aggravation of cardiovascular conditions, some effects on exertion.", "severity": "Moderate"}, # Assuming 'v' is index-like or ppm requires context
        {"threshold": 30, "risk": "Significant aggravation of cardiovascular symptoms, potential neurological effects.", "severity": "High"}, # Requires context
    ]
    # Add other pollutants if relevant and thresholds are known
}

def interpret_pollutant_risks(iaqi_data):
    """
    Analyzes individual pollutant levels from AQICN data and identifies potential
    respiratory health risks based on predefined thresholds.

    Args:
        iaqi_data (dict): The 'iaqi' part of the AQICN API response, containing
                          pollutant keys (e.g., 'pm25', 'o3') mapped to dictionaries
                          containing the value under the 'v' key (e.g., {'v': 150}).

    Returns:
        list: A list of strings, each describing a potential health risk identified
              based on the pollutant levels exceeding thresholds. Returns an empty
              list if no thresholds are exceeded or input is invalid.
              Example: ["PM2.5 High: Increased risk of respiratory symptoms...", "O3 Moderate: ..."]
    """
    triggered_risks = []
    if not iaqi_data or not isinstance(iaqi_data, dict):
        logging.warning("Invalid or empty iaqi_data received for interpretation.")
        return triggered_risks

    logging.info(f"Interpreting risks for iaqi data: {iaqi_data}")

    for pollutant, thresholds in POLLUTANT_HEALTH_THRESHOLDS.items():
        if pollutant in iaqi_data and isinstance(iaqi_data[pollutant], dict) and 'v' in iaqi_data[pollutant]:
            try:
                value = float(iaqi_data[pollutant]['v'])
                logging.debug(f"Checking {pollutant.upper()} with value {value}")

                # Check thresholds from highest to lowest, add only the highest risk triggered
                highest_risk_found = None
                for level_info in sorted(thresholds, key=lambda x: x['threshold'], reverse=True):
                    if value >= level_info["threshold"]:
                        highest_risk_found = f"{pollutant.upper()} ({level_info['severity']}): {level_info['risk']}"
                        logging.info(f"Threshold exceeded for {pollutant.upper()} at value {value} (>= {level_info['threshold']}). Risk: {level_info['risk']}")
                        break # Stop checking lower thresholds for this pollutant

                if highest_risk_found:
                    triggered_risks.append(highest_risk_found)

            except (ValueError, TypeError) as e:
                logging.warning(f"Could not parse value for pollutant '{pollutant}': {iaqi_data[pollutant].get('v')}. Error: {e}")
                continue # Skip to next pollutant if value is not numeric
        else:
            logging.debug(f"Pollutant '{pollutant}' not found in iaqi_data or format invalid.")

    if not triggered_risks:
        logging.info("No significant pollutant thresholds exceeded based on defined rules.")

    return triggered_risks

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Setup logger if running directly
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

    print("\n" + "="*30)
    print(" Testing Health Risk Interpreter ")
    print("="*30 + "\n")

    # Example 1: Data similar to the Delhi test output
    test_data_delhi = {
        'co': {'v': 1.2}, 'h': {'v': 27.3}, 'no2': {'v': 15.8},
        'o3': {'v': 41.2}, 'p': {'v': 983.2}, 'pm10': {'v': 178},
        'pm25': {'v': 161}, 'so2': {'v': 7.6}, 't': {'v': 36.7},
        'w': {'v': 0.5}, 'wd': {'v': 329.7}, 'wg': {'v': 8.2}
    }
    print(f"--- Testing with Delhi-like data: {test_data_delhi} ---")
    risks_delhi = interpret_pollutant_risks(test_data_delhi)
    print("Identified Risks:")
    if risks_delhi:
        for risk in risks_delhi:
            print(f"- {risk}")
    else:
        print("- None")

    # Example 2: Hypothetical data with high Ozone
    test_data_high_o3 = {
        'pm25': {'v': 40}, 'o3': {'v': 170}, 'no2': {'v': 50}
    }
    print(f"\n--- Testing with High Ozone data: {test_data_high_o3} ---")
    risks_high_o3 = interpret_pollutant_risks(test_data_high_o3)
    print("Identified Risks:")
    if risks_high_o3:
        for risk in risks_high_o3:
            print(f"- {risk}")
    else:
        print("- None")

    # Example 3: Clean air data
    test_data_clean = {
        'pm25': {'v': 10}, 'pm10': {'v': 20}, 'o3': {'v': 30}, 'no2': {'v': 15}
    }
    print(f"\n--- Testing with Clean Air data: {test_data_clean} ---")
    risks_clean = interpret_pollutant_risks(test_data_clean)
    print("Identified Risks:")
    if risks_clean:
        for risk in risks_clean:
            print(f"- {risk}")
    else:
        print("- None (Expected)")

    # Example 4: Invalid/Empty data
    print("\n--- Testing with Invalid/Empty data ---")
    risks_invalid = interpret_pollutant_risks(None)
    print(f"Invalid data risks: {risks_invalid} (Expected: [])")
    risks_empty = interpret_pollutant_risks({})
    print(f"Empty data risks: {risks_empty} (Expected: [])")
    risks_bad_format = interpret_pollutant_risks({'pm25': 150}) # Incorrect format
    print(f"Bad format data risks: {risks_bad_format} (Expected: [])")


    print("\n" + "="*30)
    print(" Health Risk Interpreter Tests Finished ")
    print("="*30 + "\n")