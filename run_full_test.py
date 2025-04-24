# File: run_full_test.py
"""
Runs an end-to-end test of the BreatheEasy backend components for a selected city.
Fetches live data and generates forecasts, displaying output in a text format
that mimics the intended UI sections.
"""

import sys
import os
import logging
import pprint
import pandas as pd # Needed for formatting/display

# --- Setup Path ---
# Add project root to path to allow importing project modules
PROJECT_ROOT = os.path.abspath('.') # Assume running from project root
if 'src' not in os.listdir(PROJECT_ROOT): # Basic check
     alt_root = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
     if 'src' in os.listdir(alt_root):
          PROJECT_ROOT = alt_root
     else:
          print(f"ERROR: Cannot determine project root containing 'src' from CWD: {os.path.abspath('.')}")
          sys.exit(1)
if PROJECT_ROOT not in sys.path:
     sys.path.insert(0, PROJECT_ROOT)

# --- Imports (after path setup) ---
# Import CONFIG first to set up logging
try:
    from src.config_loader import CONFIG # Triggers logging setup
    from src.api_integration.weather_client import get_current_weather
    from src.analysis.historical import get_available_cities, get_city_aqi_trend_data
    from src.health_rules.info import AQI_DEFINITION, AQI_SCALE, get_aqi_info
    from src.api_integration.client import get_current_aqi_for_city, get_current_pollutant_risks_for_city
    from src.modeling.predictor import format_forecast_for_ui, get_predicted_weekly_risks, generate_forecast # Need generate_forecast for Sec 4 formatting
    log = logging.getLogger(__name__) # Get logger after config_loader ran
except ImportError as e:
    print(f"ERROR: Could not import necessary modules. Check setup and ensure venv is active. Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR during import or logging setup: {e}")
    sys.exit(1)

# --- Main Display Function ---
def display_all_sections(city_name_internal, city_name_weather, city_name_aqicn):
    """Fetches and displays data for all UI sections for the given city."""

    print(f"\n{'='*25} Generating Report for: {city_name_internal.upper()} {'='*25}")

    # --- Section 0.5: Current Weather (Text) ---
    print("\n--- [ Section 0.5: Current Weather ] ---")
    try:
        weather_data = get_current_weather(city_name_weather)
        if weather_data:
            print(f"  Location:     {weather_data.get('city', 'N/A')}, {weather_data.get('country', 'N/A')}")
            print(f"  Conditions:   {weather_data.get('condition_text', 'N/A')}")
            print(f"  Temperature:  {weather_data.get('temp_c', 'N/A')}°C (Feels like {weather_data.get('feelslike_c', 'N/A')}°C)")
            print(f"  Wind:         {weather_data.get('wind_kph', 'N/A')} kph from {weather_data.get('wind_dir', 'N/A')}")
            print(f"  Humidity:     {weather_data.get('humidity', 'N/A')}%")
            print(f"  Last Updated: {weather_data.get('last_updated', 'N/A')} (Local Time)")
        else:
            print("  Weather data currently unavailable for this location.")
    except Exception as e:
        print(f"  ERROR retrieving weather data: {e}")
        log.error(f"Error in Sec 0.5 for {city_name_internal}", exc_info=True)

    # --- Section 1: Historical Summary (Graph Placeholder) ---
    print("\n--- [ Section 1: Historical AQI Graph ] ---")
    try:
        hist_data_check = get_city_aqi_trend_data(city_name_internal)
        if hist_data_check is not None and not hist_data_check.empty:
            print(f"  (Data available for historical graph - {len(hist_data_check)} points from {hist_data_check.index.min().date()} to {hist_data_check.index.max().date()})")
        else:
            print("  (Historical data not available or city mismatch for graph)") # Updated message
    except Exception as e:
         print(f"  ERROR checking historical data: {e}")
         log.error(f"Error in Sec 1 for {city_name_internal}", exc_info=True)

    # --- Section 2: Understanding AQI (Text) ---
    print("\n--- [ Section 2: Understanding AQI (CPCB Scale) ] ---")
    print(f"  Definition: {AQI_DEFINITION.strip()}")
    print("  Scale & Implications:")
    for category in AQI_SCALE:
        print(f"    - {category['level']} ({category['range']}): {category['implications']}")

    # --- Section 3: Current City AQI (Text / Visual) ---
    print("\n--- [ Section 3: Current AQI ] ---")
    try:
        current_aqi = get_current_aqi_for_city(city_name_aqicn)
        if current_aqi:
            aqi_val = current_aqi.get('aqi')
            aqi_info = get_aqi_info(aqi_val) if aqi_val is not None else None
            level = f"({aqi_info['level']})" if aqi_info else "(Level Unknown)"
            color = aqi_info['color'] if aqi_info else '#808080' # Grey if unknown
            print(f"  >>> Current AQI: {aqi_val} {level} [Color: {color}] <<<")
            print(f"  Reporting Station: {current_aqi.get('station', 'N/A')}")
            print(f"  Timestamp: {current_aqi.get('time', 'N/A')}")
        else:
            print("  Current AQI data currently unavailable for this location.")
    except Exception as e:
        print(f"  ERROR retrieving current AQI data: {e}")
        log.error(f"Error in Sec 3 for {city_name_internal}", exc_info=True)


    # --- Section 4: AQI Forecast (Table/Dict) ---
    print("\n--- [ Section 4: AQI Forecast (Next 5 Days) ] ---")
    try:
        # Use generate_forecast and format_forecast_for_ui
        raw_forecast_df = generate_forecast(city_name_internal, days_ahead=5, apply_residual_correction=True)
        ui_forecast = format_forecast_for_ui(raw_forecast_df)
        if ui_forecast:
            print("  Date        | Predicted AQI | Level")
            print("  ------------|---------------|---------------")
            for day_forecast in ui_forecast:
                aqi_val = day_forecast['predicted_aqi']
                aqi_info = get_aqi_info(aqi_val) if aqi_val is not None else None
                level = aqi_info['level'] if aqi_info else "Unknown"
                print(f"  {day_forecast['date']} | {aqi_val:<13d} | {level}")
        else:
            print("  AQI forecast currently unavailable.")
    except Exception as e:
        print(f"  ERROR generating forecast data: {e}")
        log.error(f"Error in Sec 4 for {city_name_internal}", exc_info=True)


    # --- Section 5: Current Pollutant Risks (Text) ---
    print("\n--- [ Section 5: Current Pollutant Health Triggers ] ---")
    try:
        pollutant_risks = get_current_pollutant_risks_for_city(city_name_aqicn)
        if pollutant_risks:
            print(f"  (Based on data from: {pollutant_risks.get('time', 'N/A')})")
            if pollutant_risks.get('risks'):
                print("  Potential Triggers:")
                for risk in pollutant_risks['risks']:
                    print(f"    - {risk}")
            else:
                print("  - No specific pollutant risks detected at current levels based on thresholds.")
            # print("\n  Current Pollutant Levels (Raw):") # Optional Raw Data
            # if pollutant_risks.get('pollutants'):
            #     pprint.pprint(pollutant_risks['pollutants'], indent=4)
        else:
            print("  Current pollutant risk data currently unavailable.")
    except Exception as e:
         print(f"  ERROR retrieving current pollutant risks: {e}")
         log.error(f"Error in Sec 5 for {city_name_internal}", exc_info=True)


    # --- Section 6: Predicted Weekly Risks (Table/Dict) ---
    print("\n--- [ Section 6: Predicted Weekly Health Risks ] ---")
    try:
        predicted_risks = get_predicted_weekly_risks(city_name_internal, days_ahead=5)
        if predicted_risks:
             print("  Date        | Pred. AQI | Level         | Implications")
             print("  ------------|-----------|---------------|--------------------------------------------------")
             for day_risk in predicted_risks:
                 print(f"  {day_risk['date']} | {day_risk['predicted_aqi']:<9d} | {day_risk['level']:<13} | {day_risk['implications']}")
        else:
            print("  Predicted weekly risk data currently unavailable.")
    except Exception as e:
         print(f"  ERROR generating predicted weekly risks: {e}")
         log.error(f"Error in Sec 6 for {city_name_internal}", exc_info=True)


    print(f"\n{'='*25} END REPORT FOR: {city_name_internal.upper()} {'='*25}\n")


# --- Main Execution Logic ---
if __name__ == "__main__":
    log.info("Starting end-to-end backend test script...")

    # --- Get available cities list FIRST ---
    log.info("Attempting to get available cities...")
    available_cities = get_available_cities() # Call the function from historical.py
    if not available_cities:
        log.warning("Could not get cities from historical data, falling back to config.")
        available_cities = CONFIG.get('modeling', {}).get('target_cities', [])
        if not available_cities:
             print("ERROR: No available cities found in historical data or config. Exiting.")
             log.critical("No available cities found. Exiting test script.")
             sys.exit(1)
        else:
             log.info(f"Using cities from config: {available_cities}")
    else:
         log.info(f"Using cities from historical data: {available_cities}")

    # --- Now display choices and get input ---
    print("\nAvailable cities:")
    for i, city in enumerate(available_cities):
        print(f"  {i+1}. {city}")

    while True:
        try:
            choice = input(f"Enter the number of the city (1-{len(available_cities)}) or city name: ")
            selected_city_internal = None # For historical data and model files

            # Try interpreting as a number first
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_cities):
                    selected_city_internal = available_cities[choice_num - 1]
                else:
                    print("Invalid number selection.")
            except ValueError:
                # If not a number, treat as a city name (case-insensitive check)
                for city in available_cities:
                    if choice.strip().lower() == city.lower():
                         selected_city_internal = city
                         break
                # Handle potential direct input of clarified name like "Delhi, India"
                # This will only work if the base city name is in our list
                if not selected_city_internal and ', ' in choice:
                     base_city_maybe = choice.strip().split(', ')[0]
                     if base_city_maybe in available_cities:
                          selected_city_internal = base_city_maybe
                          print(f"(Interpreted '{choice}' as base city '{selected_city_internal}')")

            # If a valid city from our list was selected/derived
            if selected_city_internal:
                 # Determine query names for APIs
                 weather_api_query_city = selected_city_internal # Default
                 aqicn_api_query_city = selected_city_internal   # Default

                 # Clarify for specific APIs if needed (based on previous findings)
                 if selected_city_internal in ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Hyderabad']:
                      weather_api_query_city = f"{selected_city_internal}, India"
                      # AQICN API preferred the simple name based on earlier tests
                      # aqicn_api_query_city = selected_city_internal # Already the default

                 log.info(f"Selected internal city: '{selected_city_internal}'")
                 log.info(f"Using query name for WeatherAPI: '{weather_api_query_city}'")
                 log.info(f"Using query name for AQICN: '{aqicn_api_query_city}'")

                 # Call the main display function using appropriate names
                 display_all_sections(
                     city_name_internal=selected_city_internal,
                     city_name_weather=weather_api_query_city,
                     city_name_aqicn=aqicn_api_query_city
                 )
                 break # Exit loop after successful run
            else:
                 print(f"City '{choice}' not found or invalid selection from list: {available_cities}. Please try again.")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred in the main loop: {e}")
            log.error("Error in main execution loop of run_full_test.py", exc_info=True)
            break # Exit loop on unexpected error