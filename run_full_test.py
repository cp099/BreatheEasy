# File: run_full_test.py (Updated for 3-day forecast)
"""
Runs an end-to-end test of the BreatheEasy backend components for a selected city.
Fetches live data and generates forecasts, displaying output in a text format
that mimics the intended UI sections. Hides INFO level console logs for cleaner output.
"""

import sys
import os
import logging
import pprint
import pandas as pd

# --- Setup Path ---
# (Keep existing path setup)
PROJECT_ROOT = os.path.abspath('.')
if 'src' not in os.listdir(PROJECT_ROOT):
     alt_root = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
     if 'src' in os.listdir(alt_root): PROJECT_ROOT = alt_root
     else: print(f"ERROR: Cannot determine project root from CWD: {os.path.abspath('.')}"); sys.exit(1)
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

# --- Imports ---
try:
    from src.config_loader import CONFIG
    from src.api_integration.weather_client import get_current_weather
    from src.analysis.historical import get_available_cities, get_city_aqi_trend_data
    from src.health_rules.info import AQI_DEFINITION, AQI_SCALE, get_aqi_info
    from src.api_integration.client import get_current_aqi_for_city, get_current_pollutant_risks_for_city
    from src.modeling.predictor import format_forecast_for_ui, get_predicted_weekly_risks, generate_forecast
    log = logging.getLogger(__name__)
except ImportError as e: print(f"ERROR: Could not import modules: {e}"); sys.exit(1)
except Exception as e: print(f"ERROR during import/setup: {e}"); sys.exit(1)

# --- Get Forecast Days from Config ---
# Use the value set in config.yaml (should be 3)
FORECAST_DAYS = CONFIG.get('modeling', {}).get('forecast_days', 3)

# --- Console Log Level Override ---
def suppress_console_info_logs():
    # (Keep existing function as is)
    try:
        root_logger = logging.getLogger(); found_console = False
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                 print(f"Updating level for Console Handler: {handler} to WARNING") # Keep this print for visibility
                 handler.setLevel(logging.WARNING); found_console = True; break
        if not found_console: print("WARNING: Could not find console handler.")
    except Exception as e: print(f"WARNING: Failed to adjust console log level: {e}")

# --- Main Display Function ---
def display_all_sections(city_name_internal, city_name_weather, city_name_aqicn, days_to_forecast=FORECAST_DAYS): # Pass days
    """Fetches and displays data for all UI sections for the given city."""
    print(f"\n{'='*25} Generating Report for: {city_name_internal.upper()} ({days_to_forecast}-Day Forecast) {'='*25}")

    # Section 0.5: Current Weather
    print("\n--- [ Section 0.5: Current Weather ] ---")
    try:
        weather_data = get_current_weather(city_name_weather)
        if weather_data:
            # (Keep display logic as is)
            print(f"  Location:     {weather_data.get('city', 'N/A')}, {weather_data.get('country', 'N/A')}")
            print(f"  Conditions:   {weather_data.get('condition_text', 'N/A')}")
            print(f"  Temperature:  {weather_data.get('temp_c', 'N/A')}°C (Feels like {weather_data.get('feelslike_c', 'N/A')}°C)")
            print(f"  Wind:         {weather_data.get('wind_kph', 'N/A')} kph from {weather_data.get('wind_dir', 'N/A')}")
            print(f"  Humidity:     {weather_data.get('humidity', 'N/A')}%")
            print(f"  Last Updated: {weather_data.get('last_updated', 'N/A')} (Local Time)")
        else: print("  Weather data currently unavailable.")
    except Exception as e: print(f"  ERROR retrieving weather data: {e}"); log.error(f"Error Sec 0.5: {e}", exc_info=True)

    # Section 1: Historical Summary
    print("\n--- [ Section 1: Historical AQI Graph ] ---")
    try:
        hist_data_check = get_city_aqi_trend_data(city_name_internal)
        if hist_data_check is not None and not hist_data_check.empty: print("  (Data available for historical graph)")
        else: print("  (Historical data not available)")
    except Exception as e: print(f"  ERROR checking historical data: {e}"); log.error(f"Error Sec 1: {e}", exc_info=True)

    # Section 2: Understanding AQI
    print("\n--- [ Section 2: Understanding AQI (CPCB Scale) ] ---")
    # (Keep display logic as is)
    print(f"  Definition: {AQI_DEFINITION.strip()}")
    print("  Scale & Implications:")
    for category in AQI_SCALE: print(f"    - {category['level']} ({category['range']}): {category['implications']}")

    # Section 3: Current City AQI
    print("\n--- [ Section 3: Current AQI ] ---")
    try:
        current_aqi = get_current_aqi_for_city(city_name_aqicn)
        if current_aqi:
             # (Keep display logic as is)
            aqi_val = current_aqi.get('aqi'); aqi_info = get_aqi_info(aqi_val) if aqi_val is not None else None
            level = f"({aqi_info['level']})" if aqi_info else "(Level Unknown)"; color = aqi_info['color'] if aqi_info else '#808080'
            print(f"  >>> Current AQI: {aqi_val} {level} [Color: {color}] <<<")
            print(f"  Reporting Station: {current_aqi.get('station', 'N/A')}")
            print(f"  Timestamp: {current_aqi.get('time', 'N/A')}")
        else: print("  Current AQI data currently unavailable.")
    except Exception as e: print(f"  ERROR retrieving current AQI data: {e}"); log.error(f"Error Sec 3: {e}", exc_info=True)

    # Section 4: AQI Forecast (Table/Dict) - UPDATED
    print(f"\n--- [ Section 4: AQI Forecast (Next {days_to_forecast} Days) ] ---") # Update title
    try:
        # Pass days_to_forecast to generate_forecast
        raw_forecast_df = generate_forecast(city_name_internal, days_ahead=days_to_forecast, apply_residual_correction=True)
        ui_forecast = format_forecast_for_ui(raw_forecast_df)
        if ui_forecast:
            print("  Date        | Predicted AQI | Level")
            print("  ------------|---------------|---------------")
            for day_forecast in ui_forecast:
                aqi_val = day_forecast['predicted_aqi']
                aqi_info = get_aqi_info(aqi_val) if aqi_val is not None else None
                level = aqi_info['level'] if aqi_info else "Unknown"
                print(f"  {day_forecast['date']} | {aqi_val:<13d} | {level}")
        else: print("  AQI forecast currently unavailable.")
    except Exception as e: print(f"  ERROR generating forecast data: {e}"); log.error(f"Error Sec 4: {e}", exc_info=True)

    # Section 5: Current Pollutant Risks (Text)
    print("\n--- [ Section 5: Current Pollutant Health Triggers ] ---")
    try:
        pollutant_risks = get_current_pollutant_risks_for_city(city_name_aqicn)
        if pollutant_risks:
            # (Keep display logic as is)
            print(f"  (Based on data from: {pollutant_risks.get('time', 'N/A')})")
            if pollutant_risks.get('risks'):
                print("  Potential Triggers:")
                for risk in pollutant_risks['risks']: print(f"    - {risk}")
            else: print("  - No specific pollutant risks detected.")
        else: print("  Current pollutant risk data currently unavailable.")
    except Exception as e: print(f"  ERROR retrieving current pollutant risks: {e}"); log.error(f"Error Sec 5: {e}", exc_info=True)

    # Section 6: Predicted Weekly Risks (Table/Dict) - UPDATED
    print(f"\n--- [ Section 6: Predicted Weekly Health Risks (Next {days_to_forecast} Days) ] ---") # Update title
    try:
        # Pass days_to_forecast to get_predicted_weekly_risks
        predicted_risks = get_predicted_weekly_risks(city_name_internal, days_ahead=days_to_forecast)
        if predicted_risks:
             print("  Date        | Pred. AQI | Level         | Implications")
             print("  ------------|-----------|---------------|--------------------------------------------------")
             for day_risk in predicted_risks:
                 print(f"  {day_risk['date']} | {day_risk['predicted_aqi']:<9d} | {day_risk['level']:<13} | {day_risk['implications']}")
        else: print("  Predicted weekly risk data currently unavailable.")
    except Exception as e: print(f"  ERROR generating predicted weekly risks: {e}"); log.error(f"Error Sec 6: {e}", exc_info=True)

    print(f"\n{'='*25} END REPORT FOR: {city_name_internal.upper()} {'='*25}\n")


# --- Main Execution Logic ---
if __name__ == "__main__":
    suppress_console_info_logs() # Call function to adjust logging level

    log.info("Starting end-to-end backend test script...") # Will only show in file log

    # Get available cities list
    log.info("Attempting to get available cities...")
    available_cities = get_available_cities()
    if not available_cities:
        log.warning("Could not get cities from historical data, falling back to config.")
        available_cities = CONFIG.get('modeling', {}).get('target_cities', [])
        if not available_cities:
             print("ERROR: No available cities found. Exiting.")
             log.critical("No available cities found. Exiting.")
             sys.exit(1)
        else: log.info(f"Using cities from config: {available_cities}")
    else: log.info(f"Using cities from historical data: {available_cities}")

    print("\nAvailable cities:")
    for i, city in enumerate(available_cities): print(f"  {i+1}. {city}")

    while True:
        try:
            choice = input(f"Enter the number of the city (1-{len(available_cities)}) or city name: ")
            selected_city_internal = None
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_cities): selected_city_internal = available_cities[choice_num - 1]
                else: print("Invalid number selection.")
            except ValueError:
                for city in available_cities:
                    if choice.strip().lower() == city.lower(): selected_city_internal = city; break
                if not selected_city_internal and ', ' in choice:
                     base_city_maybe = choice.strip().split(', ')[0]
                     if base_city_maybe in available_cities: selected_city_internal = base_city_maybe; print(f"(Interpreted '{choice}' as base city '{selected_city_internal}')")

            if selected_city_internal:
                 weather_api_query_city = selected_city_internal
                 aqicn_api_query_city = selected_city_internal
                 if selected_city_internal in CONFIG.get('modeling', {}).get('target_cities', []):
                      weather_api_query_city = f"{selected_city_internal}, India"
                 log.warning(f"Selected internal city: '{selected_city_internal}'") # Use warning to show on console
                 log.warning(f"Using query name for WeatherAPI: '{weather_api_query_city}'")
                 log.warning(f"Using query name for AQICN: '{aqicn_api_query_city}'")

                 # Pass the configured FORECAST_DAYS to the display function
                 display_all_sections(
                     city_name_internal=selected_city_internal,
                     city_name_weather=weather_api_query_city,
                     city_name_aqicn=aqicn_api_query_city,
                     days_to_forecast=FORECAST_DAYS # Use variable from config
                 )
                 break
            else: print(f"City '{choice}' not found or invalid selection. Please try again.")
        except KeyboardInterrupt: print("\nExiting."); break
        except Exception as e: print(f"\nAn unexpected error occurred: {e}"); log.error("Error in main loop", exc_info=True); break