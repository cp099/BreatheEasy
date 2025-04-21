# File: src/api_integration/client.py

import requests
import os
import logging
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file in the project root
# Assumes .env is in the directory two levels up from this file (src/api_integration -> src -> BREATHEEASY)
try:
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
    logging.info(f"Attempted to load .env file from: {dotenv_path}")
except Exception as e:
    logging.error(f"Error finding or loading .env file: {e}", exc_info=True)
    # Handle appropriately - maybe raise error or try default location load_dotenv()


# Get API token from environment variable
AQICN_TOKEN = os.getenv('AQICN_API_TOKEN')
AQICN_BASE_URL = "https://api.waqi.info/feed"

# Setup logger for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# --- API Client Function ---

def get_city_aqi_data(city_name):
    """
    Fetches real-time Air Quality Index and pollutant data for a specific city
    from the World Air Quality Index project API (aqicn.org).

    Args:
        city_name (str): The name of the city (e.g., 'Delhi', 'Mumbai', 'London').
                         The API is quite flexible with names but exact station IDs are best if known.

    Returns:
        dict: A dictionary containing the fetched data if successful and status is 'ok'.
              Includes overall AQI, station info, time, and individual pollutant data (iaqi).
              Example keys: 'status', 'data' (which contains 'aqi', 'city', 'time', 'iaqi', etc.)
        None: If the API key is missing, the request fails, the city is not found,
              or the API response status is not 'ok'.
    """
    if not AQICN_TOKEN:
        logging.error("AQICN_API_TOKEN not found in environment variables. Check .env file.")
        return None

    # Construct the API request URL
    # Format: https://api.waqi.info/feed/{city}/?token={your_token}
    api_url = f"{AQICN_BASE_URL}/{city_name}/?token={AQICN_TOKEN}"
    logging.info(f"Requesting data from AQICN API: {AQICN_BASE_URL}/{city_name}/?token=***TOKEN_HIDDEN***") # Hide token in logs

    try:
        response = requests.get(api_url, timeout=10) # Add timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # Parse the JSON response
        data = response.json()
        logging.debug(f"Raw API response JSON: {data}") # Log raw data only if debugging

        # --- Check API Response Status ---
        # The AQICN API includes a 'status' field ("ok", "error")
        if data.get("status") == "ok":
            logging.info(f"Successfully received 'ok' status from AQICN API for city '{city_name}'.")
            # The actual data is usually nested under the 'data' key
            return data # Return the full parsed JSON response which includes the 'data' field
        elif data.get("status") == "error":
            error_message = data.get("data", "Unknown error reason") # Error message is often in 'data' field on error
            logging.error(f"AQICN API returned error for city '{city_name}': {error_message}")
            # Specific handling for common errors
            if "Unknown station" in str(error_message):
                 logging.warning(f"City/Station '{city_name}' not found by AQICN API.")
            return None
        else:
            # Handle cases where 'status' key might be missing or has unexpected value
            logging.error(f"Received unexpected status from AQICN API for city '{city_name}': {data.get('status', 'Status field missing')}")
            return None

    except requests.exceptions.Timeout:
        logging.error(f"Request to AQICN API timed out for city '{city_name}'.")
        return None
    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP errors (like 404 Not Found, 401 Unauthorized, 500 Server Error)
        logging.error(f"HTTP error occurred: {http_err} - Response Status: {http_err.response.status_code}")
        if http_err.response.status_code == 401:
             logging.error("Authorization Error (401): Check if AQICN_API_TOKEN is correct and valid.")
        return None
    except requests.exceptions.RequestException as req_err:
        # Handle other network-related errors (DNS failure, connection error, etc.)
        logging.error(f"Error during request to AQICN API: {req_err}")
        return None
    except ValueError as json_err:
        # Handle errors during JSON decoding (if response is not valid JSON)
        logging.error(f"Error decoding JSON response from AQICN API: {json_err}")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"An unexpected error occurred in get_city_aqi_data: {e}", exc_info=True)
        return None
# (Keep existing imports, config, CITY_STATION_MAP if you added it - though we won't use it now, and get_city_aqi_data function)


# --- Function specifically for Section 3 ---

def get_current_aqi_for_city(city_name):
    """
    Fetches the current overall AQI value for a specific city.
    This function is tailored for Section 3 of the application.

    Args:
        city_name (str): The name of the city (e.g., 'Delhi').

    Returns:
        dict: A dictionary containing the city name, current AQI, and timestamp,
              e.g., {'city': 'Delhi', 'aqi': 178, 'time': '2025-04-21 12:00:00'}
        None: If data cannot be fetched, the API response is invalid,
              or the AQI value is missing.
    """
    logging.info(f"Getting current AQI value specifically for Section 3 for city: {city_name}")

    # Call the main data fetching function (using city name query)
    full_data = get_city_aqi_data(city_name) # Uses city name as query target now

    if full_data and full_data.get("status") == "ok" and "data" in full_data:
        api_data = full_data["data"]
        try:
            # Extract the necessary fields
            aqi = int(api_data.get("aqi", None)) # Ensure AQI is integer if possible
            station_name = api_data.get("city", {}).get("name", city_name) # Get specific station if available
            timestamp = api_data.get("time", {}).get("s", None)

            if aqi is not None and timestamp is not None:
                result = {
                    "city": city_name, # Return the requested city name
                    "aqi": aqi,
                    "station": station_name, # Include the station providing the data
                    "time": timestamp
                }
                logging.info(f"Successfully extracted current AQI info for {city_name}: {result}")
                return result
            else:
                logging.error(f"Could not extract 'aqi' or 'time' from API data for {city_name}. Data: {api_data}")
                return None

        except (ValueError, TypeError) as e:
             logging.error(f"Error processing AQI data for {city_name}: {e}. Data: {api_data}", exc_info=True)
             return None
        except Exception as e: # Catch any other unexpected errors during extraction
             logging.error(f"Unexpected error extracting AQI data for {city_name}: {e}", exc_info=True)
             return None
    else:
        logging.warning(f"Failed to get valid API response for {city_name} to extract current AQI.")
        return None

# --- Example Usage Block (for testing the module directly) ---
if __name__ == "__main__":
    print("\n" + "="*30)
    print(" Running api_integration/client.py Tests ")
    print("="*30 + "\n")

    # --- Test 1: Fetching FULL data for a valid city ---
    test_city_full = "Delhi"
    print(f"[Test 1: Fetching FULL data for '{test_city_full}']")
    city_data_full = get_city_aqi_data(test_city_full)
    if city_data_full and city_data_full.get("status") == "ok":
        print(f"Success! Received FULL data for '{test_city_full}'.")
        # Optional: print parts of the full data if needed for debug
        # print(city_data_full.get("data", {}).get("iaqi", {}))
    else:
        print(f"Failure! Could not retrieve valid FULL data for '{test_city_full}'.")


    # --- Test 2: Fetching ONLY CURRENT AQI for Section 3 ---
    test_city_sec3 = "Mumbai" # Test a different city
    print(f"\n[Test 2: Fetching CURRENT AQI (Section 3) for '{test_city_sec3}']")
    current_aqi_info = get_current_aqi_for_city(test_city_sec3)
    if current_aqi_info:
         print(f"Success! Received Current AQI Info for {test_city_sec3}:")
         print(f"  AQI: {current_aqi_info.get('aqi')}")
         print(f"  Station: {current_aqi_info.get('station')}")
         print(f"  Timestamp: {current_aqi_info.get('time')}")
    else:
         print(f"Failure! Could not retrieve current AQI info for {test_city_sec3}.")


    # --- Test 3: Fetching data for an invalid city ---
    print("\n[Test 3: Fetching data for 'Atlantis']")
    invalid_city_data = get_city_aqi_data("Atlantis")
    if invalid_city_data is None:
        print("Success! Correctly handled non-existent city 'Atlantis' for full data (returned None).")
    else:
        print("Failure! Expected None for 'Atlantis' but received some data.")

    invalid_aqi_info = get_current_aqi_for_city("Atlantis")
    if invalid_aqi_info is None:
         print("Success! Correctly handled non-existent city 'Atlantis' for current AQI (returned None).")
    else:
         print("Failure! Expected None for 'Atlantis' current AQI but received data.")

    # --- Test 4: Check for API Key Missing (Manual Simulation Needed) ---
    print("\n[Test 4: Check for API Key Missing (Manual Simulation Needed)]")
    print("(Test requires manual modification or checking logs for 'AQICN_API_TOKEN not found')")


    print("\n" + "="*30)
    print(" api_integration/client.py Tests Finished ")
    print("="*30 + "\n")