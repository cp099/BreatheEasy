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


# --- Example Usage Block (for testing the module directly) ---
if __name__ == "__main__":
    print("\n" + "="*30)
    print(" Running api_integration/client.py Tests ")
    print("="*30 + "\n")

    # --- Test with a valid city ---
    test_city = "Delhi" # Or try 'Mumbai', 'Bangalore', etc.
    print(f"[Test 1: Fetching data for '{test_city}']")
    city_data = get_city_aqi_data(test_city)

    if city_data and city_data.get("status") == "ok":
        print(f"Success! Received data for '{test_city}'.")
        # Extract and display key information
        aqi = city_data.get("data", {}).get("aqi", "N/A")
        station = city_data.get("data", {}).get("city", {}).get("name", "N/A")
        time = city_data.get("data", {}).get("time", {}).get("s", "N/A")
        pollutants = city_data.get("data", {}).get("iaqi", {}) # Individual pollutant data

        print(f"  Station: {station}")
        print(f"  Overall AQI: {aqi}")
        print(f"  Timestamp: {time}")
        print("  Pollutant Values (iaqi):")
        if pollutants:
            for pollutant, details in pollutants.items():
                print(f"    - {pollutant.upper()}: {details.get('v', 'N/A')}")
        else:
            print("    - No individual pollutant data found.")
    else:
        print(f"Failure! Could not retrieve valid data for '{test_city}'. Check logs for details.")


    # --- Test with a potentially invalid city ---
    print("\n[Test 2: Fetching data for 'Atlantis']")
    invalid_city_data = get_city_aqi_data("Atlantis")
    if invalid_city_data is None:
        print("Success! Correctly handled non-existent city 'Atlantis' (returned None).")
    else:
        print("Failure! Expected None for 'Atlantis' but received some data.")

    # --- Test without API Key (if possible to simulate) ---
    # You could temporarily rename your .env file and run this test,
    # but remember to rename it back afterwards. Or comment out the token loading temporarily.
    print("\n[Test 3: Check for API Key Missing (Manual Simulation Needed)]")
    # AQICN_TOKEN = None # Temporarily unset for testing - uncomment carefully
    # missing_key_data = get_city_aqi_data("Delhi")
    # if missing_key_data is None:
    #     print("Success! Correctly handled missing API key (returned None).")
    # else:
    #     print("Failure! Expected None when API key is missing.")
    # Remember to restore AQICN_TOKEN loading if you modify it above.
    print("(Test requires manual modification or checking logs for 'AQICN_API_TOKEN not found')")


    print("\n" + "="*30)
    print(" api_integration/client.py Tests Finished ")
    print("="*30 + "\n")