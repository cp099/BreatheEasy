# BreatheEasy - AQI Analysis and Forecasting Backend

## Overview

BreatheEasy provides the backend logic for a system designed to analyze historical Air Quality Index (AQI) data, fetch real-time AQI and weather information, provide insights into associated health risks based on the Indian CPCB standards, and predict future AQI levels using weather-enhanced models for major Indian cities (Delhi, Mumbai, Bangalore, Chennai, Hyderabad).

The system interprets both current pollutant levels and predicted AQI values to indicate potential respiratory health implications, aiming to empower users with actionable air quality information. This repository contains the complete backend Python code, including data processing modules, API clients, configuration management, exception handling, automated tests, ML model training scripts (using Prophet with weather regressors), prediction logic, and health rule interpreters.

## Problem Statement

Air quality index (AQI) data is widely available, but translating numerical values into clear health risks remains a challenge for many users, especially those with respiratory conditions. Furthermore, anticipating changes in air quality requires reliable forecasting that accounts for meteorological factors. BreatheEasy aims to bridge this gap by providing accessible interpretations of current and future air quality based on established standards and predictive modeling incorporating weather data.

## Core Backend Features Implemented

*(Mapping to original UI sections)*

*   **(Sec 0.5) Current Weather:** Fetches real-time weather conditions (temp, description, icon, etc.) using the WeatherAPI.com service (`src/api_integration/weather_client.py`). Also includes a function to fetch **3-day weather forecasts**.
*   **(Sec 1) Historical Summary:** Provides functions to load historical data (AQI + Weather up to early 2024) and retrieve AQI trends and distributions per city (`src/analysis/historical.py`).
*   **(Sec 2) Educational Info:** Contains the definition of AQI and the CPCB India AQI scale with associated health implications (`src/health_rules/info.py`). Also includes pollutant-specific health risk thresholds (`src/health_rules/interpreter.py`).
*   **(Sec 3) Current AQI:** Fetches real-time AQI from city-level queries via the AQICN API (`src/api_integration/client.py`).
*   **(Sec 4) AQI Forecast:** Trains city-specific Prophet models incorporating **weather regressors** (`src/modeling/train.py`) and provides functions to generate a **3-day AQI forecast**, adjusted using the latest live AQI and decaying residual correction (`src/modeling/predictor.py`). Includes UI-friendly formatting.
*   **(Sec 5) Current Pollutants & Risks:** Fetches real-time pollutant data via AQICN API (`client.py`) and interprets immediate health risks based on CPCB-derived thresholds (`interpreter.py`).
*   **(Sec 6) Predicted Risks:** Takes the 3-day AQI forecast and interprets the potential health implications for each day based on the predicted AQI level and CPCB categories (`predictor.py` calls functions in `info.py`).

## Technology Stack

*   **Language:** Python 3.x
*   **Core Libraries:**
    *   Pandas: Data manipulation and analysis.
    *   Prophet (by Meta): Time series forecasting with regressors.
    *   Requests: HTTP requests for external APIs.
    *   python-dotenv: Managing environment variables (API keys).
    *   scikit-learn: Evaluation metrics (MAE, RMSE, MAPE).
    *   CmdStanPy: Backend for Prophet model fitting.
    *   PyYAML: Parsing YAML configuration files.
    *   pytest, pytest-mock: Automated testing and mocking.
*   **Data Sources:**
    *   Historical AQI & Weather: `Master_AQI_Dataset.csv` (user provided, expected to contain daily AQI, pollutants, and daily summarized weather features up to approx. Feb 2024).
    *   Real-time AQI/Pollutants: [World Air Quality Index Project (aqicn.org)](https://aqicn.org/api/)
    *   Real-time Weather & Forecast: [WeatherAPI.com](https://www.weatherapi.com/)

## Project Structure
```
BREATHEEASY/
├── config/                                                                           # Configuration files
│ └── config.yaml                                                                     # Main YAML configuration
├── data/
│ └── Post-Processing/
│ └── CSV_Files/
│ └── Master_AQI_Dataset.csv                                                          # Merged historical AQI + Weather data
├── models/                                                                           # Saved Prophet models (with weather regressors)
│ ├── Bangalore_prophet_model_v3_weather.json
│ ├── Chennai_prophet_model_v3_weather.json
│ ├── Delhi_prophet_model_v3_weather.json
│ ├── Hyderabad_prophet_model_v3_weather.json
│ └── Mumbai_prophet_model_v3_weather.json
├── notebook/                                                                         # Jupyter notebooks for exploration/development
│ ├── 1_Historical_Analysis.ipynb                                                     # Notebook for Sec 1 analysis ideas
│ └── 2_AQI_Forecasting_Development.ipynb                                             # Notebook for Sec 4 model dev/eval ideas
├── src/                                                                              # Main backend source code
│ ├── init.py
│ ├── config_loader.py                                                                # Loads config.yaml, sets up logging
│ ├── exceptions.py                                                                   # Custom exception classes
│ ├── analysis/                                                                       # Historical data analysis functions
│ │ ├── init.py
│ │ └── historical.py
│ ├── api_integration/                                                                # External API clients
│ │ ├── init.py
│ │ ├── client.py                                                                     # AQICN API client (AQI/Pollutants)
│ │ └── weather_client.py                                                             # WeatherAPI.com client (Current & Forecast)
│ ├── health_rules/                                                                   # AQI scales, thresholds, interpretation
│ │ ├── init.py
│ │ ├── info.py                                                                       # AQI scale definitions (CPCB)
│ │ └── interpreter.py                                                                # Pollutant risk interpretation (Sec 5)
│ └── modeling/                                                                       # ML model training and prediction
│ ├── init.py
│ ├── predictor.py                                                                    # Loads models, generates forecasts (Sec 4 & 6)
│ └── train.py                                                                        # Trains and saves models for all cities
├── tests/                                                                            # Automated tests
│ ├── init.py
│ ├── api_integration/
│ │ ├── init.py
│ │ ├── test_client.py
│ │ └── test_weather_client.py
│ └── health_rules/
│ ├── init.py
│ ├── test_info.py
│ └── test_interpreter.py
├── .env                                                                              # Environment variables (API keys) - MANUALLY CREATED
├── .gitignore                                                                        # Files/directories ignored by Git
├── requirements.txt                                                                  # Python dependencies
├── run_full_test.py                                                                  # Script for end-to-end backend testing
└── README.md                                                                         # This file
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/cp099/BreatheEasy.git
cd BreatheEasy
```

### 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment.

```bash
# Create (use python or python3 as appropriate for your system)
python3 -m venv venv
# or
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate on Windows (Command Prompt)
venv\Scripts\activate.bat

# Activate on Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note:** Prophet may require dependencies like `cmdstanpy`. Refer to Prophet’s installation guide if issues occur.

### 4. Prepare Data File
* Ensure your primary data file, named `Master_AQI_Dataset.csv`, is placed in `BreatheEasy/data/Post-Processing/CSV_Files/Master_AQI_Dataset.csv`

* This file __must__ contain daily AQI, pollutant data, and the __daily summarized weather data__ (used as regressors, e.g., `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`) for the target cities, covering at least the period from __Jan 1, 2018, to Feb 20, 2024__ (or the end date of your weather data). Data should be clean with no missing values in required columns for this period.

### 5.  Configure Settings (`config/config.yaml`)
* Review `config/config.yaml`.

* Verify the `paths:` section points correctly to your data and desired models directory.

* Ensure the `modeling: weather_regressors:` list contains the __exact column names__ of the daily weather features present in your `Master_AQI_Dataset.csv` that you want to use for training.

* Adjust `modeling: forecast_days:` (default is 3) if needed.

* Verify API base URLs under `apis:`.

### 6. Setup API Keys (`.env` file)
1. Create the `.env` __file:__ In the __root directory__ (`BreatheEasy/`), create a file named `.env`.

2. __Get API Keys:__
 - __AQICN:__ Sign up and get a token from [(https://aqicn.org/data-platform/token/)].
 - __WeatherAPI:__ Sign up and get a key from [(https://www.weatherapi.com/)].

3. __Add Keys to `.env`:__ Add the following lines to your `.env` file, replacing the placeholders with your actual keys:
```bash
AQICN_API_TOKEN=YOUR_ACTUAL_AQICN_TOKEN_HERE
WEATHERAPI_API_KEY=YOUR_ACTUAL_WEATHERAPI_KEY_HERE
```

4. __IMPORTANT:__ Ensure `.env` is listed in your `.gitignore` file to prevent accidentally committing your secrets. The provided `.gitignore` should already include this.

### 7. Train Forecast Models
The repository includes pre-trained V3 models incorporating weather data. To retrain them (e.g., after updating data or configuration):
```bash
python src/modeling/train.py
```
This will generate/overwrite `*_prophet_model_v3_weather.json` files in the `models/` directory based on the current data and configuration.

## Usage (Testing Backend Components)
Run these commands from the project __root directory (`BreatheEasy/`)__ with the virtual environment active:

* __Run Automated Tests:__
  ```bash
  pytest
  ```
  _(Check for passing tests. A warning from the `holidays` library about future incompatibility is expected and can be ignored if tests pass)._

* __Run End-to-End Script:__ Provides a text-based simulation of the application's output for a chosen city.
  ```bash
  python run_full_test.py
  ```
  _(Follow prompts to select a city)._

* __(Optional) Run Individual Module Tests:__ You can still run the if `__name__ == "__main__":` blocks within individual `src` files (e.g., `python src/api_integration/weather_client.py`), but `run_full_test.py` provides a more comprehensive check.

## Integrating into an Application
Import functions from the `src` modules (e.g., `generate_forecast`, `get_current_weather`, `get_predicted_weekly_risks`, etc.) into your chosen web framework (Flask, FastAPI, Streamlit, etc.). Create API endpoints/routes that call these backend functions, format the results (often using helpers like `format_forecast_for_ui`), and return JSON to your frontend UI.

## Contributing
Contributions welcome! Fork the repo, create a feature branch, commit your changes, and open a PR with a clear description.

## License
This project is licensed under the MIT License. 

---