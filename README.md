# BreatheEasy - AQI Analysis and Forecasting Backend

## Overview
BreatheEasy provides the backend logic for a system designed to analyze historical Air Quality Index (AQI) data, fetch real-time AQI and weather information, provide insights into associated health risks based on the Indian CPCB standards, and predict future AQI levels for major Indian cities (Delhi, Mumbai, Bangalore, Chennai, Hyderabad).

The system interprets both current pollutant levels and predicted AQI values to indicate potential respiratory health implications, aiming to empower users with actionable air quality information. This repository contains the complete backend Python code, including data processing modules, API clients, ML model training scripts, prediction logic, and health rule interpreters.

## Problem Statement
Air quality index (AQI) data is widely available, but translating numerical values into clear health risks remains a challenge for many users, especially those with respiratory conditions. Furthermore, anticipating changes in air quality requires reliable forecasting. BreatheEasy aims to bridge this gap by providing accessible interpretations of current and future air quality based on established standards and predictive modeling.

## Core Backend Features Implemented
*(Mapping to original UI sections)*

- **(Sec 0.5) Current Weather:** Fetches real-time weather conditions (temp, description, icon, etc.) using the WeatherAPI.com service (`src/api_integration/weather_client.py`).
- **(Sec 1) Historical Summary:** Provides functions to load historical data and retrieve AQI trends and distributions per city (`src/analysis/historical.py`).
- **(Sec 2) Educational Info:** Contains the definition of AQI and the CPCB India AQI scale with associated health implications (`src/health_rules/info.py`). Also includes pollutant-specific health risk thresholds (`src/health_rules/interpreter.py`).
- **(Sec 3) Current AQI:** Fetches real-time AQI from specified city stations via the AQICN API (`src/api_integration/client.py`).
- **(Sec 4) AQI Forecast:** Trains city-specific Prophet models (`src/modeling/train.py`) and provides functions to generate a 5-day AQI forecast, optionally adjusted using the latest live AQI (`src/modeling/predictor.py`). Includes UI-friendly formatting.
- **(Sec 5) Current Pollutants & Risks:** Fetches real-time pollutant data via AQICN API (`client.py`) and interprets immediate health risks based on CPCB-derived thresholds (`interpreter.py`).
- **(Sec 6) Predicted Risks:** Takes the 5-day AQI forecast and interprets the potential health implications for each day based on the predicted AQI level and CPCB categories (`predictor.py` calls functions in `info.py`).

## Technology Stack
- **Language:** Python 3.x
- **Core Libraries:**
  - Pandas: Data manipulation and analysis.
  - Prophet (by Meta): Time series forecasting.
  - Requests: HTTP requests for external APIs.
  - python-dotenv: Managing environment variables (API keys).
  - scikit-learn: Evaluation metrics (MAE, RMSE, MAPE).
  - CmdStanPy: Backend for Prophet model fitting.
- **Data Sources**:
  - Historical AQI: `Master_AQI_Dataset.csv` (user provided). The AQI data was sourced from the Central Pollution Control Board (CPCB), and pollutant concentrations were calculated using CPCB standards and formulas to ensure accuracy.
  - Real-time AQI/Pollutants: [World Air Quality Index Project (aqicn.org)](https://aqicn.org/api/)
  - Real-time Weather: [WeatherAPI.com](https://www.weatherapi.com/)

## Project Structure
```
BREATHEEASY/
├── data/
│   └── Post-Processing/
│       └── CSV_Files/
│           └── Master_AQI_Dataset.csv  # Main historical dataset
├── models/                             # Saved Prophet models (*_prophet_model_v2.json)
├── notebook/                           # Jupyter notebooks for exploration/development
│   └── 2_AQI_Forecasting_Development.ipynb
├── src/                                # Main backend source code
│   ├── __init__.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── historical.py
│   ├── api_integration/
│   │   ├── __init__.py
│   │   ├── client.py                   # AQICN API client
│   │   └── weather_client.py          # WeatherAPI.com client
│   ├── health_rules/
│   │   ├── __init__.py
│   │   ├── info.py                    # AQI scale definitions (CPCB)
│   │   └── interpreter.py            # Pollutant risk interpretation (Sec 5)
│   └── modeling/
│       ├── __init__.py
│       ├── predictor.py              # Loads models, generates forecasts (Sec 4 & 6)
│       └── train.py                  # Trains and saves models for all cities
├── .env                               # Environment variables (API keys) - MUST BE CREATED MANUALLY
├── .gitignore                         # Files/directories ignored by Git
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/cp099/BreatheEasy.git # Replace with your actual repo URL if different
cd BreatheEasy
```

### 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment.

```bash
# Create
python3 -m venv venv
# or
python -m venv venv

# Activate on macOS / Linux
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

### 4. Place Data File
Ensure your historical data file is located at:
```
BreatheEasy/data/Post-Processing/CSV_Files/Master_AQI_Dataset.csv
```

### 5. Setup API Keys (.env file)
Create a `.env` file in the root directory and add:
```
AQICN_API_TOKEN=YOUR_ACTUAL_AQICN_TOKEN_HERE
WEATHERAPI_API_KEY=YOUR_ACTUAL_WEATHERAPI_KEY_HERE
```
> Ensure `.env` is included in `.gitignore` (already done).

### 6. Train Forecast Models (Optional if already committed)
```bash
python src/modeling/train.py
```

## Usage (Testing Backend Components)
Run the following from the project root (`BreatheEasy/`):

```bash
# Test AQICN API Client
python src/api_integration/client.py

# Test WeatherAPI Client
python src/api_integration/weather_client.py

# Test AQI Scale Info
python src/health_rules/info.py

# Test Pollutant Risk Interpreter
python src/health_rules/interpreter.py

# Test Model Predictor
python src/modeling/predictor.py

# Test Historical Analysis Module
python src/analysis/historical.py

# Run Development Notebook
jupyter notebook notebook/2_AQI_Forecasting_Development.ipynb
```

## Integrating into an Application
You can import functions like `generate_forecast`, `get_current_weather`, `get_predicted_weekly_risks`, etc., into a web framework (Flask, FastAPI, Streamlit) for UI integration.

## Contributing
Contributions are welcome! Fork the repo, create a branch, commit your changes, and open a PR with a clear description.

## License
This project is licensed under the MIT License.

---

**Key Updates Made:**

- Refined Overview and Feature descriptions.
- Added a Technology Stack section.
- Updated the Project Structure.
- Clarified and detailed setup steps.
- Emphasized API key setup and data placement.
- Explained optional model retraining.
- Added backend module test instructions.
- Suggested integration path with web frameworks.