# BreatheEasy - AQI Analysis and Forecasting

## Overview
BreatheEasy is a project designed to analyze historical Air Quality Index (AQI) data, provide insights into associated health risks, and predict future AQI levels for major Indian cities. The system not only interprets AQI values to indicate potential respiratory diseases but also forecasts AQI trends for the upcoming week using historical data and pollutant trends.

## Problem Statement
Air quality index (AQI) data is widely available, but most weather websites only provide numerical values without clear insights into the health risks associated with air pollution. This lack of detailed information makes it difficult for individuals, especially those with respiratory conditions, to understand the specific health threats they face. Additionally, accurate AQI forecasting is essential to help people prepare for worsening air quality in advance.

This project aims to bridge that gap by developing a system that provides both real-time health risk analysis and predictive insights. By offering timely warnings and educational information, BreatheEasy empowers individuals to take proactive measures for their respiratory health.

## Features
- **Historical Summary:** Displays trends and summaries of past AQI data per city.
- **Educational Info:** Explains AQI, pollutants, and health risks.
- **Current AQI:** Shows real-time AQI using an external API.
- **AQI Forecast:** Predicts AQI for the next week using machine learning models.
- **Current Pollutants & Risks:** Displays real-time pollutant levels and associated immediate health triggers.
- **Predicted Risks:** Forecasts potential health risks for the upcoming week based on predicted AQI levels.

## Project Structure
```
BREATHEEASY/
├── config/                # Configuration files
├── data/                  # Raw and processed data
├── data_processing_code/   # Scripts used for initial data processing
├── models/                # Saved machine learning models
├── notebook/              # Jupyter notebooks for exploration and analysis
├── src/                   # Main source code (modules, app logic)
├── tests/                 # Unit and integration tests
├── .gitignore             # Files ignored by Git
├── README.md              # This file
├── requirements.txt       # Project dependencies
└── .env                   # Environment variables (API keys, etc.)
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd BREATHEEASY
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Place Data Files
Ensure `Master_AQI_Dataset.csv` is located in:
```
data/Post-Processing/CSV_Files/
```

### 5. API Keys (if applicable)
- Create a `.env` file in the root directory.
- Add necessary API keys:
  ```env
  AQI_API_KEY=your_key_here
  ```
- Ensure `.env` is listed in `.gitignore`.

## Usage

### Running the Analysis Notebook
```bash
jupyter notebook notebook/1_Historical_Analysis.ipynb
```

### Starting the Web App (if applicable)
```bash
python src/app/main.py
```

## Contributing
If you'd like to contribute, please follow these guidelines:
1. Fork the repository and create a new branch.
2. Make your changes and commit with clear messages.
3. Submit a pull request for review.

## License
This project is licensed under the **MIT License**. See `LICENSE` for more details.

---

## Additional Setup: `BREATHEEASY/src/__init__.py`
- **Action:** Create this *empty* file inside the `src` directory.
- **Purpose:** Marks `src` as a Python package, allowing structured imports.
- **Content:** The file can remain empty or include a basic comment.

```python
# File: src/__init__.py
# This file intentionally left blank to mark 'src' as a Python package.
```

---

### Next Steps
- Finalize the problem statement and ensure it reflects all project goals.
- Update setup instructions based on project progress.
- Document API endpoints and model details for easy reference.

Let me know if you need any further improvements! 🚀
