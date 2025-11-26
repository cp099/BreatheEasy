# File: src/health_rules/calculator.py

"""
Provides functions to calculate the Air Quality Index (AQI) from raw
pollutant concentrations based on the Indian CPCB NAQI standard.
"""
import pandas as pd

POLLUTANT_BREAKPOINTS = {
    'pm10': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, float('inf'), 401, 500)],
    'pm25': [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, float('inf'), 401, 500)],
    'no2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, float('inf'), 401, 500)],
    'o3': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, float('inf'), 401, 500)],
    'co': [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10.0, 101, 200), (10.1, 17.0, 201, 300), (17.1, 34.0, 301, 400), (34.1, float('inf'), 401, 500)],
    'so2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, float('inf'), 401, 500)],
    'nh3': [(0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200), (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, float('inf'), 401, 500)],
}

def calculate_sub_index(value, pollutant):
    """Calculates the AQI sub-index for a single pollutant value."""
    if pd.isna(value):
        return None
    breakpoints = POLLUTANT_BREAKPOINTS.get(pollutant.lower())
    if not breakpoints:
        return None

    for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
        if bp_low <= value <= bp_high:
            sub_index = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (value - bp_low) + aqi_low
            return round(sub_index)
    return None 

def calculate_aqi_from_pollutants(data_row):
    """
    Calculates the final AQI for a row of data by taking the max of all sub-indices.
    """
    sub_indices = []
    pollutant_map = {'PM2.5': 'pm25', 'PM10': 'pm10', 'NO2': 'no2', 'O3': 'o3', 'CO': 'co', 'SO2': 'so2', 'NH3': 'nh3'}
    
    for df_col, pol_key in pollutant_map.items():
        if df_col in data_row:
            sub_index = calculate_sub_index(data_row[df_col], pol_key)
            if sub_index is not None:
                sub_indices.append(sub_index)

    if not sub_indices:
        return None
    
    return max(sub_indices)