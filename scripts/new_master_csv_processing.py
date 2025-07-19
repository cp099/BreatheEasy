import pandas as pd

# Define cities and their weather files
weather_files = {
    'Bangalore': 'Bangalore.csv',
    'Chennai': 'Chennai.csv',
    'Kolkata': 'Kolkata.csv',
    'Mumbai': 'Mumbai.csv',
    'New Delhi': 'New Delhi.csv'
}

# Load AQI data
aqi_cols = [
    'City', 'Datetime', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
    'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI', 'AQI_Bucket'
]
aqi_df = pd.read_csv('city_hour.csv', usecols=aqi_cols, parse_dates=['Datetime'])
# Make AQI datetime column timezone-naive (should already be, but just in case)
aqi_df['Datetime'] = pd.to_datetime(aqi_df['Datetime']).dt.tz_localize(None)

merged_city_dfs = []

for city, weather_file in weather_files.items():
    ## Load weather data for the city
    weather_df = pd.read_csv(weather_file, parse_dates=['date'])
    weather_df.rename(columns={'date': 'Datetime'}, inplace=True)
    # Ensure timezone-naive datetimes to avoid merge errors
    weather_df['Datetime'] = pd.to_datetime(weather_df['Datetime']).dt.tz_localize(None)
    weather_df['City'] = city  # Add city column

    ## Filter AQI for just this city
    city_aqi_df = aqi_df[aqi_df['City'].str.lower() == city.lower()]

    ## Merge on City and Datetime
    merged = pd.merge(city_aqi_df, weather_df, on=['City', 'Datetime'], how='inner')
    merged_city_dfs.append(merged)

# Concatenate all merged dataframes
final_df = pd.concat(merged_city_dfs, ignore_index=True)

# One-hot encode city columns: City_Bangalore, City_Chennai, City_Kolkata, City_Mumbai, City_New_Delhi
city_names = list(weather_files.keys())
for c in city_names:
    colname = 'City_' + c.replace(" ", "_")
    final_df[colname] = (final_df['City'].str.lower() == c.lower()).astype(int)

# Only keep data from 2015-01-01 onward
start_date = pd.to_datetime('2015-01-01 00:00:00')
final_df = final_df[final_df['Datetime'] >= start_date]

# Desired column order
final_columns = [
    'Datetime', 'City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
    'Benzene', 'Toluene', 'Xylene', 'AQI', 'AQI_Bucket',
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
    'precipitation', 'rain', 'snowfall', 'snow_depth', 'pressure_msl', 'surface_pressure',
    'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
    'wind_speed_10m', 'wind_speed_100m', 'wind_direction_10m',
    'wind_direction_100m', 'wind_gusts_10m'
] + ['City_' + c.replace(" ", "_") for c in city_names]

# Only include final columns that exist (in case some weather stations lack certain fields)
final_columns = [col for col in final_columns if col in final_df.columns]

# Save the result
final_df = final_df[final_columns]
final_df.to_csv('Master_AQI_Weather_India_CatEncoded.csv', index=False)

print('Done! Merged, encoded master file saved as Master_AQI_Weather_India_CatEncoded.csv')
print('Final dataframe shape:', final_df.shape)
print('Sample rows:')
print(final_df.head())