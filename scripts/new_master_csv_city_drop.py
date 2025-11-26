# File: scripts/new_master_csv_city_drop.py

import pandas as pd

df = pd.read_csv('Master_AQI_Weather_India_CatEncoded.csv', parse_dates=['Datetime'])

df = df.drop(columns=['City'])

df.to_csv('Master_AQI_Weather_India_CatEncoded_noCity.csv', index=False)

print("Column 'City' dropped. New file saved as Master_AQI_Weather_India_CatEncoded_noCity.csv")
print('Data shape:', df.shape)
print(df.head())