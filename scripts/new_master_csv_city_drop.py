import pandas as pd

# Load the merged master file
df = pd.read_csv('Master_AQI_Weather_India_CatEncoded.csv', parse_dates=['Datetime'])

# Drop the 'City' column
df = df.drop(columns=['City'])

# Save the new file
df.to_csv('Master_AQI_Weather_India_CatEncoded_noCity.csv', index=False)

print("Column 'City' dropped. New file saved as Master_AQI_Weather_India_CatEncoded_noCity.csv")
print('Data shape:', df.shape)
print(df.head())