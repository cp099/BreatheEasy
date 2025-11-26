# File: scripts/Catogorical_Encoding.py

import pandas as pd

# --- Configuration ---
file_path = "/Users/apple/Personal_Files/Codes/AQI_Prediction_Project/Data/Post-Processing/CSV_Files/Master_AQI_Dataset.csv"

# --- Data Loading and Processing ---
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y').dt.strftime('%d/%m/%Y')
df_encoded = pd.get_dummies(df, columns=['City'], drop_first=False)
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# --- Save Output ---
output_file = "Encoded_AQI_Dataset.csv"
df_encoded.to_csv(output_file, index=False)
print(f"Encoded dataset saved as {output_file} with Date format DD/MM/YYYY and categorical values as 0/1.")