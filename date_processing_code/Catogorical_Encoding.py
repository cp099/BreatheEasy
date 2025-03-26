import pandas as pd

# Load the dataset
file_path = "/Users/apple/Personal_Files/Codes/AQI_Prediction_Project/Data/Post-Processing/CSV_Files/Master_AQI_Dataset.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format and ensure DD/MM/YYYY format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y').dt.strftime('%d/%m/%Y')

# Apply One-Hot Encoding on 'City' (keeping all cities)
df_encoded = pd.get_dummies(df, columns=['City'], drop_first=False)

# Convert only boolean columns (City_*) to integer (0 and 1)
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# Save the processed dataset to a new CSV file
output_file = "Encoded_AQI_Dataset.csv"
df_encoded.to_csv(output_file, index=False)

print(f"Encoded dataset saved as {output_file} with Date format DD/MM/YYYY and categorical values as 0/1.")