import pandas as pd

# --- Configuration ---
# Path to the input dataset.
# IMPORTANT: This is an absolute path. For portability, consider changing this
# to a relative path (e.g., "data/Master_AQI_Dataset.csv") and running the
# script from the project's root directory.
file_path = "/Users/apple/Personal_Files/Codes/AQI_Prediction_Project/Data/Post-Processing/CSV_Files/Master_AQI_Dataset.csv"

# --- Data Loading and Processing ---

# Load the dataset from the specified file path into a pandas DataFrame.
df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime objects using the DD/MM/YY format,
# then immediately format it back into a DD/MM/YYYY string.
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y').dt.strftime('%d/%m/%Y')

# Apply One-Hot Encoding to the 'City' column.
# This creates new columns for each unique city (e.g., 'City_Delhi', 'City_Mumbai').
# 'drop_first=False' ensures that a column is created for every city.
df_encoded = pd.get_dummies(df, columns=['City'], drop_first=False)

# The get_dummies function creates columns of type 'bool' (True/False).
# This step converts all boolean columns into integers (1 for True, 0 for False).
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# --- Save Output ---

# Define the name for the output file.
output_file = "Encoded_AQI_Dataset.csv"

# Save the fully processed DataFrame to a new CSV file.
# 'index=False' prevents pandas from writing the DataFrame index as a column.
df_encoded.to_csv(output_file, index=False)

# Print a confirmation message to the console.
print(f"Encoded dataset saved as {output_file} with Date format DD/MM/YYYY and categorical values as 0/1.")