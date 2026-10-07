import pandas as pd
import numpy as np
import os

# Set file path
data_path = r"d:\ML\task\Data.xlsx"

# Threshold for considering an integer column as categorical
# Columns with <= 25 unique values are treated as categorical.
NUNIQUE_THRESHOLD = 25

print(f"Reading datasets from {data_path}...")
xls = pd.ExcelFile(data_path)
sheets = ['D1', 'D2', 'D3', 'D4']

encoded_data_dict = {}

for sheet in sheets:
    df = pd.read_excel(xls, sheet_name=sheet)
    print(f"\n--- Processing Sheet: {sheet} ---")
    
    encoded_df = df.copy()
    categorical_cols = []
    
    for col in encoded_df.columns:
        if col == 'ID':
            continue
            
        # Treat as categorical if it's an object/category type OR has few unique values
        if encoded_df[col].dtype == 'object' or encoded_df[col].dtype.name == 'category' or encoded_df[col].nunique() <= NUNIQUE_THRESHOLD:
            categorical_cols.append(col)
            # Apply Frequency Encoding
            freq = encoded_df[col].value_counts(normalize=True)
            encoded_df[col] = encoded_df[col].map(freq)
            
    if categorical_cols:
        print(f"Detected and Frequency Encoded Categorical Columns: {categorical_cols}")
    else:
        print("No categorical columns detected.")
        
    encoded_data_dict[f"{sheet}_Encoded"] = encoded_df

print("\nSaving encoded datasets to Excel...")
try:
    with pd.ExcelWriter(data_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        for sheet_name, df_encoded in encoded_data_dict.items():
            df_encoded.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved successfully to {data_path}")
except Exception as e:
    print(f"Error saving to Excel: {e}")
