import pandas as pd
import numpy as np

# --- Load your raw data ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Encoded_Data"

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Configuration ---
# REPLACE 'Target' with the exact name of your label column (e.g., 'Class', 'Label', 'Fault_Type')
target_column = df.columns[-1]

# --- 1. Drop Timestamp ---
# Requirement: "Do not Consider Timestamp as input"
if 'Timestamp' in df.columns:
    df = df.drop(columns=['Timestamp'])

# --- 2. Trim to full 60-sample blocks (Hourly) ---
block_size = 60
n = (len(df) // block_size) * block_size 
df = df.iloc[:n].copy()

print(f"Original samples processed: {len(df)}")

# --- 3. Assign Block_ID ---
df['Block_ID'] = np.arange(len(df)) // block_size

# --- 4. Define Aggregation Strategy ---
# We create a dictionary to tell pandas how to treat each column specifically
agg_dict = {}

for col in df.columns:
    if col == 'Block_ID':
        continue
    elif col == target_column:
        # --- SELECT TARGET ---
        # Select the first label in the block. 
        # Alternatives: 'last', or lambda x: x.mode()[0] (most frequent)
        agg_dict[col] = 'first' 
    else:
        # --- AVERAGE FEATURES ---
        # Average all other columns (Inputs)
        agg_dict[col] = 'mean'

# --- 5. Apply Grouping and Aggregation ---
df_hourly = df.groupby('Block_ID').agg(agg_dict).reset_index(drop=True)

print(f"✅ Hourly rows: {len(df_hourly)}")
print(df_hourly.head())

# --- Export ---
df_hourly.to_clipboard(index=False)