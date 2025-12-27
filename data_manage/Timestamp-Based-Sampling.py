import pandas as pd
import numpy as np

# --- Load your raw data ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data"

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Ensure Timestamp is datetime ---
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# --- Trim to full 6-sample blocks (Hourly) ---
# Logic: 10 mins * 6 = 60 mins (1 Hour)
block_size = 6
n = (len(df) // block_size) * block_size 
df = df.iloc[:n].copy()

print(f"Original samples processed: {len(df)}") # Should be approx 50088 or 50090 based on divisibility

# --- Assign Block_ID (each hour = 6 samples) ---
df['Block_ID'] = np.arange(len(df)) // block_size

# --- Average each block of 6 rows ---
# numeric_only=True automatically drops 'Timestamp' and other non-numeric columns
df_hourly = df.groupby('Block_ID', as_index=False).mean(numeric_only=True)

# --- Drop the helper column ---
df_hourly = df_hourly.drop(columns=['Block_ID'])

print("✅ Hourly rows (Target ~8348/8349):", len(df_hourly))
print(df_hourly.head())

# --- Export ---
df_hourly.to_clipboard(index=False)