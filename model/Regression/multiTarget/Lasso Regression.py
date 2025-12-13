import pandas as pd
import numpy as np

# --- Load your raw data ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "BSE19_Raw"   # update if needed

df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Ensure Timestamp is datetime ---
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# --- Trim to full 96-sample blocks ---
block = 96
n = (len(df) // block) * block   # keep only complete days
df = df.iloc[:n].copy()

# --- Assign block_id (each day = 96 samples) ---
df['Day_ID'] = np.arange(len(df)) // block

# --- Average each block of 96 rows ---
df_daily = df.groupby('Day_ID', as_index=False).mean(numeric_only=True)

# --- Optional: keep the first timestamp of each day ---
df_daily['Timestamp'] = df.groupby('Day_ID')['Timestamp'].first().values

# --- Drop helper column ---
df_daily = df_daily.drop(columns=['Day_ID'])

print("✅ Daily rows:", len(df_daily))  # should be 52
print(df_daily.head())

# --- Save to Excel ---
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_daily.to_excel(writer, sheet_name="BSE19_Daily", index=False)