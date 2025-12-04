import pandas as pd

# --- Config ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data"
timestamp_col = "timestamp"

# --- Load ---
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# Ensure timestamp is datetime
df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
df = df.dropna(subset=[timestamp_col])

# --- Sort by timestamp ---
df = df.sort_values(timestamp_col)

# --- Group by hour and take the first row of each hour ---
df_hourly = df.groupby(df[timestamp_col].dt.floor("H")).first().reset_index(drop=True)

# --- Show result ---
print(df_hourly.head(10))

# --- Copy to clipboard for Excel ---
df_hourly.to_clipboard(index=False)