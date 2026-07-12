import pandas as pd

# ==========================
# Configuration
# ==========================
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data"
timestamp_col = "Timestamp"
samples_per_group = 6

# ==========================
# Load data
# ==========================
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# Ensure timestamp is datetime
df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
df = df.dropna(subset=[timestamp_col])

# Sort by timestamp
df = df.sort_values(timestamp_col).reset_index(drop=True)

# ==========================
# Convert every 6 samples into 1 sample
# by averaging numeric columns
# ==========================

# Create group IDs (0,0,0,0,0,0,1,1,1,1,1,1,...)
groups = df.index // samples_per_group

# Average only numeric columns
df_daily = df.groupby(groups).mean(numeric_only=True)

# Remove timestamp column if it still exists
if timestamp_col in df_daily.columns:
    df_daily = df_daily.drop(columns=[timestamp_col])

# ==========================
# Results
# ==========================
print(f"Original samples : {len(df)}")
print(f"Converted samples: {len(df_daily)}")
print(f"Expected samples : {len(df)//samples_per_group}")

print(df_daily.head())

# ==========================
# Copy to clipboard (Excel)
# ==========================
df_daily.to_clipboard(index=False)