import pandas as pd
import numpy as np

# --- Config ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data"  # replace with the actual sheet
timestamp_col = "timestamp"   # the column name in your Excel

# Optional: if you want exactly 31 days -> 744 hours, set the month explicitly
target_month = "2023-01"      # e.g., January 2023; set to None to skip month filtering

# --- Load ---
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# Ensure timestamp is datetime
df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
df = df.dropna(subset=[timestamp_col])

print("Initial number of samples:", len(df))

# Sort by time and set index
df = df.sort_values(timestamp_col)
df = df.set_index(timestamp_col)

# Optional: filter to a specific month to ensure 24*31=744 hours
if target_month is not None:
    # Select the entire month (inclusive)
    start = pd.Timestamp(target_month)
    end = (start + pd.offsets.MonthEnd(0)).normalize() + pd.Timedelta(days=1)  # next day after month end
    df = df.loc[start:end]

# Resample hourly and take the first sample per hour
# Lowercase 'h' avoids the FutureWarning
df_hourly = df.resample('h').first()

# If some hours have no data, drop them
df_hourly = df_hourly.dropna(how="all")

# Check size; if you filtered to a full 31-day month, you expect 744 rows
print("Number of hourly samples after resampling:", len(df_hourly))

# If you need exactly 744, you can enforce it by trimming or filling; here we trim if there are extras
if len(df_hourly) >= 744:
    df_hourly = df_hourly.iloc[:744]
else:
    print(f"Warning: Only {len(df_hourly)} hourly rows found; fewer than 744 due to missing hours.")

# Reset index to restore 'timestamp' as a column
df_hourly = df_hourly.reset_index()

# Randomize/shuffle rows
df_randomized = df_hourly.sample(frac=1, random_state=42).reset_index(drop=True)

# Show sample
print(df_randomized.head())

# Copy to clipboard
df_randomized.to_clipboard(index=False)