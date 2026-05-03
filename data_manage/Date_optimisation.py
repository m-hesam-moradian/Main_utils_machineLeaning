import pandas as pd

# --- Config ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
sheet_name = "Data"
timestamp_col = "Timestamp"

# --- Load ---
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# --- Ensure timestamp is datetime ---
df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
df = df.dropna(subset=[timestamp_col])

# --- Sort by timestamp ---
df = df.sort_values(timestamp_col)

# --- Extract date (daily grouping key) ---
df["date"] = df[timestamp_col].dt.date

# --- Separate numerical & categorical columns ---
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

# Remove timestamp/date from categorical if present
cat_cols = [col for col in cat_cols if col not in [timestamp_col, "date"]]

# --- Aggregation functions ---
agg_dict = {}

# Mean for numerical
for col in num_cols:
    agg_dict[col] = "mean"

# Mode for categorical (safe mode handling)
def get_mode(series):
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else None

for col in cat_cols:
    agg_dict[col] = get_mode

# --- Group by day ---
df_daily = df.groupby("date").agg(agg_dict).reset_index()

# --- Show result ---
print("Final shape:", df_daily.shape)  # should be ~1520 rows
print(df_daily.head(10))

# --- Copy to clipboard ---
df_daily.to_clipboard(index=False)