import pandas as pd

# --- Input Excel file path ---
excel_path = r"C:\Users\Sam\Desktop\ML\task\Data - Copy.xlsx"

# --- Load all sheet names ---
sheet_names = pd.ExcelFile(excel_path).sheet_names

# --- Collect processed DataFrames ---
merged_columns = []

for sheet in sheet_names:
    # Read sheet from row 3 onward (skip first 2 rows)
    df = pd.read_excel(excel_path, sheet_name=sheet, header=None, skiprows=2, usecols=[0, 1])

    # Rename columns using sheet name
    df.columns = [f"{sheet}", f"{sheet}"]

    merged_columns.append(df)

# --- Concatenate all horizontally ---
df_merged = pd.concat(merged_columns, axis=1)

# --- Copy to clipboard (no index, no header row) ---
df_merged.to_clipboard(index=False, header=True)

# --- Optional preview ---
print("✅ Merged table copied to clipboard:")
print(df_merged.head())