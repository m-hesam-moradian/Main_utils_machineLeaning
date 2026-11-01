import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import win32com.client

sys.stdout.reconfigure(encoding='utf-8')

# ================== Excel Helpers ==================
def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            try:
                if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                    wb.Save()
                    wb.Close(SaveChanges=False)
                    print("Saved and closed Excel file:", filepath)
                    break
            except Exception:
                pass
    except Exception as e:
        print("Note: Excel is not running or COM skipped:", e)

def open_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("Opened Excel file:", filepath)
    except Exception as e:
        print("Note: Could not auto-open Excel GUI:", e)

# ================== Paths ==================
input_path = r"task\BMM-EI. No.240 (Dataset).xlsx"
output_path = r"task\Data.xlsx"

close_excel_file(output_path)

print(f"Loading raw dataset from '{input_path}'...")
df_raw = pd.read_excel(input_path)
print(f"Loaded {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

# ================== Preprocessing: Timestamp Extraction ==================
# Extract hour, month, and day from Timestamp variable and add to the inputs, then remove Timestamp
print("Extracting Hour, Month, Day from Timestamp...")
dt_series = pd.to_datetime(df_raw['Timestamp'], format='%m/%d/%Y %H:%M')
df_raw['Hour'] = dt_series.dt.hour
df_raw['Month'] = dt_series.dt.month
df_raw['Day'] = dt_series.dt.day

# Drop Timestamp
df_processed = df_raw.drop(columns=['Timestamp'])

# Handle missing values in Auth_Method (assign 'None' for missing)
df_processed['Auth_Method'] = df_processed['Auth_Method'].fillna('None')

# Reorder columns so target 'Transaction_Status' is strictly the last column
target_col = 'Transaction_Status'
features = [c for c in df_processed.columns if c != target_col]
df_processed = df_processed[features + [target_col]]

print(f"Preprocessed dataset shape: {df_processed.shape}")
print(f"Features ({len(features)}): {features}")
print(f"Target: {target_col}")

# ================== Label Encoding ==================
df_encoded = df_processed.copy()
mappings = []

for col in df_encoded.columns:
    if not pd.api.types.is_numeric_dtype(df_encoded[col]) or df_encoded[col].dtype.name in ['str', 'string', 'object', 'category', 'bool']:
        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
        for orig, enc in zip(encoder.classes_, encoder.transform(encoder.classes_)):
            mappings.append({"Column": col, "Original_Value": orig, "Encoded_Value": int(enc)})
        print(f"Encoded '{col}': {len(encoder.classes_)} unique classes -> {list(encoder.classes_)[:5]}")

df_mapping = pd.DataFrame(mappings)

# ================== Save to Excel ==================
print(f"Saving to '{output_path}'...")
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df_processed.to_excel(writer, sheet_name="Data", index=False)
    df_encoded.to_excel(writer, sheet_name="Encoded_Data", index=False)
    df_mapping.to_excel(writer, sheet_name="Encoding_Mapping", index=False)

print("[OK] Data, Encoded_Data, and Encoding_Mapping saved successfully!")
open_excel_file(output_path)
