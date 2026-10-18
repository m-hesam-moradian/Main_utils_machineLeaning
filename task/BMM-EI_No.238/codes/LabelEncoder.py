"""
Step 1 - Data Inspection + Label Encoding
=========================================
Inspect the dataset, apply LabelEncoder to categorical columns,
and save Encoded_Data sheet to Data.xlsx.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA = r"/home/z/my-project/task_238/Data.xlsx"

# Load
print("Loading data ...")
df = pd.read_excel(DATA, sheet_name="Optical_Network_Enhanced_Datase")
print(f"  Shape: {df.shape}")

# Rename sheet: save as 'Data' sheet for standard pipeline
# Encode categorical columns
df_encoded = df.copy()
encoder = LabelEncoder()
encoded_cols = []
for col in df_encoded.columns:
    if (df_encoded[col].dtype == "object"
            or df_encoded[col].dtype.name == "category"
            or df_encoded[col].dtype.name == "bool"):
        df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
        encoded_cols.append(col)

print(f"\nEncoded columns ({len(encoded_cols)}):")
for c in encoded_cols:
    print(f"  - {c}")

# Target distribution
target_col = df_encoded.columns[-1]
print(f"\nTarget '{target_col}' encoded class distribution:")
print(df_encoded[target_col].value_counts().sort_index().to_string())

# Save Data + Encoded_Data sheets
print("\nSaving 'Data' and 'Encoded_Data' sheets ...")
with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                     if_sheet_exists="replace") as writer:
    df.to_excel(writer, sheet_name="Data", index=False)
    df_encoded.to_excel(writer, sheet_name="Encoded_Data", index=False)

print(f"[+] Done. Sheets: {pd.ExcelFile(DATA).sheet_names}")
print(f"[+] Encoded_Data shape: {df_encoded.shape}")
