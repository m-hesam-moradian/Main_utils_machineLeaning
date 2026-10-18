"""
Step 1 - Label Encoding
========================
Apply sklearn.preprocessing.LabelEncoder to every object/category/bool column
in the 'Data' sheet and save the encoded matrix as a new sheet 'Encoded_Data'
in the same Data.xlsx file.

This mirrors the pipeline used in BMM-EI No.219 / No.221 reference packages.
"""
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA = r"/home/z/my-project/task_223/Data.xlsx"

# ------------------------------------------------------------------
# 1. Load the raw 'Data' sheet
# ------------------------------------------------------------------
print("Loading 'Data' sheet ...")
df = pd.read_excel(DATA, sheet_name="Data")
print(f"  Shape: {df.shape}")

# ------------------------------------------------------------------
# 2. Label-encode every object / category / bool column
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 3. Quick sanity: class distribution after encoding (target = last col)
# ------------------------------------------------------------------
target_col = df_encoded.columns[-1]
print(f"\nTarget '{target_col}' encoded class distribution:")
print(df_encoded[target_col].value_counts().sort_index().to_string())

# ------------------------------------------------------------------
# 4. Save Encoded_Data sheet (replace if it already exists)
# ------------------------------------------------------------------
print("\nSaving 'Encoded_Data' sheet ...")
with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                     if_sheet_exists="replace") as writer:
    df_encoded.to_excel(writer, sheet_name="Encoded_Data", index=False)

print(f"[+] Encoded_Data saved to {DATA}")
print(f"[+] Final Encoded_Data shape: {df_encoded.shape}")
print(f"[+] Sheet names now: {pd.ExcelFile(DATA).sheet_names}")
