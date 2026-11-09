import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

excel_path = r"task/Data.xlsx"

def close_excel_file(filepath):
    try:
        import win32com.client
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName).lower() == os.path.abspath(filepath).lower():
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception as e:
        pass

def open_excel_file(filepath):
    try:
        import win32com.client
        excel = win32com.client.GetActiveObject("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("[+] Opened Excel file:", filepath)
    except Exception as e:
        print("Note: Could not auto-open Excel GUI:", e)

print("[1] Closing Excel if open...")
close_excel_file(excel_path)

print("[2] Reading raw Data sheet from", excel_path)
df_raw = pd.read_excel(excel_path, sheet_name="Data")
print(f"Raw data shape: {df_raw.shape}")

# Identify categorical columns
cat_cols = []
for col in df_raw.columns:
    if df_raw[col].dtype == "object" or df_raw[col].dtype.name == "category" or df_raw[col].dtype.name == "bool" or pd.api.types.is_string_dtype(df_raw[col]):
        cat_cols.append(col)

print(f"Identified {len(cat_cols)} categorical columns: {cat_cols}")

# Apply LabelEncoder and build mapping table
df_encoded = df_raw.copy()
mapping_rows = []

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    encoders[col] = le
    for orig_cls, enc_val in zip(le.classes_, range(len(le.classes_))):
        cnt = (df_raw[col] == orig_cls).sum()
        pct = (cnt / len(df_raw)) * 100
        mapping_rows.append({
            "Column": col,
            "Original Category": orig_cls,
            "Encoded Value": enc_val,
            "Sample Count": cnt,
            "Percentage (%)": round(pct, 2)
        })

df_mapping = pd.DataFrame(mapping_rows)
print("\nEncoding Mapping Summary:")
print(df_mapping.to_string(index=False))

# Target column verification
target_col = "Adaptation Status"
print(f"\nTarget column: '{target_col}'")
print(f"Encoded classes: {encoders[target_col].classes_}")
for idx, cls_name in enumerate(encoders[target_col].classes_):
    print(f"  Class {idx} -> {cls_name}: {(df_encoded[target_col] == idx).sum()} samples")

# Early 20% Holdout Test Separation
print("\n[3] Splitting into 80% Train_Data and 20% Test_Data (stratified, random_state=42)...")
train_df, test_df = train_test_split(
    df_encoded,
    test_size=0.20,
    random_state=42,
    stratify=df_encoded[target_col]
)

# Reset index
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train_Data shape: {train_df.shape} (80.0%)")
print(f"Test_Data shape:  {test_df.shape} (20.0%)")

print("\nTrain_Data class distribution:")
for idx, cls_name in enumerate(encoders[target_col].classes_):
    cnt = (train_df[target_col] == idx).sum()
    pct = (cnt / len(train_df)) * 100
    print(f"  Class {idx} ({cls_name}): {cnt} ({pct:.2f}%)")

print("\nTest_Data class distribution:")
for idx, cls_name in enumerate(encoders[target_col].classes_):
    cnt = (test_df[target_col] == idx).sum()
    pct = (cnt / len(test_df)) * 100
    print(f"  Class {idx} ({cls_name}): {cnt} ({pct:.2f}%)")

# Save sheets to task/Data.xlsx
print("\n[4] Writing sheets to task/Data.xlsx...")
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_encoded.to_excel(writer, sheet_name="Encoded_Data", index=False)
    df_mapping.to_excel(writer, sheet_name="Encoding_Mapping", index=False)
    train_df.to_excel(writer, sheet_name="Train_Data", index=False)
    test_df.to_excel(writer, sheet_name="Test_Data", index=False)

print("[+] Successfully saved sheets: 'Encoded_Data', 'Encoding_Mapping', 'Train_Data', 'Test_Data'")
