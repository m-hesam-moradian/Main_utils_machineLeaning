import pandas as pd
import os
import win32com.client

def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)

xl = pd.ExcelFile(excel_path)
all_sheets = xl.sheet_names

# Exclude non-model data sheets
ignore_sheets = ["Data", "Encoded_Data", "SMOTE_Data", "Model_Comparison_Summary(SMOTE)"]
sheet_names = [
    s for s in all_sheets
    if s not in ignore_sheets and not s.endswith("(SMOTE)")
]

print("Matching model sheets for DataCatcher:")
print(sheet_names)

merged_columns = []

for sheet in sheet_names:
    df_raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
    
    # Locate row containing 'y_real' or 'y_pred' (typically row 1)
    header_row_idx = 1
    for r_idx in range(min(5, len(df_raw))):
        row_vals = [str(v).lower() for v in df_raw.iloc[r_idx].values]
        if any("y_real" in v or "y_pred" in v for v in row_vals):
            header_row_idx = r_idx
            break
            
    df = pd.read_excel(excel_path, sheet_name=sheet, header=header_row_idx)
    
    # Filter columns for y_real, y_pred, and probability columns
    cols = []
    for col in df.columns:
        col_name = str(col).lower()
        if "y_real" in col_name or "y_pred" in col_name or "prob" in col_name:
            cols.append(col)
            
    df = df[cols].dropna(how="all").reset_index(drop=True)
    
    # Rename columns dynamically
    new_cols = []
    for i in range(df.shape[1]):
        if i == 0:
            new_cols.append(f"{sheet}_y_real")
        elif i == 1:
            new_cols.append(f"{sheet}_y_pred")
        else:
            new_cols.append(f"{sheet}_prob_{i-2}")
            
    df.columns = new_cols
    merged_columns.append(df)

# Merge all model columns side-by-side
df_merged = pd.concat(merged_columns, axis=1)

print("\nDataCatcher Probability Matrix:")
print(df_merged.head())
print("Shape:", df_merged.shape)

# Save to Excel sheet 'Probs(SMOTE)'
close_excel_file(excel_path)
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_merged.to_excel(writer, sheet_name="Probs(SMOTE)", index=False)

print(f"\nSaved combined model predictions and probabilities to sheet 'Probs(SMOTE)' in {excel_path}")