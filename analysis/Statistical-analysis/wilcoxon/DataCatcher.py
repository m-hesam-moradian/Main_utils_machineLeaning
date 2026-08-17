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
ignore_sheets = [
    "Data", "Encoded_Data", "SMOTE_Data", "Model_Comparison_Summary(SMOTE)",
    "Probs(SMOTE)", "Brier_Decomposition(SMOTE)", "predicts(SMOTE)", "predicts",
    "Statistical_t-test(SMOTE)"
]

sheet_names = [
    s for s in all_sheets
    if s not in ignore_sheets and not s.endswith("(SMOTE)") and not s.endswith("(ENN)")
]

print("Matching model sheets for DataCatcher predictions:")
print(sheet_names)

merged_columns = []

for sheet in sheet_names:
    df_raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
    
    # Locate header row
    header_row_idx = 1
    for r_idx in range(min(5, len(df_raw))):
        row_vals = [str(v).lower() for v in df_raw.iloc[r_idx].values]
        if any("y_real" in v or "y_pred" in v for v in row_vals):
            header_row_idx = r_idx
            break
            
    df = pd.read_excel(excel_path, sheet_name=sheet, header=header_row_idx)
    
    # Filter first two prediction columns (y_real, y_pred)
    cols = []
    for col in df.columns:
        col_name = str(col).lower()
        if "y_real" in col_name or "y_pred" in col_name:
            cols.append(col)
            if len(cols) == 2:
                break
                
    df_sub = df[cols].dropna(how="all").reset_index(drop=True)
    df_sub.columns = [f"{sheet}", f"{sheet}"]
    merged_columns.append(df_sub)

# Merge all model prediction pairs side-by-side
df_merged = pd.concat(merged_columns, axis=1)

print("\nPredictions Matrix Preview:")
print(df_merged.head())
print("Shape:", df_merged.shape)

# Save to Excel sheet 'predicts(SMOTE)'
close_excel_file(excel_path)
out_sheet = "predicts(SMOTE)"
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_merged.to_excel(writer, sheet_name=out_sheet, index=False)

print(f"\nSaved combined model predictions to sheet '{out_sheet}' in {excel_path}")