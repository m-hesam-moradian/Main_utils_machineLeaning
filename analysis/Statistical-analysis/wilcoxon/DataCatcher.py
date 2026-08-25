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
    "Data", "Encoded_Data", "SMOTE_Data", "Balanced_Data", "vif_horizontal",
    "data_after_vif", "Search Space", "Run time", "Probs", "Probs(SMOTE)",
    "Brier_Decomposition", "Brier_Decomposition(SMOTE)", "predicts", "predicts(SMOTE)",
    "Statistical_t-test", "Statistical_t-test(SMOTE)", "Model_Comparison_Summary",
    "Model_Comparison_Summary(SMOTE)", "McNemar", "McNemar ", "FAST", "Entropy", "Entropy ",
    "Entropy_Uncertainty", "Entropy_Summary", "Morris_Sensitivity"
]

sheet_names = [
    s for s in all_sheets
    if s.strip() not in [x.strip() for x in ignore_sheets]
    and not s.endswith("_Metrics")
    and not s.endswith("_Metrics(SMOTE)")
    and not s.startswith("Data_after_KFold_")
    and not s.endswith("(SMOTE)")
    and not s.endswith("(ENN)")
]

print("Matching model sheets for DataCatcher predictions:")
print(sheet_names)

merged_columns = []

for sheet in sheet_names:
    try:
        df_raw = pd.read_excel(excel_path, sheet_name=sheet, header=None, nrows=5)
    except Exception:
        continue
    
    # Locate header row containing 'y_real' or 'y_pred'
    header_row_idx = None
    for r_idx in range(min(5, len(df_raw))):
        row_vals = [str(v).lower() for v in df_raw.iloc[r_idx].values]
        if any("y_real" in v or "y_pred" in v for v in row_vals):
            header_row_idx = r_idx
            break
            
    if header_row_idx is None:
        print(f"Skipping non-prediction sheet: {sheet}")
        continue
        
    df = pd.read_excel(excel_path, sheet_name=sheet, header=header_row_idx)
    
    # Filter first two prediction columns (y_real, y_pred)
    cols = []
    for col in df.columns:
        col_name = str(col).lower()
        if "y_real" in col_name or "y_pred" in col_name:
            cols.append(col)
            if len(cols) == 2:
                break
                
    if len(cols) < 2:
        print(f"Skipping sheet (insufficient target columns): {sheet}")
        continue
        
    df_sub = df[cols].dropna(how="all").reset_index(drop=True)
    df_sub.columns = [f"{sheet}", f"{sheet}"]
    merged_columns.append(df_sub)
    print(f"Loaded predictions from sheet: {sheet}")

# Merge all model prediction pairs side-by-side
df_merged = pd.concat(merged_columns, axis=1)

print("\nPredictions Matrix Preview:")
print(df_merged.head())
print("Shape:", df_merged.shape)

# Save to Excel sheet 'predicts(ENN)' and 'predicts'
close_excel_file(excel_path)
out_sheet = "predicts(ENN)" if "ENN_Data" in all_sheets else ("predicts(SMOTE)" if "SMOTE_Data" in all_sheets else "predicts")
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_merged.to_excel(writer, sheet_name=out_sheet, index=False)
    df_merged.to_excel(writer, sheet_name="predicts", index=False)

print(f"\nSaved combined model predictions to sheet '{out_sheet}' and 'predicts' in {excel_path}")