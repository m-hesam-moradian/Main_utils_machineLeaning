import pandas as pd
import numpy as np
import os
import win32com.client
from scipy.stats import zscore

def close_excel_file(filepath):
    try:
        excel = win32com.client.Dispatch("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("💾 Saved and 🔒 Closed Excel file:", filepath)
                break
    except Exception:
        pass

def z_score_processing(df, sheet_name, threshold=3.0):
    df_raw = df.copy()
    id_col = df_raw['ID']
    features = df_raw.drop(columns=['ID'])
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    
    # Calculate Z-scores
    z_scores = features[numeric_cols].apply(zscore)
    
    # Z-Score columns
    z_score_columns = z_scores.add_prefix('Z_Score_')
    df_full_details = pd.concat([df_raw, z_score_columns], axis=1)
    
    # Create mask for outliers > 3 or < -3
    outlier_mask = (z_scores > threshold) | (z_scores < -threshold)
    rows_with_outliers = outlier_mask.any(axis=1)
    total_removed = rows_with_outliers.sum()
    
    df_full_details['Outlier_Status'] = np.where(rows_with_outliers, 'Removed (Outlier)', 'Kept')
    
    report_data = []
    for col in numeric_cols:
        count = outlier_mask[col].sum()
        if count > 0:
            report_data.append({"Feature / Detail": col, "Rows Triggered For Removal": count})
            
    # Filter dataset: Keep only rows without outliers
    clean_features = z_scores[~rows_with_outliers].reset_index(drop=True)  # Using Scaled Values
    clean_id = id_col[~rows_with_outliers].reset_index(drop=True)
    df_cleaned = pd.concat([clean_id, clean_features], axis=1)
    
    original_len = len(df)
    remaining_len = len(df_cleaned)
    
    report_data.append({"Feature / Detail": "-----------------------------", "Rows Triggered For Removal": "---"})
    report_data.append({"Feature / Detail": "Total Outlier Rows Removed", "Rows Triggered For Removal": total_removed})
    report_data.append({"Feature / Detail": "Original Row Count", "Rows Triggered For Removal": original_len})
    report_data.append({"Feature / Detail": "Remaining Row Count", "Rows Triggered For Removal": remaining_len})
    
    report_df = pd.DataFrame(report_data)
    
    # We rename the sheets similar to Z_Score.py
    base_name = sheet_name.replace("_Encoded", "")
    out_sheet = f"{base_name}_Z-Score"
    rep_sheet = f"{base_name}_Z-Score_Report"
    det_sheet = f"{base_name}_Z-Score_Details"
    
    return {out_sheet: df_cleaned, rep_sheet: report_df, det_sheet: df_full_details}

# === PROCESSING ===
excel_path = r'd:\ML\task\Data.xlsx'
close_excel_file(excel_path)
xls = pd.ExcelFile(excel_path)

input_sheets = ['D1_Encoded', 'D2_Encoded', 'D3_Encoded', 'D4_Encoded']
all_results = {}

for sheet in input_sheets:
    print(f"Processing Z-Score for {sheet}...")
    df = pd.read_excel(xls, sheet_name=sheet)
    results = z_score_processing(df, sheet)
    all_results.update(results)

# === SAVE TO EXCEL ===
print("\nSaving Z-Score reports and standardized data...")
with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    for sheet_name, dataframe in all_results.items():
        dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"[+] Saved '{sheet_name}'")

print("\nProcess Completed Successfully.")
