import pandas as pd
import numpy as np
import os
import win32com.client
from scipy.stats import zscore

# ================== Excel Helpers ==================
def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception as e:
        print("Note: Excel is not running or COM skipped:", e)

def open_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("[*] Opened Excel file:", filepath)
    except Exception as e:
        print("Note: Could not auto-open Excel GUI:", e)

def replace_outliers_zscore_median_mode(df, threshold=3.0):
    df_cleaned = df.copy()
    target_column = 'adaptation_status' if 'adaptation_status' in df_cleaned.columns else df_cleaned.columns[-1]
    
    # Categorical columns in the dataset
    cat_cols_candidates = ['domain_type', 'device_type', 'protocol_type', 'network_mode', 'traffic_state', 'attack_category']
    cat_cols = [c for c in cat_cols_candidates if c in df_cleaned.columns and c != target_column]
    
    # Numeric columns
    num_cols = [c for c in df_cleaned.columns if c not in cat_cols and c != target_column]

    report_data = []
    total_numeric_replaced = 0
    total_categorical_replaced = 0

    # 1. Process Numeric Variables: Replace outliers with Median
    for col in num_cols:
        z_scores = zscore(df_cleaned[col], nan_policy='omit')
        outlier_mask = (z_scores > threshold) | (z_scores < -threshold)
        outlier_count = int(np.sum(outlier_mask))

        if outlier_count > 0:
            median_val = float(df_cleaned.loc[~outlier_mask, col].median())
            df_cleaned.loc[outlier_mask, col] = median_val
            total_numeric_replaced += outlier_count
            print(f"[Numeric] {col}: replaced {outlier_count} outliers with Median ({median_val})")

            report_data.append({
                "Feature": col,
                "Variable Type": "Numeric",
                "Outliers Detected": outlier_count,
                "Treatment Method": "Z-Score + Median",
                "Replacement Value": median_val
            })

    # 2. Process Categorical Variables: Replace outliers with Mode
    for col in cat_cols:
        z_scores = zscore(df_cleaned[col], nan_policy='omit')
        outlier_mask = (z_scores > threshold) | (z_scores < -threshold)
        outlier_count = int(np.sum(outlier_mask))

        if outlier_count > 0:
            mode_val = int(df_cleaned.loc[~outlier_mask, col].mode()[0])
            df_cleaned.loc[outlier_mask, col] = mode_val
            total_categorical_replaced += outlier_count
            print(f"[Categorical] {col}: replaced {outlier_count} outliers with Mode ({mode_val})")

            report_data.append({
                "Feature": col,
                "Variable Type": "Categorical",
                "Outliers Detected": outlier_count,
                "Treatment Method": "Z-Score + Mode",
                "Replacement Value": mode_val
            })

    total_replaced = total_numeric_replaced + total_categorical_replaced
    print(f"\n[+] Total Numeric outliers replaced: {total_numeric_replaced}")
    print(f"[+] Total Categorical outliers replaced: {total_categorical_replaced}")
    print(f"[+] Grand Total outliers replaced: {total_replaced}")

    # Summary row in report
    report_data.append({
        "Feature": "-----------------------------",
        "Variable Type": "---",
        "Outliers Detected": "---",
        "Treatment Method": "---",
        "Replacement Value": "---"
    })
    report_data.append({
        "Feature": "Total Numeric Outliers Replaced",
        "Variable Type": "Numeric",
        "Outliers Detected": total_numeric_replaced,
        "Treatment Method": "Median Imputation",
        "Replacement Value": "-"
    })
    report_data.append({
        "Feature": "Total Categorical Outliers Replaced",
        "Variable Type": "Categorical",
        "Outliers Detected": total_categorical_replaced,
        "Treatment Method": "Mode Imputation",
        "Replacement Value": "-"
    })
    report_data.append({
        "Feature": "Grand Total Outliers Treated",
        "Variable Type": "All Features",
        "Outliers Detected": total_replaced,
        "Treatment Method": "Median/Mode Imputation",
        "Replacement Value": "-"
    })

    report_df = pd.DataFrame(report_data)
    return df_cleaned, report_df

# === CONFIGURATION ===
input_file = r'C:\Users\Sam\Desktop\ML\task\Data.xlsx'
input_sheet = 'Encoded_Data'
output_sheet = 'Z-Score'
report_sheet = 'Z-Score_Report'

# === PROCESSING ===
close_excel_file(input_file)
df = pd.read_excel(input_file, sheet_name=input_sheet)
df_cleaned, report_df = replace_outliers_zscore_median_mode(df, threshold=3.0)

# === SAVE TO EXCEL ===
with pd.ExcelWriter(input_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    df_cleaned.to_excel(writer, sheet_name=output_sheet, index=False)
    df_cleaned.to_excel(writer, sheet_name='Z-Score_Median_Mode', index=False)
    report_df.to_excel(writer, sheet_name=report_sheet, index=False)

print(f"[+] Data successfully saved to '{output_sheet}' and 'Z-Score_Median_Mode'")
print(f"[+] Outlier report saved to '{report_sheet}'")
open_excel_file(input_file)