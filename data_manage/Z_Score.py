import pandas as pd
import numpy as np
import os
import win32com.client
from scipy.stats import zscore

def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("💾 Saved and 🔒 Closed Excel file:", filepath)
                break
    except Exception as e:
        print("Note: Excel is not running or COM skipped:", e)

def open_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("📂 Opened Excel file:", filepath)
    except Exception as e:
        print("Note: Could not auto-open Excel GUI:", e)

def remove_outliers(df, threshold=3.0):
    df_raw = df.copy()
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    
    # Calculate Z-scores
    z_scores = df_raw[numeric_cols].apply(zscore)

    # Rename Z-score columns so we know what they are, and add them to the full audit dataframe
    z_score_columns = z_scores.add_prefix('Z_Score_')
    df_full_details = pd.concat([df_raw, z_score_columns], axis=1)

    # Create a mask for outliers (True if value is an outlier > 3 or < -3)
    outlier_mask = (z_scores > threshold) | (z_scores < -threshold)
    
    # Find rows that have AT LEAST ONE outlier in any column
    rows_with_outliers = outlier_mask.any(axis=1)
    total_removed = rows_with_outliers.sum()

    # Add status to full audit details
    df_full_details['Outlier_Status'] = np.where(rows_with_outliers, 'Removed (Outlier)', 'Kept')

    # Create a list to store the report data
    report_data = []

    for col in numeric_cols:
        count = outlier_mask[col].sum()
        if count > 0:
            print(f"{col}: triggered removal of {count} rows")
            report_data.append({"Feature / Detail": col, "Rows Triggered For Removal": count})

    # Filter dataset: Keep only rows that DO NOT have outliers, preserving clean original columns
    df_cleaned = df_raw[~rows_with_outliers].reset_index(drop=True)
    
    original_len = len(df)
    remaining_len = len(df_cleaned)

    print(f"\n[+] Total outlier rows removed: {total_removed} (Original: {original_len}, Remaining: {remaining_len})")
    
    # Add summary statistics to the bottom of the report
    report_data.append({"Feature / Detail": "-----------------------------", "Rows Triggered For Removal": "---"})
    report_data.append({"Feature / Detail": "Total Outlier Rows Removed", "Rows Triggered For Removal": total_removed})
    report_data.append({"Feature / Detail": "Original Row Count", "Rows Triggered For Removal": original_len})
    report_data.append({"Feature / Detail": "Remaining Row Count", "Rows Triggered For Removal": remaining_len})

    # Convert report list to DataFrame
    report_df = pd.DataFrame(report_data)

    # Return all three DataFrames
    return df_cleaned, report_df, df_full_details

# === CONFIGURATION ===
input_file = r'C:\Users\Sam\Desktop\ML\task\Data.xlsx'
input_sheet = 'Encoded_Data'
output_sheet = 'Z-Score'                 # Contains ONLY kept clean data
report_sheet = 'Z-Score_Report'          # Contains summary table
details_sheet = 'Z-Score_Full_Details'   # Contains ALL data with Z-scores and status

# === PROCESSING ===
close_excel_file(input_file)
df = pd.read_excel(input_file, sheet_name=input_sheet)
df_cleaned, report_df, df_full_details = remove_outliers(df, threshold=3.0)

# === SAVE TO EXCEL ===
with pd.ExcelWriter(input_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    # Save the cleaned data (Only 'Kept' rows with original columns)
    df_cleaned.to_excel(writer, sheet_name=output_sheet, index=False)
    
    # Save the full details data (Original rows + Z-Scores + 'Kept'/'Removed' Status)
    df_full_details.to_excel(writer, sheet_name=details_sheet, index=False)
    
    # Save the report
    report_df.to_excel(writer, sheet_name=report_sheet, index=False)

open_excel_file(input_file)

# Optional: Copy only the cleaned data to clipboard
df_cleaned.to_clipboard(index=False)
print(f"[+] Cleaned data saved to '{output_sheet}'")
print(f"[+] Full audit details (with Z-scores) saved to '{details_sheet}'")
print(f"[+] Report saved to '{report_sheet}'")
