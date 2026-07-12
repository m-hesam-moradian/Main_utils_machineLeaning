import pandas as pd
import numpy as np
from scipy.stats import zscore

def remove_outliers(df):
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    
    # Calculate Z-scores
    z_scores = df_cleaned[numeric_cols].apply(zscore)

    # NEW: Rename Z-score columns so we know what they are, and add them to the dataframe
    z_score_columns = z_scores.add_prefix('Z_Score_')
    df_cleaned = pd.concat([df_cleaned, z_score_columns], axis=1)

    # Create a mask for outliers (True if value is an outlier)
    outlier_mask = (z_scores > 1.8) | (z_scores < -1.8)
    
    # Find rows that have AT LEAST ONE outlier in any column
    rows_with_outliers = outlier_mask.any(axis=1)
    total_removed = rows_with_outliers.sum()

    # NEW: Add a column to explicitly state why a row is kept or removed
    df_cleaned['Outlier_Status'] = np.where(rows_with_outliers, 'Removed (Outlier)', 'Kept')

    # Create a list to store the report data
    report_data = []

    for col in numeric_cols:
        count = outlier_mask[col].sum()
        if count > 0:
            print(f"{col}: triggered removal of {count} rows")
            report_data.append({"Feature / Detail": col, "Rows Triggered For Removal": count})

    # NEW: Save a copy of the FULL dataset (including the ones to be removed) so you can review them
    df_full_details = df_cleaned.copy()

    # Filter dataset: Keep only rows that DO NOT have outliers
    df_cleaned = df_cleaned[~rows_with_outliers]
    
    original_len = len(df)
    remaining_len = len(df_cleaned)

    print(f"\n✅ Total outlier rows removed: {total_removed} (Original: {original_len}, Remaining: {remaining_len})")
    
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
output_sheet = 'Z-Score'                 # Will contain ONLY kept data (with Z-scores attached)
report_sheet = 'Z-Score_Report'          # Will contain the summary
details_sheet = 'Z-Score_Full_Details'   # NEW: Will contain ALL data (Kept + Removed) with Z-scores

# === PROCESSING ===
df = pd.read_excel(input_file, sheet_name=input_sheet)
df_cleaned, report_df, df_full_details = remove_outliers(df)

# === SAVE TO EXCEL ===
with pd.ExcelWriter(input_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    # Save the cleaned data (Only 'Kept' rows)
    df_cleaned.to_excel(writer, sheet_name=output_sheet, index=False)
    
    # Save the full details data (Original rows + Z-Scores + 'Kept'/'Removed' Status)
    df_full_details.to_excel(writer, sheet_name=details_sheet, index=False)
    
    # Save the report
    report_df.to_excel(writer, sheet_name=report_sheet, index=False)

# Optional: Copy only the cleaned data to clipboard
df_cleaned.to_clipboard(index=False)
print(f"✅ Cleaned data saved to '{output_sheet}'")
print(f"✅ Full audit details (with Z-scores) saved to '{details_sheet}'")
print(f"✅ Report saved to '{report_sheet}'")