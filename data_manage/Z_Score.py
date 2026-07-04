import pandas as pd
import numpy as np
from scipy.stats import zscore

def remove_outliers(df):
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    
    # Calculate Z-scores
    z_scores = df_cleaned[numeric_cols].apply(zscore)

    # Create a mask for outliers (True if value is an outlier)
    outlier_mask = (z_scores > 1.8) | (z_scores < -1.8)
    
    # Find rows that have AT LEAST ONE outlier in any column
    rows_with_outliers = outlier_mask.any(axis=1)
    total_removed = rows_with_outliers.sum()

    # Create a list to store the report data
    report_data = []

    for col in numeric_cols:
        count = outlier_mask[col].sum()
        if count > 0:
            print(f"{col}: triggered removal of {count} rows")
            report_data.append({"Feature / Detail": col, "Rows Triggered For Removal": count})

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

    return df_cleaned, report_df

# === CONFIGURATION ===
input_file = r'C:\Users\Sam\Desktop\ML\task\Data.xlsx'
input_sheet = 'Encoded_Data'
output_sheet = 'Z-Score'
report_sheet = 'Z-Score_Report'  # Name of the new sheet for the report

# === PROCESSING ===
df = pd.read_excel(input_file, sheet_name=input_sheet)
df_cleaned, report_df = remove_outliers(df)

# === SAVE TO EXCEL ===
with pd.ExcelWriter(input_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    # Save the cleaned data
    df_cleaned.to_excel(writer, sheet_name=output_sheet, index=False)
    # Save the report
    report_df.to_excel(writer, sheet_name=report_sheet, index=False)

# Optional: Copy only the cleaned data to clipboard
df_cleaned.to_clipboard(index=False)
print(f"✅ Data saved to '{output_sheet}' and Report saved to '{report_sheet}'")