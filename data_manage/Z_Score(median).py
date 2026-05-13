import pandas as pd
import numpy as np
from scipy.stats import zscore

def replace_outliers_with_median(df):
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns

    report_data = []
    total_replaced = 0

    for col in numeric_cols:

        # Calculate Z-scores
        z_scores = zscore(df_cleaned[col], nan_policy='omit')

        # Create outlier mask
        outlier_mask = (z_scores > 3) | (z_scores < -3)

        # Count outliers
        outlier_count = np.sum(outlier_mask)

        if outlier_count > 0:

            # Median of NON-outlier values
            median_value = df_cleaned.loc[~outlier_mask, col].median()

            # Replace outliers with median
            df_cleaned.loc[outlier_mask, col] = median_value

            total_replaced += outlier_count

            print(f"{col}: replaced {outlier_count} outliers with median ({median_value})")

            report_data.append({
                "Feature / Detail": col,
                "Outliers Replaced": outlier_count,
                "Replacement Value (Median)": median_value
            })

    # Summary
    original_len = len(df_cleaned)

    print(f"\n✅ Total outlier values replaced: {total_replaced}")

    report_data.append({
        "Feature / Detail": "-----------------------------",
        "Outliers Replaced": "---",
        "Replacement Value (Median)": "---"
    })

    report_data.append({
        "Feature / Detail": "Total Outlier Values Replaced",
        "Outliers Replaced": total_replaced,
        "Replacement Value (Median)": "-"
    })

    report_df = pd.DataFrame(report_data)

    return df_cleaned, report_df


# === CONFIGURATION ===
input_file = r'C:\Users\Sam\Desktop\ML\task\Data.xlsx'
input_sheet = 'Encoded_Data'

output_sheet = 'Z-Score_Median'
report_sheet = 'Z-Score_Median_Report'


# === PROCESSING ===
df = pd.read_excel(input_file, sheet_name=input_sheet)

df_cleaned, report_df = replace_outliers_with_median(df)


# === SAVE TO EXCEL ===
with pd.ExcelWriter(
    input_file,
    mode='a',
    engine='openpyxl',
    if_sheet_exists='replace'
) as writer:

    # Save cleaned data
    df_cleaned.to_excel(writer, sheet_name=output_sheet, index=False)

    # Save report
    report_df.to_excel(writer, sheet_name=report_sheet, index=False)


# Optional: Copy cleaned data to clipboard
df_cleaned.to_clipboard(index=False)

print(f"✅ Data saved to '{output_sheet}'")
print(f"✅ Report saved to '{report_sheet}'")