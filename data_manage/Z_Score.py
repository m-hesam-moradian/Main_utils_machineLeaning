import pandas as pd
import numpy as np
from scipy.stats import zscore

def replace_outliers_with_median(df):
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    z_scores = df_cleaned[numeric_cols].apply(zscore)

    total_replacements = 0

    for col in numeric_cols:
        outliers = (z_scores[col] > 3) | (z_scores[col] < -3)
        count = outliers.sum()
        total_replacements += count

        if count > 0:
            median_val = df_cleaned[col].median()
            df_cleaned.loc[outliers, col] = median_val
            print(f"{col}: replaced {count} outliers with median")

    print(f"\n✅ Total outliers replaced: {total_replacements}")
    return df_cleaned

# === CONFIGURATION ===
input_file = r'C:\Users\Sam\Desktop\ML\task\BMM-EI. No.25-Data.xlsx'
input_sheet = 'Data'
# output_sheet = 'Z-Score'

# === PROCESSING ===
df = pd.read_excel(input_file, sheet_name=input_sheet)
df_cleaned = replace_outliers_with_median(df)

# === SAVE TO NEW SHEET ===
# with pd.ExcelWriter(input_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
#     df_cleaned.to_excel(writer, sheet_name=output_sheet, index=False)

df_cleaned.to_clipboard(index=False)