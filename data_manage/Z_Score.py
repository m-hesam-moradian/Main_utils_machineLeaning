import pandas as pd
import numpy as np
from scipy.stats import zscore

def replace_outliers_with_median(df):
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    z_scores = df_cleaned[numeric_cols].apply(zscore)

    for col in numeric_cols:
        outliers = (z_scores[col] > 3) | (z_scores[col] < -3)
        if outliers.any():
            median_val = df_cleaned[col].median()
            df_cleaned.loc[outliers, col] = median_val

    return df_cleaned

# === CONFIGURATION ===
input_file = 'your_file.xlsx'
input_sheet = 'RawData'
output_sheet = 'CleanedData'

# === PROCESSING ===
df = pd.read_excel(input_file, sheet_name=input_sheet)
df_cleaned = replace_outliers_with_median(df)

# === SAVE TO NEW SHEET ===
with pd.ExcelWriter(input_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    df_cleaned.to_excel(writer, sheet_name=output_sheet, index=False)