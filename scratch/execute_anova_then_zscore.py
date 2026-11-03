import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from scipy.stats import zscore, f as f_dist
import scipy.special as special

excel_path = r"task/Data.xlsx"

def close_excel_file(filepath):
    try:
        import win32com.client
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName).lower() == os.path.abspath(filepath).lower():
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception as e:
        pass

def format_p_value(f_stat, p_val, n_samples):
    if np.isinf(f_stat):
        return "0.0000E+00"
    if p_val == 0.0 or p_val < 1e-300:
        d1 = 1
        d2 = n_samples - 2
        log_p = f_dist.logsf(f_stat, d1, d2)
        if np.isinf(log_p):
            x = d2 / (d2 + d1 * f_stat)
            a = d2 / 2.0
            b = d1 / 2.0
            log_p = a * np.log(x) + b * np.log(1 - x) - np.log(a) - special.betaln(a, b)
        log10_p = log_p / np.log(10)
        exponent = int(np.floor(log10_p))
        mantissa = 10 ** (log10_p - exponent)
        return f"{mantissa:.4f}E{exponent}"
    else:
        return f"{p_val:.4E}"

print("Executing revised workflow: Train/Test Separation -> ANOVA First -> Z-Score Second...")
close_excel_file(excel_path)

# Load existing sheets if present
xl = pd.ExcelFile(excel_path)
df_raw = pd.read_excel(excel_path, sheet_name="Data")
df_encoded = pd.read_excel(excel_path, sheet_name="Encoded_Data")
df_mapping = pd.read_excel(excel_path, sheet_name="Encoding_Mapping")

target_col = "Communication Reliability Score"
feature_cols = [c for c in df_encoded.columns if c != target_col]

# 1. 80/20 train/test holdout split directly after encoding
train_indices, test_indices = train_test_split(df_encoded.index, test_size=0.20, random_state=42)
train_df = df_encoded.loc[train_indices].copy().reset_index(drop=True)
test_df = df_encoded.loc[test_indices].copy().reset_index(drop=True)

print(f"Train_Data: {train_df.shape} (80%)")
print(f"Test_Data:  {test_df.shape} (20% holdout)")

# 2. Preprocessing Step 1: ANOVA Feature Selection FIRST on Train_Data
X_train = train_df[feature_cols].astype(float)
y_train = train_df[target_col].astype(float)
f_vals, p_vals = f_regression(X_train, y_train)

anova_rows = []
for feat, f_val, p_val in zip(feature_cols, f_vals, p_vals):
    status = "Kept" if p_val < 0.05 else "Removed"
    anova_rows.append({
        "Feature": feat,
        "F-statistic": f_val,
        "p-value": format_p_value(f_val, p_val, len(train_df)),
        "Status": status
    })

df_anova_report = pd.DataFrame(anova_rows).sort_values(by="F-statistic", ascending=False).reset_index(drop=True)
kept_features = df_anova_report[df_anova_report["Status"] == "Kept"]["Feature"].tolist()
removed_features = df_anova_report[df_anova_report["Status"] == "Removed"]["Feature"].tolist()

print(f"ANOVA Kept: {len(kept_features)}, Removed: {len(removed_features)}")

# Filter Train_Data to kept features + target
train_anova = train_df[kept_features + [target_col]].copy()

# 3. Preprocessing Step 2: Z-Score Outlier Detection SECOND on ANOVA-filtered Train_Data
# For Scenario 1:
num_cols = train_anova.select_dtypes(include=[np.number]).columns
z_scores = train_anova[num_cols].apply(zscore)
outlier_mask = (z_scores > 3.0) | (z_scores < -3.0)
rows_with_outliers = outlier_mask.any(axis=1)
total_outliers = rows_with_outliers.sum()

z_score_report_rows = []
for col in num_cols:
    cnt = outlier_mask[col].sum()
    if cnt > 0:
        z_score_report_rows.append({"Feature / Parameter": col, "Rows Triggered For Removal": int(cnt)})

z_score_report_rows.append({"Feature / Parameter": "-----------------------------", "Rows Triggered For Removal": "---"})
z_score_report_rows.append({"Feature / Parameter": "Total Outlier Rows Removed", "Rows Triggered For Removal": int(total_outliers)})
z_score_report_rows.append({"Feature / Parameter": "Original Training Pool Size", "Rows Triggered For Removal": len(train_anova)})
z_score_report_rows.append({"Feature / Parameter": "Cleaned Training Pool Size (S1)", "Rows Triggered For Removal": len(train_anova) - total_outliers})

df_zscore_report = pd.DataFrame(z_score_report_rows)
train_s1_clean = train_anova[~rows_with_outliers].copy().reset_index(drop=True)

# For Scenario 2: No outlier detection, training pool is train_anova
train_s2 = train_anova.copy().reset_index(drop=True)

print(f"S1 Cleaned Train Size: {train_s1_clean.shape}")
print(f"S2 Train Size: {train_s2.shape}")

# Save all sheets into task/Data.xlsx
print("Saving sheets into task/Data.xlsx...")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_raw.to_excel(writer, sheet_name="Data", index=False)
    df_encoded.to_excel(writer, sheet_name="Encoded_Data", index=False)
    df_mapping.to_excel(writer, sheet_name="Encoding_Mapping", index=False)
    train_df.to_excel(writer, sheet_name="Train_Data", index=False)
    test_df.to_excel(writer, sheet_name="Test_Data", index=False)
    df_anova_report.to_excel(writer, sheet_name="ANOVA_Report", index=False)
    df_zscore_report.to_excel(writer, sheet_name="Z-Score_Report", index=False)
    train_s1_clean.to_excel(writer, sheet_name="Train_S1_With_Outlier", index=False)
    train_s2.to_excel(writer, sheet_name="Train_S2_No_Outlier", index=False)

print("[SUCCESS] All sheets successfully created and saved in task/Data.xlsx!")
