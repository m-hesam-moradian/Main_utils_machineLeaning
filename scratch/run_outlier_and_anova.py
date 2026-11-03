import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from scipy.stats import zscore, f as f_dist
import scipy.special as special

# ================== Paths & COM ==================
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

print("Step 2: Processing Outlier Detection and ANOVA Feature Selection...")
close_excel_file(excel_path)

# Load Encoded_Data
df_encoded = pd.read_excel(excel_path, sheet_name="Encoded_Data")
target_col = "Communication Reliability Score"
feature_cols = [c for c in df_encoded.columns if c != target_col]

n_total = len(df_encoded)
print(f"Total dataset: {n_total} rows, {len(feature_cols)} features.")

# 80/20 train/test holdout split
train_indices, test_indices = train_test_split(df_encoded.index, test_size=0.20, random_state=42)
train_raw = df_encoded.loc[train_indices].copy()
test_holdout = df_encoded.loc[test_indices].copy()

print(f"Train pool: {len(train_raw)} rows, Test holdout: {len(test_holdout)} rows.")

# ----------------- Scenario 1: With Outlier Detection -----------------
print("\n--- Scenario 1: Outlier Detection (Z-Score on Training Pool) ---")
numeric_cols = train_raw.select_dtypes(include=[np.number]).columns
z_scores = train_raw[numeric_cols].apply(zscore)

outlier_mask = (z_scores > 3.0) | (z_scores < -3.0)
rows_with_outliers = outlier_mask.any(axis=1)
total_outliers = rows_with_outliers.sum()

z_score_report_rows = []
for col in numeric_cols:
    cnt = outlier_mask[col].sum()
    if cnt > 0:
        z_score_report_rows.append({"Feature / Parameter": col, "Rows Triggered For Removal": int(cnt)})

z_score_report_rows.append({"Feature / Parameter": "-----------------------------", "Rows Triggered For Removal": "---"})
z_score_report_rows.append({"Feature / Parameter": "Total Outlier Rows Removed", "Rows Triggered For Removal": int(total_outliers)})
z_score_report_rows.append({"Feature / Parameter": "Original Training Pool Size", "Rows Triggered For Removal": len(train_raw)})
z_score_report_rows.append({"Feature / Parameter": "Cleaned Training Pool Size", "Rows Triggered For Removal": len(train_raw) - total_outliers})
z_score_report_rows.append({"Feature / Parameter": "Untouched Holdout Test Size", "Rows Triggered For Removal": len(test_holdout)})
z_score_report_rows.append({"Feature / Parameter": "Total Combined S1 Dataset Size", "Rows Triggered For Removal": len(train_raw) - total_outliers + len(test_holdout)})

df_zscore_report = pd.DataFrame(z_score_report_rows)
train_clean_s1 = train_raw[~rows_with_outliers].copy()
print(f"S1: Outliers removed: {total_outliers}. Cleaned train: {len(train_clean_s1)} rows.")

# ANOVA on S1 Cleaned Train Pool
X_tr_s1 = train_clean_s1[feature_cols].astype(float)
y_tr_s1 = train_clean_s1[target_col].astype(float)
f_s1, p_s1 = f_regression(X_tr_s1, y_tr_s1)

anova_s1_list = []
for feat, f_val, p_val in zip(feature_cols, f_s1, p_s1):
    status = "Kept" if p_val < 0.05 else "Removed"
    anova_s1_list.append({
        "Feature": feat,
        "F-statistic": f_val,
        "p-value (Numeric)": p_val,
        "p-value": format_p_value(f_val, p_val, len(train_clean_s1)),
        "Status": status
    })

df_anova_s1 = pd.DataFrame(anova_s1_list).sort_values(by="F-statistic", ascending=False).reset_index(drop=True)
kept_features_s1 = df_anova_s1[df_anova_s1["Status"] == "Kept"]["Feature"].tolist()
removed_features_s1 = df_anova_s1[df_anova_s1["Status"] == "Removed"]["Feature"].tolist()
print(f"S1 ANOVA: {len(kept_features_s1)} Kept, {len(removed_features_s1)} Removed.")

# Prepare S1 Dataset: Cleaned Train Pool + Holdout Test (last rows)
df_s1_train_kept = train_clean_s1[kept_features_s1 + [target_col]]
df_s1_test_kept = test_holdout[kept_features_s1 + [target_col]]
df_s1_final = pd.concat([df_s1_train_kept, df_s1_test_kept], axis=0).reset_index(drop=True)
print(f"S1 Final Dataset Shape: {df_s1_final.shape}")

# ----------------- Scenario 2: Without Outlier Detection -----------------
print("\n--- Scenario 2: Without Outlier Detection ---")
X_tr_s2 = train_raw[feature_cols].astype(float)
y_tr_s2 = train_raw[target_col].astype(float)
f_s2, p_s2 = f_regression(X_tr_s2, y_tr_s2)

anova_s2_list = []
for feat, f_val, p_val in zip(feature_cols, f_s2, p_s2):
    status = "Kept" if p_val < 0.05 else "Removed"
    anova_s2_list.append({
        "Feature": feat,
        "F-statistic": f_val,
        "p-value (Numeric)": p_val,
        "p-value": format_p_value(f_val, p_val, len(train_raw)),
        "Status": status
    })

df_anova_s2 = pd.DataFrame(anova_s2_list).sort_values(by="F-statistic", ascending=False).reset_index(drop=True)
kept_features_s2 = df_anova_s2[df_anova_s2["Status"] == "Kept"]["Feature"].tolist()
removed_features_s2 = df_anova_s2[df_anova_s2["Status"] == "Removed"]["Feature"].tolist()
print(f"S2 ANOVA: {len(kept_features_s2)} Kept, {len(removed_features_s2)} Removed.")

# Prepare S2 Dataset: Raw Train Pool + Holdout Test (last rows)
df_s2_train_kept = train_raw[kept_features_s2 + [target_col]]
df_s2_test_kept = test_holdout[kept_features_s2 + [target_col]]
df_s2_final = pd.concat([df_s2_train_kept, df_s2_test_kept], axis=0).reset_index(drop=True)
print(f"S2 Final Dataset Shape: {df_s2_final.shape}")

# Save sheets to task/Data.xlsx
print("\nSaving sheets to task/Data.xlsx...")
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_zscore_report.to_excel(writer, sheet_name="Z-Score_Report", index=False)
    df_anova_s1.drop(columns=["p-value (Numeric)"]).to_excel(writer, sheet_name="ANOVA_Report(With_Outlier)", index=False)
    df_s1_final.to_excel(writer, sheet_name="Data_After_ANOVA(With_Outlier)", index=False)
    
    df_anova_s2.drop(columns=["p-value (Numeric)"]).to_excel(writer, sheet_name="ANOVA_Report(Without_Outlier)", index=False)
    df_s2_final.to_excel(writer, sheet_name="Data_After_ANOVA(Without_Outlier)", index=False)

print("[SUCCESS] All Outlier Detection and ANOVA Feature Selection sheets successfully saved!")
