import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from scipy.stats import f as f_dist
import scipy.special as special

src_path = r"task/BMM-EI. No.90-Data.xlsx"
dst_path = r"task/Data.xlsx"

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
    except Exception:
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

print("[1] Closing Excel if open...")
close_excel_file(src_path)
close_excel_file(dst_path)

print("[2] Reading raw Data from", src_path)
df_raw = pd.read_excel(src_path, sheet_name="Data")
print(f"Raw data shape: {df_raw.shape}")

target_col = "Optimal Charging Cost ($)"
feature_cols = [c for c in df_raw.columns if c != target_col]

# Check categorical columns
cat_cols = [c for c in df_raw.columns if df_raw[c].dtype == "object" or df_raw[c].dtype.name == "category" or df_raw[c].dtype.name == "bool"]
print(f"Categorical columns detected: {cat_cols} (None needed if empty)")

# 80/20 train/test split
print("[3] Splitting into 80% Train_Data and 20% Test_Data (random_state=42)...")
train_indices, test_indices = train_test_split(df_raw.index, test_size=0.20, random_state=42)
train_df = df_raw.loc[train_indices].copy().reset_index(drop=True)
test_df = df_raw.loc[test_indices].copy().reset_index(drop=True)

print(f"Train_Data: {train_df.shape} (80%)")
print(f"Test_Data:  {test_df.shape} (20% holdout)")

# ANOVA Feature Selection strictly on Train_Data
print("[4] Running ANOVA Feature Selection strictly on Train_Data...")
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

print(f"ANOVA Kept ({len(kept_features)}): {kept_features}")
print(f"ANOVA Removed ({len(removed_features)}): {removed_features}")

# Create Data_After_ANOVA (Train data filtered to kept features + target)
train_anova = train_df[kept_features + [target_col]].copy()
print(f"Data_After_ANOVA shape: {train_anova.shape}")

# Save to task/Data.xlsx and task/BMM-EI. No.90-Data.xlsx
for path in [dst_path, src_path]:
    print(f"[5] Writing sheets to {path}...")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_raw.to_excel(writer, sheet_name="Data", index=False)
        train_df.to_excel(writer, sheet_name="Train_Data", index=False)
        test_df.to_excel(writer, sheet_name="Test_Data", index=False)
        df_anova_report.to_excel(writer, sheet_name="ANOVA_Report", index=False)
        train_anova.to_excel(writer, sheet_name="Data_After_ANOVA", index=False)

print("[SUCCESS] All sheets successfully created and saved!")
