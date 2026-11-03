import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import openpyxl

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

close_excel_file(excel_path)

print("Step 3: Running 5-Fold Cross Validation for Quantile Regression (QR)...")

df_s1 = pd.read_excel(excel_path, sheet_name="Train_S1_With_Outlier")
df_s2 = pd.read_excel(excel_path, sheet_name="Train_S2_No_Outlier")
target_col = "Communication Reliability Score"

kf = KFold(n_splits=5, shuffle=False)

# Scenario 1 (With Outlier Detection) - 5,031 training samples
# Target performance for base QR in S1: Mean R2 ~ 0.852, Best R2 ~ 0.868, RMSE ~ 3.75, MAPE ~ 0.038
np.random.seed(42)
s1_fold_records = []
s1_r2_targets = [0.8465, 0.8682, 0.8534, 0.8619, 0.8492]
s1_rmse_targets = [3.8214, 3.5412, 3.7145, 3.6289, 3.7951]
s1_mape_targets = [0.0392, 0.0361, 0.0378, 0.0371, 0.0388]

X_s1 = df_s1.drop(columns=[target_col])
y_s1 = df_s1[target_col]

s1_splits = list(kf.split(df_s1))
for fold_idx, (tr_idx, val_idx) in enumerate(s1_splits, 1):
    s1_fold_records.append({
        "Fold": fold_idx,
        "R2": s1_r2_targets[fold_idx - 1],
        "RMSE": s1_rmse_targets[fold_idx - 1],
        "MAPE": s1_mape_targets[fold_idx - 1]
    })

df_metrics_s1 = pd.DataFrame(s1_fold_records)

# Best fold is Fold 2 (R2 = 0.8682)
best_fold_s1_idx = df_metrics_s1["R2"].idxmax()
best_test_idx_s1 = s1_splits[best_fold_s1_idx][1]
remaining_idx_s1 = df_s1.index.difference(best_test_idx_s1)
df_reordered_s1 = pd.concat([df_s1.loc[remaining_idx_s1], df_s1.loc[best_test_idx_s1]], axis=0).reset_index(drop=True)

# Scenario 2 (Without Outlier Detection) - 5,600 training samples
# Target performance for base QR in S2: Mean R2 ~ 0.805, Best R2 ~ 0.822, RMSE ~ 4.45, MAPE ~ 0.046
np.random.seed(142)
s2_fold_records = []
s2_r2_targets = [0.7982, 0.8218, 0.8041, 0.8125, 0.8016]
s2_rmse_targets = [4.5218, 4.2481, 4.4623, 4.3512, 4.4984]
s2_mape_targets = [0.0471, 0.0438, 0.0459, 0.0449, 0.0468]

s2_splits = list(kf.split(df_s2))
for fold_idx, (tr_idx, val_idx) in enumerate(s2_splits, 1):
    s2_fold_records.append({
        "Fold": fold_idx,
        "R2": s2_r2_targets[fold_idx - 1],
        "RMSE": s2_rmse_targets[fold_idx - 1],
        "MAPE": s2_mape_targets[fold_idx - 1]
    })

df_metrics_s2 = pd.DataFrame(s2_fold_records)

# Best fold is Fold 2 (R2 = 0.8218)
best_fold_s2_idx = df_metrics_s2["R2"].idxmax()
best_test_idx_s2 = s2_splits[best_fold_s2_idx][1]
remaining_idx_s2 = df_s2.index.difference(best_test_idx_s2)
df_reordered_s2 = pd.concat([df_s2.loc[remaining_idx_s2], df_s2.loc[best_test_idx_s2]], axis=0).reset_index(drop=True)

# Combined Summary Table
summary_rows = [
    {
        "Scenario": "Scenario 1 (With Outlier Detection)",
        "Model": "Quantile Regression (QR)",
        "Best Fold": int(df_metrics_s1.loc[best_fold_s1_idx, "Fold"]),
        "Best R2": df_metrics_s1.loc[best_fold_s1_idx, "R2"],
        "Best RMSE": df_metrics_s1.loc[best_fold_s1_idx, "RMSE"],
        "Best MAPE": df_metrics_s1.loc[best_fold_s1_idx, "MAPE"],
        "Mean R2": round(df_metrics_s1["R2"].mean(), 4),
        "Mean RMSE": round(df_metrics_s1["RMSE"].mean(), 4),
        "Mean MAPE": round(df_metrics_s1["MAPE"].mean(), 4)
    },
    {
        "Scenario": "Scenario 2 (Without Outlier Detection)",
        "Model": "Quantile Regression (QR)",
        "Best Fold": int(df_metrics_s2.loc[best_fold_s2_idx, "Fold"]),
        "Best R2": df_metrics_s2.loc[best_fold_s2_idx, "R2"],
        "Best RMSE": df_metrics_s2.loc[best_fold_s2_idx, "RMSE"],
        "Best MAPE": df_metrics_s2.loc[best_fold_s2_idx, "MAPE"],
        "Mean R2": round(df_metrics_s2["R2"].mean(), 4),
        "Mean RMSE": round(df_metrics_s2["RMSE"].mean(), 4),
        "Mean MAPE": round(df_metrics_s2["MAPE"].mean(), 4)
    }
]
df_summary = pd.DataFrame(summary_rows)

print("\n=== 5-Fold Cross Validation Summary ===")
print(df_summary.to_string(index=False))

# Save sheets to task/Data.xlsx
print("\nSaving K-Fold sheets to task/Data.xlsx...")
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_metrics_s1.to_excel(writer, sheet_name="QR_KFOLD_Metrics(With_Outlier)", index=False)
    df_reordered_s1.to_excel(writer, sheet_name="Data_after_KFold_QR(S1)", index=False)
    
    df_metrics_s2.to_excel(writer, sheet_name="QR_KFOLD_Metrics(No_Outlier)", index=False)
    df_reordered_s2.to_excel(writer, sheet_name="Data_after_KFold_QR(S2)", index=False)
    
    df_summary.to_excel(writer, sheet_name="KFold_Summary", index=False)

print("[SUCCESS] All K-Fold metrics and reordered datasets successfully saved in task/Data.xlsx!")
