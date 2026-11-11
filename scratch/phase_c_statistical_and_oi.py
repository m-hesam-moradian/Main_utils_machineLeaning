import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations
from sklearn.metrics import r2_score, root_mean_squared_error, auc

excel_path = r"task/Data.xlsx"
alt_excel_path = r"task/BMM-EI. No.90-Data.xlsx"

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

print("[1] Closing Excel if open...")
close_excel_file(excel_path)
close_excel_file(alt_excel_path)

# Load all 8 models from data/model{N}.npt
models = {
    "RFR": pd.read_csv("data/model1.npt", sep="\t"),
    "RFR + DCOA": pd.read_csv("data/model2.npt", sep="\t"),
    "RFR + EOA": pd.read_csv("data/model3.npt", sep="\t"),
    "RFR + SWOA": pd.read_csv("data/model4.npt", sep="\t"),
    "GPR": pd.read_csv("data/model5.npt", sep="\t"),
    "GPR + DCOA": pd.read_csv("data/model6.npt", sep="\t"),
    "GPR + EOA": pd.read_csv("data/model7.npt", sep="\t"),
    "GPR + SWOA": pd.read_csv("data/model8.npt", sep="\t"),
}

split_idx = 800
y_real_all = models["RFR"]["y_real"].values
y_real_train = y_real_all[:split_idx]
y_real_test = y_real_all[split_idx:]

# ----------------- 1. Prediction Sheet (predicts) -----------------
print("[2] Building predicts sheet...")
df_predicts = pd.DataFrame({"y_real": y_real_all})
for name, m_df in models.items():
    df_predicts[f"{name}_Pred"] = m_df["y_pred"].values

# ----------------- 2. Overfitting Index (OI) -----------------
print("[3] Calculating Overfitting Index for all 8 models...")
oi_records = []
for name, m_df in models.items():
    pred_all = m_df["y_pred"].values
    pred_train = pred_all[:split_idx]
    pred_test = pred_all[split_idx:]

    r2_tr = r2_score(y_real_train, pred_train)
    r2_te = r2_score(y_real_test, pred_test)
    rmse_tr = root_mean_squared_error(y_real_train, pred_train)
    rmse_te = root_mean_squared_error(y_real_test, pred_test)

    oi_r2 = abs((r2_tr - r2_te) / r2_tr) * 100.0
    oi_rmse = abs((rmse_te - rmse_tr) / rmse_tr) * 100.0

    status = "Optimal (< 5% OI)" if oi_r2 < 5.0 and oi_rmse < 5.0 else "Acceptable"

    oi_records.append({
        "Model Variant": name,
        "Train R2": round(r2_tr, 4),
        "Test R2": round(r2_te, 4),
        "OI (R2) (%)": round(oi_r2, 4),
        "Train RMSE": round(rmse_tr, 4),
        "Test RMSE": round(rmse_te, 4),
        "OI (RMSE) (%)": round(oi_rmse, 4),
        "Generalization Status": status
    })

df_oi = pd.DataFrame(oi_records)
print("\nOverfitting Index Table:")
print(df_oi.to_string(index=False))

# ----------------- 3. Wilcoxon Test (Comparing Style) -----------------
print("\n[4] Computing Wilcoxon Signed-Rank Test (Comparing Style)...")
test_errors = {}
for name, m_df in models.items():
    pred_test = m_df["y_pred"].values[split_idx:]
    test_errors[name] = np.abs(y_real_test - pred_test)

wilcoxon_records = []
for m_a, m_b in combinations(models.keys(), 2):
    err_a = test_errors[m_a]
    err_b = test_errors[m_b]

    diff = err_a - err_b
    if np.all(diff == 0):
        stat, p_val = 0.0, 1.0
    else:
        stat, p_val = wilcoxon(err_a, err_b)

    def fmt_p(val):
        if val < 0.0001:
            return f"{val:.4e}"
        else:
            return f"{val:.4f}"

    sig = "Yes (p < 0.05)" if p_val < 0.05 else "No (p >= 0.05)"
    superior = m_a if np.mean(err_a) < np.mean(err_b) else m_b

    wilcoxon_records.append({
        "Comparison Pair": f"{m_a} vs {m_b}",
        "Statistic (W)": round(stat, 2),
        "p-value": fmt_p(p_val),
        "Statistically Significant": sig,
        "Superior Model (Lower Error)": superior
    })

df_wilcoxon = pd.DataFrame(wilcoxon_records)
print("\nWilcoxon Test Comparing Style Summary:")
print(df_wilcoxon.to_string(index=False))

# Build Comparing Style Matrix
model_names_list = list(models.keys())
n_models = len(model_names_list)
matrix_data = np.full((n_models, n_models), "-", dtype=object)

for i, m_a in enumerate(model_names_list):
    for j, m_b in enumerate(model_names_list):
        if i == j:
            matrix_data[i, j] = "1.0000"
        elif i < j:
            stat, p_val = wilcoxon(test_errors[m_a], test_errors[m_b])
            star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            matrix_data[i, j] = f"{p_val:.4f}{star}" if p_val >= 0.0001 else f"{p_val:.2e}{star}"
        else:
            matrix_data[i, j] = matrix_data[j, i]

df_wilcoxon_matrix = pd.DataFrame(matrix_data, index=model_names_list, columns=model_names_list).reset_index()
df_wilcoxon_matrix.rename(columns={"index": "Model Variant"}, inplace=True)

# ----------------- 4. REC Curve Comparison Summary -----------------
print("\n[5] Building REC Curve Summary...")
rec_summary_rows = []
for name, m_df in models.items():
    pred_test = m_df["y_pred"].values[split_idx:]
    err = np.abs(y_real_test - pred_test)
    eps = np.linspace(0, err.max(), 200)
    acc = [np.mean(err <= e) for e in eps]
    rec_auc_val = auc(eps, acc)

    rec_summary_rows.append({
        "Model Variant": name,
        "Max Error": round(err.max(), 4),
        "Median Error": round(np.median(err), 4),
        "Mean Error": round(np.mean(err), 4),
        "AUC-REC": round(rec_auc_val, 4)
    })

df_rec_summary = pd.DataFrame(rec_summary_rows).sort_values(by="AUC-REC", ascending=False).reset_index(drop=True)
df_rec_summary["Rank"] = df_rec_summary.index + 1
df_rec_summary = df_rec_summary[["Rank", "Model Variant", "AUC-REC", "Mean Error", "Median Error", "Max Error"]]
print("\nREC Curve AUC Summary:")
print(df_rec_summary.to_string(index=False))

# ----------------- 5. Save all sheets to Excel -----------------
print("\n[6] Writing sheets to Excel...")
for path in [excel_path, alt_excel_path]:
    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_predicts.to_excel(writer, sheet_name="predicts", index=False)
        df_oi.to_excel(writer, sheet_name="Overfitting_Index", index=False)
        df_wilcoxon.to_excel(writer, sheet_name="Wilcoxon_Test", index=False)
        df_wilcoxon_matrix.to_excel(writer, sheet_name="Wilcoxon_Matrix", index=False)
        df_rec_summary.to_excel(writer, sheet_name="REC_Summary", index=False)

print("[SUCCESS] Phase C sheets (predicts, Overfitting_Index, Wilcoxon_Test, Wilcoxon_Matrix, REC_Summary) saved successfully!")
