import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

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
print("Step 6: Processing Prediction Catching, Wilcoxon Test, Overfitting Index, and Confidence Intervals...")

# Load models predictions
m1 = pd.read_csv("data/model1.npt", sep="\t")
m2 = pd.read_csv("data/model2.npt", sep="\t")
m3 = pd.read_csv("data/model3.npt", sep="\t")
m4 = pd.read_csv("data/model4.npt", sep="\t")
m5 = pd.read_csv("data/model5.npt", sep="\t")
m6 = pd.read_csv("data/model6.npt", sep="\t")

models = {
    "QR(With_Outlier)": m1,
    "QR(With_Outlier) + NOA": m2,
    "QR(With_Outlier) + SDOA": m3,
    "QR(Without_Outlier)": m4,
    "QR(Without_Outlier) + NOA": m5,
    "QR(Without_Outlier) + SDOA": m6,
}

# ----------------- 1. Prediction Sheets -----------------
# S1 predictions (6,431 rows)
df_pred_s1 = pd.DataFrame()
for name in ["QR(With_Outlier)", "QR(With_Outlier) + NOA", "QR(With_Outlier) + SDOA"]:
    df_pred_s1[f"{name}_Real"] = models[name]["y_real"]
    df_pred_s1[f"{name}_Pred"] = models[name]["y_pred"]

# S2 predictions (7,000 rows)
df_pred_s2 = pd.DataFrame()
for name in ["QR(Without_Outlier)", "QR(Without_Outlier) + NOA", "QR(Without_Outlier) + SDOA"]:
    df_pred_s2[f"{name}_Real"] = models[name]["y_real"]
    df_pred_s2[f"{name}_Pred"] = models[name]["y_pred"]

# Combined Holdout Test predictions (1,400 rows)
df_pred_test = pd.DataFrame()
for name in models:
    df_pred_test[f"{name}_Real"] = models[name]["y_real"].values[-1400:]
    df_pred_test[f"{name}_Pred"] = models[name]["y_pred"].values[-1400:]

# ----------------- 2. Overfitting Index -----------------
oi_records = []
for name, df in models.items():
    split_idx = 5031 if "With_Outlier" in name else 5600
    y_tr_real = df["y_real"].values[:split_idx]
    y_tr_pred = df["y_pred"].values[:split_idx]
    y_te_real = df["y_real"].values[split_idx:]
    y_te_pred = df["y_pred"].values[split_idx:]
    
    r2_tr = r2_score(y_tr_real, y_tr_pred)
    r2_te = r2_score(y_te_real, y_te_pred)
    rmse_tr = np.sqrt(mean_squared_error(y_tr_real, y_tr_pred))
    rmse_te = np.sqrt(mean_squared_error(y_te_real, y_te_pred))
    
    oi_r2 = abs((r2_tr - r2_te) / r2_tr) * 100.0
    oi_rmse = abs((rmse_te - rmse_tr) / rmse_tr) * 100.0
    
    oi_records.append({
        "Model Variant": name,
        "Train R2": round(r2_tr, 4),
        "Test R2": round(r2_te, 4),
        "OI (R2) (%)": round(oi_r2, 4),
        "Train RMSE": round(rmse_tr, 4),
        "Test RMSE": round(rmse_te, 4),
        "OI (RMSE) (%)": round(oi_rmse, 4),
        "Generalization Status": "Optimal (< 5% OI)" if oi_r2 < 5.0 else "Acceptable"
    })
df_oi = pd.DataFrame(oi_records)

# ----------------- 3. Wilcoxon Signed-Rank Test -----------------
test_preds = {name: df["y_pred"].values[-1400:] for name, df in models.items()}
y_true_test = models["QR(With_Outlier)"]["y_real"].values[-1400:]
wilcoxon_records = []

for m_a, m_b in combinations(models.keys(), 2):
    p_a = test_preds[m_a]
    p_b = test_preds[m_b]
    
    err_a = np.abs(y_true_test - p_a)
    err_b = np.abs(y_true_test - p_b)
    
    stat_pred, p_pred = wilcoxon(p_a, p_b)
    stat_err, p_err = wilcoxon(err_a, err_b)
    
    def fmt_p(val):
        return f"{val:.4e}" if val < 0.001 else f"{val:.4f}"
    
    wilcoxon_records.append({
        "Comparison Pair": f"{m_a} vs {m_b}",
        "W-Statistic (Error)": round(stat_err, 2),
        "p-value (Error Difference)": fmt_p(p_err),
        "Significance (alpha=0.05)": "Statistically Significant" if p_err < 0.05 else "Not Significant",
        "Superior Model": m_a if np.mean(err_a) < np.mean(err_b) else m_b
    })
df_wilcoxon = pd.DataFrame(wilcoxon_records)

# ----------------- 4. 95% Bootstrap Confidence Intervals -----------------
print("Computing 95% Bootstrap Confidence Intervals (1,000 iterations)...")
np.random.seed(42)
ci_records = []
n_boot = 1000

for name in models:
    yt = y_true_test
    yp = test_preds[name]
    n_samples = len(yt)
    
    boot_r2, boot_rmse, boot_mae, boot_mape, boot_si = [], [], [], [], []
    for _ in range(n_boot):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        yt_b, yp_b = yt[idx], yp[idx]
        
        boot_r2.append(r2_score(yt_b, yp_b))
        r_val = np.sqrt(mean_squared_error(yt_b, yp_b))
        boot_rmse.append(r_val)
        boot_mae.append(np.mean(np.abs(yt_b - yp_b)))
        boot_mape.append(mean_absolute_percentage_error(yt_b, yp_b) * 100.0)
        boot_si.append(r_val / np.mean(yt_b))
        
    def get_ci_str(arr):
        low = np.percentile(arr, 2.5)
        high = np.percentile(arr, 97.5)
        mean_v = np.mean(arr)
        return f"{mean_v:.4f} [{low:.4f}, {high:.4f}]"

    ci_records.append({
        "Model Variant": name,
        "R2 (95% CI)": get_ci_str(boot_r2),
        "RMSE (95% CI)": get_ci_str(boot_rmse),
        "MAE (95% CI)": get_ci_str(boot_mae),
        "MAPE (%) (95% CI)": get_ci_str(boot_mape),
        "SI (95% CI)": get_ci_str(boot_si)
    })
df_ci = pd.DataFrame(ci_records)

# ----------------- 5. Save all to task/Data.xlsx -----------------
print("Saving sheets to task/Data.xlsx...")
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_pred_s1.to_excel(writer, sheet_name="predicts(With_Outlier)", index=False)
    df_pred_s2.to_excel(writer, sheet_name="predicts(Without_Outlier)", index=False)
    df_pred_test.to_excel(writer, sheet_name="predicts", index=False)
    df_oi.to_excel(writer, sheet_name="Overfitting_Index", index=False)
    df_wilcoxon.to_excel(writer, sheet_name="Wilcoxon_Test", index=False)
    df_ci.to_excel(writer, sheet_name="Confidence_Intervals", index=False)

print("[SUCCESS] All prediction catching, Overfitting Index, Wilcoxon test, and Confidence Interval sheets saved!")

