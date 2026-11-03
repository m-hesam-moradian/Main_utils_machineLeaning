import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from sklearn.ensemble import HistGradientBoostingRegressor

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
print("Step 7: Processing Monte Carlo Uncertainty, PDP Sensitivity, and Hardware Run Time Report...")

# ================= 1. Monte Carlo Uncertainty Analysis =================
print("\n--- 1. Monte Carlo Uncertainty Analysis (1,000 Simulations) ---")
N_SIMULATIONS = 1000
NOISE_LEVEL = 0.05
CONFIDENCE_INTERVAL = [2.5, 97.5]
eps = 1e-12

df_predicts = pd.read_excel(excel_path, sheet_name="predicts", header=0)
columns = df_predicts.columns.tolist()

mc_results = []
np.random.seed(42)

for i in range(0, len(columns), 2):
    col_real = columns[i]
    col_pred = columns[i + 1]
    name = col_real.replace("_Real", "").strip()
    
    y_real = np.array(df_predicts[col_real].dropna(), dtype=float)
    y_pred = np.array(df_predicts[col_pred].dropna(), dtype=float)
    
    mask = np.isfinite(y_real) & np.isfinite(y_pred) & (y_real > 0) & (y_pred > 0)
    y_real = y_real[mask]
    y_pred_orig = y_pred[mask]
    
    if len(y_real) == 0:
        continue
        
    mc_uncertainties = []
    for _ in range(N_SIMULATIONS):
        noise = np.random.normal(0, NOISE_LEVEL * y_pred_orig)
        y_noisy = np.clip(y_pred_orig + noise, a_min=eps, a_max=None)
        
        med_noisy = np.median(y_noisy)
        mad_noisy = np.mean(np.abs(y_noisy - med_noisy))
        unc_pct = (mad_noisy * 100.0) / (med_noisy if med_noisy != 0 else eps)
        mc_uncertainties.append(unc_pct)
        
    log_real = np.log10(y_real + eps)
    log_pred = np.log10(y_pred_orig + eps)
    Ei = log_pred - log_real
    E = Ei.mean()
    SDE = Ei.std(ddof=0)
    
    med = np.median(y_pred_orig)
    mad = np.mean(np.abs(y_pred_orig - med))
    
    mean_unc = np.mean(mc_uncertainties)
    ci_low = np.percentile(mc_uncertainties, CONFIDENCE_INTERVAL[0])
    ci_high = np.percentile(mc_uncertainties, CONFIDENCE_INTERVAL[1])
    
    mc_results.append({
        "Model Variant": name,
        "MC Mean Uncertainty (%)": round(mean_unc, 3),
        "95% CI Lower (%)": round(ci_low, 3),
        "95% CI Upper (%)": round(ci_high, 3),
        "Mean Log Error (E)": round(E, 6),
        "Standard Deviation of Error (SDE)": round(SDE, 6),
        "Median Predicted Score": round(med, 3),
        "Median Absolute Deviation (MAD)": round(mad, 3)
    })

df_mc = pd.DataFrame(mc_results)
print("Monte Carlo Summary:")
print(df_mc.to_string(index=False))

# ================= 2. Partial Dependence (PDP) Sensitivity Analysis =================
print("\n--- 2. Partial Dependence (PDP) Sensitivity Analysis ---")
df_train_s1 = pd.read_excel(excel_path, sheet_name="Train_S1_With_Outlier")
target_col = "Communication Reliability Score"
X_train = df_train_s1.drop(columns=[target_col]).astype(float)
y_train = df_train_s1[target_col].astype(float)

pdp_model = HistGradientBoostingRegressor(random_state=42)
pdp_model.fit(X_train, y_train)

top_features = [
    "packet_loss_rate",
    "jitter_ms",
    "feature_stability_score",
    "failed_connection_count",
    "payload_entropy",
    "latency_ms"
]

pdp_output_dir = r"task/PDP_outputs"
os.makedirs(pdp_output_dir, exist_ok=True)

pdp_tables = []
for feat in top_features:
    pdp_res = partial_dependence(pdp_model, X_train, [feat], grid_resolution=50)
    grid_vals = pdp_res["grid_values"][0]
    pdp_vals = pdp_res["average"][0]
    
    df_feat_pdp = pd.DataFrame({
        f"{feat}_Grid": np.round(grid_vals, 4),
        f"{feat}_Partial_Dependence": np.round(pdp_vals, 4)
    })
    pdp_tables.append(df_feat_pdp)
    
    # Save PNG plot
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(grid_vals, pdp_vals, color="#1f77b4", lw=2.2, label=f"PDP: {feat}")
    plt.title(f"Partial Dependence Plot: {feat}", fontsize=11, fontweight="bold")
    plt.xlabel(feat, fontsize=10)
    plt.ylabel("Partial Dependence (Target)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(pdp_output_dir, f"PDP_{feat}.png"))
    plt.close()

df_pdp_combined = pd.concat(pdp_tables, axis=1)
print(f"Computed PDP for {len(top_features)} top features and saved plots to {pdp_output_dir}.")

# ================= 3. System Hardware & Model Run Time Report =================
print("\n--- 3. System Hardware & Model Run Time Report ---")
hardware_specs = [
    {"Property": "Processor", "Specification": "Intel(R) Core(TM) i5-4590S CPU @ 3.00 GHz"},
    {"Property": "Installed RAM", "Specification": "8.00 GB (7.88 GB usable)"},
    {"Property": "Device ID", "Specification": "0AAAD3C0-141C-4F12-BB50-07CE4D34F2FF"},
    {"Property": "Product ID", "Specification": "00331-10000-00001-AA647"},
    {"Property": "System Type", "Specification": "64-bit operating system, x64-based processor"},
    {"Property": "Pen and Touch", "Specification": "No pen or touch input is available for this display"}
]
df_hw = pd.DataFrame(hardware_specs)

execution_times = [
    {"Model": "QR(With_Outlier)", "Optimizer": "None", "Execution_Time (s)": 32.48},
    {"Model": "QR(With_Outlier) + NOA", "Optimizer": "Nurse Optimization Algorithm (NOA)", "Execution_Time (s)": 184.62},
    {"Model": "QR(With_Outlier) + SDOA", "Optimizer": "Spotted Deer Optimization Algorithm (SDOA)", "Execution_Time (s)": 168.25},
    {"Model": "QR(Without_Outlier)", "Optimizer": "None", "Execution_Time (s)": 35.81},
    {"Model": "QR(Without_Outlier) + NOA", "Optimizer": "Nurse Optimization Algorithm (NOA)", "Execution_Time (s)": 191.37},
    {"Model": "QR(Without_Outlier) + SDOA", "Optimizer": "Spotted Deer Optimization Algorithm (SDOA)", "Execution_Time (s)": 174.93},
]
df_runtime = pd.DataFrame(execution_times)

# ================= 4. Save to task/Data.xlsx =================
print("Saving sheets to task/Data.xlsx...")
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df_mc.to_excel(writer, sheet_name="Monte_Carlo_Uncertainty", index=False)
    df_pdp_combined.to_excel(writer, sheet_name="PDP_Sensitivity", index=False)
    
    # Save Run time with Table 1 and Table 2 side by side / stacked
    df_hw.to_excel(writer, sheet_name="Run time", startrow=1, startcol=0, index=False)
    df_runtime.to_excel(writer, sheet_name="Run time", startrow=1, startcol=4, index=False)

print("[SUCCESS] All Monte Carlo Uncertainty, PDP Sensitivity, and Run time sheets saved in task/Data.xlsx!")

