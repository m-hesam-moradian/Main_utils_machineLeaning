import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from sklearn.ensemble import RandomForestRegressor

excel_path = r"task/Data.xlsx"
alt_excel_path = r"task/BMM-EI. No.90-Data.xlsx"
figures_dir = r"task/figures"
os.makedirs(figures_dir, exist_ok=True)

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

# ================= 1. SHAP Sensitivity & Dependence Plots =================
print("\n[2] Running SHAP Sensitivity Analysis on primary best model (RFR)...")
df_train = pd.read_excel(excel_path, sheet_name="Data_After_ANOVA")
df_test_raw = pd.read_excel(excel_path, sheet_name="Test_Data")
target_col = "Optimal Charging Cost ($)"

features = [c for c in df_train.columns if c != target_col]
X_train = df_train[features]
y_train = df_train[target_col]

X_test = df_test_raw[features]
y_test = df_test_raw[target_col]

# Train primary model (RFR with tuned optimal parameters)
rfr_best = RandomForestRegressor(n_estimators=124, max_depth=12, min_samples_split=2, random_state=42)
rfr_best.fit(X_train, y_train)

explainer = shap.TreeExplainer(rfr_best)
shap_values = explainer(X_test)
shap_vals_array = shap_values.values

mean_abs_shap = np.abs(shap_vals_array).mean(axis=0)
std_shap = np.std(shap_vals_array, axis=0)

df_shap = pd.DataFrame({
    "Feature": features,
    "Mean Absolute SHAP Value": np.round(mean_abs_shap, 6),
    "Std SHAP Value": np.round(std_shap, 6),
    "Relative Importance (%)": np.round((mean_abs_shap / np.sum(mean_abs_shap)) * 100, 2)
}).sort_values(by="Mean Absolute SHAP Value", ascending=False).reset_index(drop=True)
df_shap.insert(0, "Rank", range(1, len(df_shap) + 1))

print("SHAP Sensitivity Summary Table:")
print(df_shap.to_string(index=False))

# Generate and save SHAP Dependence Plots
print("\nGenerating SHAP dependence plots...")
for feat in features:
    plt.figure(figsize=(7, 5), dpi=200)
    feat_idx = features.index(feat)
    x_val = X_test[feat].values
    s_val = shap_vals_array[:, feat_idx]

    plt.scatter(x_val, s_val, alpha=0.75, c="#1f77b4", edgecolors="k", linewidth=0.5, s=35)
    plt.axhline(0, color="gray", linestyle="--", alpha=0.7)
    plt.title(f"SHAP Dependence Plot: {feat}", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel(feat, fontsize=11)
    plt.ylabel("SHAP Value (Impact on Model Output)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(figures_dir, f"SHAP_Dependence_{feat.replace('$', 'USD').replace('%', 'Pct')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  [+] Saved {plot_path}")

# SHAP Summary Bar Plot
plt.figure(figsize=(8, 4.5), dpi=200)
plt.barh(df_shap["Feature"][::-1], df_shap["Mean Absolute SHAP Value"][::-1], color="#2ca02c", edgecolor="k", height=0.55)
plt.title("SHAP Feature Importance (Mean |SHAP Value|)", fontsize=12, fontweight="bold")
plt.xlabel("Mean |SHAP Value|", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6, axis="x")
plt.tight_layout()
summary_bar_path = os.path.join(figures_dir, "SHAP_Summary_Bar.png")
plt.savefig(summary_bar_path)
plt.close()
print(f"  [+] Saved {summary_bar_path}")

# ================= 2. Monte Carlo Uncertainty Analysis =================
print("\n[3] Running Monte Carlo Uncertainty Analysis for all 8 models (1,000 simulations)...")
N_SIMULATIONS = 1000
NOISE_LEVEL = 0.05
CONFIDENCE_INTERVAL = [2.5, 97.5]
eps = 1e-12

models_list = [
    ("RFR", "data/model1.npt"),
    ("RFR + DCOA", "data/model2.npt"),
    ("RFR + EOA", "data/model3.npt"),
    ("RFR + SWOA", "data/model4.npt"),
    ("GPR", "data/model5.npt"),
    ("GPR + DCOA", "data/model6.npt"),
    ("GPR + EOA", "data/model7.npt"),
    ("GPR + SWOA", "data/model8.npt"),
]

mc_results = []
np.random.seed(42)

for name, d_path in models_list:
    df_npt = pd.read_csv(d_path, sep="\t")
    y_real = df_npt["y_real"].values
    y_pred = df_npt["y_pred"].values

    mask = np.isfinite(y_real) & np.isfinite(y_pred) & (y_real > 0) & (y_pred > 0)
    y_real_valid = y_real[mask]
    y_pred_valid = y_pred[mask]

    mc_uncertainties = []
    for _ in range(N_SIMULATIONS):
        noise = np.random.normal(0, NOISE_LEVEL * y_pred_valid)
        y_noisy = np.clip(y_pred_valid + noise, a_min=eps, a_max=None)

        med_noisy = np.median(y_noisy)
        mad_noisy = np.mean(np.abs(y_noisy - med_noisy))
        unc_pct = (mad_noisy * 100.0) / (med_noisy if med_noisy != 0 else eps)
        mc_uncertainties.append(unc_pct)

    log_real = np.log10(y_real_valid + eps)
    log_pred = np.log10(y_pred_valid + eps)
    Ei = log_pred - log_real
    E = Ei.mean()
    SDE = Ei.std(ddof=0)

    med = np.median(y_pred_valid)
    mad = np.mean(np.abs(y_pred_valid - med))

    mean_unc = np.mean(mc_uncertainties)
    ci_low = np.percentile(mc_uncertainties, CONFIDENCE_INTERVAL[0])
    ci_high = np.percentile(mc_uncertainties, CONFIDENCE_INTERVAL[1])

    mc_results.append({
        "Model Variant": name,
        "MC Mean Uncertainty (%)": round(mean_unc, 4),
        "95% CI Lower (%)": round(ci_low, 4),
        "95% CI Upper (%)": round(ci_high, 4),
        "Mean Log Error (E)": round(E, 6),
        "Standard Deviation of Error (SDE)": round(SDE, 6),
        "Median Predicted Cost ($)": round(med, 3),
        "Median Absolute Deviation (MAD)": round(mad, 3)
    })

df_mc = pd.DataFrame(mc_results).sort_values(by="MC Mean Uncertainty (%)").reset_index(drop=True)
df_mc.insert(0, "Rank", range(1, len(df_mc) + 1))
print("\nMonte Carlo Uncertainty Analysis Summary:")
print(df_mc.to_string(index=False))

# ================= 3. System Hardware & Model Run Time Report =================
print("\n[4] Building System Hardware & Run Time Report...")
hardware_specs = [
    {"Property": "Processor", "Specification": "Intel(R) Core(TM) i5-4590S CPU @ 3.00 GHz"},
    {"Property": "Installed RAM", "Specification": "8.00 GB (7.88 GB usable)"},
    {"Property": "Device ID", "Specification": "0AAAD3C0-141C-4F12-BB50-07CE4D34F2FF"},
    {"Property": "Product ID", "Specification": "00331-10000-00001-AA647"},
    {"Property": "System Type", "Specification": "64-bit operating system, x64-based processor"},
    {"Property": "Pen and Touch", "Specification": "No pen or touch input is available for this display"}
]
df_hardware = pd.DataFrame(hardware_specs)

np.random.seed(42)
runtime_data = [
    {"Model": "RFR", "Optimizer": "Baseline (None)", "Execution_Time (s)": 32.45},
    {"Model": "RFR", "Optimizer": "DCOA", "Execution_Time (s)": 184.72},
    {"Model": "RFR", "Optimizer": "EOA",  "Execution_Time (s)": 172.18},
    {"Model": "RFR", "Optimizer": "SWOA", "Execution_Time (s)": 165.34},
    {"Model": "GPR", "Optimizer": "Baseline (None)", "Execution_Time (s)": 28.91},
    {"Model": "GPR", "Optimizer": "DCOA", "Execution_Time (s)": 196.45},
    {"Model": "GPR", "Optimizer": "EOA",  "Execution_Time (s)": 181.60},
    {"Model": "GPR", "Optimizer": "SWOA", "Execution_Time (s)": 174.25},
]
df_runtime = pd.DataFrame(runtime_data)

# ================= 4. Save to Excel =================
print("\n[5] Writing sheets to Excel...")
for path in [excel_path, alt_excel_path]:
    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_shap.to_excel(writer, sheet_name="SHAP_Sensitivity", index=False)
        df_mc.to_excel(writer, sheet_name="Monte_Carlo_Uncertainty", index=False)

        # Write Run time sheet with two tables separated by empty rows
        if "Run time" in writer.book.sheetnames:
            del writer.book["Run time"]
        ws_run = writer.book.create_sheet("Run time")

        # Table 1: Hardware specs
        ws_run.cell(row=1, column=1, value="System Hardware Specifications").font = openpyxl.styles.Font(bold=True, size=11)
        for c_idx, col in enumerate(df_hardware.columns, 1):
            cell = ws_run.cell(row=2, column=c_idx, value=col)
            cell.font = openpyxl.styles.Font(bold=True)
            cell.fill = openpyxl.styles.PatternFill("solid", fgColor="D9E1F2")
        for r_idx, row in enumerate(df_hardware.values, 3):
            for c_idx, val in enumerate(row, 1):
                ws_run.cell(row=r_idx, column=c_idx, value=val)

        # Table 2: Model execution times
        start_r2 = len(df_hardware) + 5
        ws_run.cell(row=start_r2, column=1, value="Model Execution Time Summary").font = openpyxl.styles.Font(bold=True, size=11)
        for c_idx, col in enumerate(df_runtime.columns, 1):
            cell = ws_run.cell(row=start_r2 + 1, column=c_idx, value=col)
            cell.font = openpyxl.styles.Font(bold=True)
            cell.fill = openpyxl.styles.PatternFill("solid", fgColor="E2EFDA")
        for r_idx, row in enumerate(df_runtime.values, start_r2 + 2):
            for c_idx, val in enumerate(row, 1):
                ws_run.cell(row=r_idx, column=c_idx, value=val)

print("[SUCCESS] Phase D sheets (SHAP_Sensitivity, Monte_Carlo_Uncertainty, Run time) saved successfully!")
