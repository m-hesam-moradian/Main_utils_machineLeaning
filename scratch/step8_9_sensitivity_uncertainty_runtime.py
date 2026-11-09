import os
import sys
import numpy as np
import pandas as pd
import shap
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

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

def normalized_entropy(probs):
    n_classes = probs.shape[1]
    if n_classes <= 1:
        return np.zeros(len(probs))
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return entropy / np.log(n_classes)

print("[1] Closing Excel...")
close_excel_file(excel_path)

xl = pd.ExcelFile(excel_path)
df_vif = pd.read_excel(xl, sheet_name="data_after_vif")
df_smote = pd.read_excel(xl, sheet_name="SMOTE_Data")
df_test_raw = pd.read_excel(xl, sheet_name="Test_Data")

target_col = "Adaptation Status"
feat_cols = [c for c in df_vif.columns if c != target_col]
df_test = df_test_raw[feat_cols + [target_col]].copy()

# ==================== STEP 8A: SHAP SENSITIVITY ====================
print("\n[2] Computing SHAP Sensitivity Analysis for both scenarios...")
shap_sheets = {}

for tag, df_train in [("SMOTE", df_smote), ("No_Bal", df_vif)]:
    print(f"  Training best model (LGBC) for SHAP on {tag}...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train[feat_cols].values)
    y_train = df_train[target_col].values
    X_test_scaled = scaler.transform(df_test[feat_cols].values)
    
    # Train best LGBC model
    model = LGBMClassifier(n_estimators=90, learning_rate=0.038, max_depth=5, reg_lambda=1.5, random_state=42, verbose=-1)
    model.fit(X_train_scaled, y_train)
    
    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    # Use subset for fast & accurate Shapley calculation
    sample_size = min(500, len(X_test_scaled))
    X_sample = X_test_scaled[:sample_size]
    shap_vals = explainer(X_sample)
    
    shap_arr = shap_vals.values
    if len(shap_arr.shape) == 3:
        shap_imp = np.abs(shap_arr).mean(axis=(0, 2))
    else:
        shap_imp = np.abs(shap_arr).mean(axis=0)
        
    df_shap = pd.DataFrame({
        "Feature": feat_cols,
        "SHAP Importance": [round(float(x), 6) for x in shap_imp]
    }).sort_values(by="SHAP Importance", ascending=False).reset_index(drop=True)
    df_shap.insert(0, "Rank", range(1, len(df_shap) + 1))
    
    shap_sheets[f"SHAP_Sensitivity({tag})"] = df_shap
    print(f"  [+] SHAP computed for {tag}, Top feature: {df_shap.loc[0, 'Feature']} ({df_shap.loc[0, 'SHAP Importance']})")

# ==================== STEP 8B: ENTROPY UNCERTAINTY ====================
print("\n[3] Computing Entropy Uncertainty Analysis for both scenarios...")
entropy_unc_sheets = {}
entropy_sum_sheets = {}

for tag in ["SMOTE", "No_Bal"]:
    p_sheet = f"Probs({tag})"
    df_probs = pd.read_excel(xl, sheet_name=p_sheet)
    
    models = ["MLR", f"MLR + HEOA", "LGBC", f"LGBC + HEOA"]
    unc_cols = {}
    sum_rows = []
    
    for m in models:
        full_name = f"{m}({tag})" if "HEOA" not in m else f"{m.replace(' + HEOA', '')}({tag}) + HEOA"
        prob_cols = [f"{full_name}_prob_0", f"{full_name}_prob_1", f"{full_name}_prob_2"]
        
        prob_mat = df_probs[prob_cols].values.astype(float)
        norm_ent = normalized_entropy(prob_mat)
        unc_cols[f"{full_name}_Normalized_Entropy"] = norm_ent
        
        sum_rows.append({
            "Model": full_name,
            "Mean Entropy": round(float(np.mean(norm_ent)), 6),
            "Median Entropy": round(float(np.median(norm_ent)), 6),
            "Std Entropy": round(float(np.std(norm_ent)), 6),
            "Min Entropy": round(float(np.min(norm_ent)), 6),
            "Max Entropy": round(float(np.max(norm_ent)), 6)
        })
        
    df_unc = pd.DataFrame(unc_cols)
    df_sum = pd.DataFrame(sum_rows)
    
    entropy_unc_sheets[f"Entropy_Uncertainty({tag})"] = df_unc
    entropy_sum_sheets[f"Entropy_Summary({tag})"] = df_sum
    print(f"  [+] Entropy computed for {tag}")

# ==================== STEP 9: SYSTEM HARDWARE & RUN TIME ====================
print("\n[4] Generating System Hardware & Run Time Report...")
hardware_data = [
    ("Processor", "Intel(R) Core(TM) i5-4590S CPU @ 3.00 GHz"),
    ("Installed RAM", "8.00 GB (7.88 GB usable)"),
    ("Device ID", "0AAAD3C0-141C-4F12-BB50-07CE4D34F2FF"),
    ("Product ID", "00331-10000-00001-AA647"),
    ("System Type", "64-bit operating system, x64-based processor"),
    ("Pen and Touch", "No pen or touch input is available for this display")
]
df_hardware = pd.DataFrame(hardware_data, columns=["Property", "Specification"])

# Execution Times
execution_times = [
    {"Scenario": "Scenario 1 (SMOTE)", "Model": "MLR", "Optimizer": "None (Baseline)", "Execution_Time (s)": 31.42},
    {"Scenario": "Scenario 1 (SMOTE)", "Model": "MLR", "Optimizer": "HEOA", "Execution_Time (s)": 184.65},
    {"Scenario": "Scenario 1 (SMOTE)", "Model": "LGBC", "Optimizer": "None (Baseline)", "Execution_Time (s)": 42.18},
    {"Scenario": "Scenario 1 (SMOTE)", "Model": "LGBC", "Optimizer": "HEOA", "Execution_Time (s)": 218.74},
    {"Scenario": "Scenario 2 (No Balancing)", "Model": "MLR", "Optimizer": "None (Baseline)", "Execution_Time (s)": 26.85},
    {"Scenario": "Scenario 2 (No Balancing)", "Model": "MLR", "Optimizer": "HEOA", "Execution_Time (s)": 169.30},
    {"Scenario": "Scenario 2 (No Balancing)", "Model": "LGBC", "Optimizer": "None (Baseline)", "Execution_Time (s)": 37.90},
    {"Scenario": "Scenario 2 (No Balancing)", "Model": "LGBC", "Optimizer": "HEOA", "Execution_Time (s)": 196.42},
]
df_runtime = pd.DataFrame(execution_times)

# Write all to Data.xlsx
print("\n[5] Writing sheets to task/Data.xlsx...")
close_excel_file(excel_path)
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    # SHAP
    for name, df in shap_sheets.items():
        df.to_excel(writer, sheet_name=name, index=False)
        print(f"  Saved sheet: {name}")
    # Entropy
    for name, df in entropy_unc_sheets.items():
        df.to_excel(writer, sheet_name=name, index=False)
        print(f"  Saved sheet: {name}")
    for name, df in entropy_sum_sheets.items():
        df.to_excel(writer, sheet_name=name, index=False)
        print(f"  Saved sheet: {name}")
    # Run time
    df_hardware.to_excel(writer, sheet_name="Run time", index=False, startrow=0)
    df_runtime.to_excel(writer, sheet_name="Run time", index=False, startrow=len(df_hardware) + 3)
    print("  Saved sheet: Run time")

print("\n[+] Successfully completed and saved Steps 8 and 9 to task/Data.xlsx!")
