import os
import time
import numpy as np
import pandas as pd
from SALib.analyze import fast
import win32com.client

def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
close_excel_file(excel_path)

xl = pd.ExcelFile(excel_path)
# Detect feature dataset sheet
data_sheet = "Selected_Data_RFE" if "Selected_Data_RFE" in xl.sheet_names else ("SMOTE_Data" if "SMOTE_Data" in xl.sheet_names else "Data")

# Load selected features
df_data = pd.read_excel(excel_path, sheet_name=data_sheet)
target_col = df_data.columns[-1]
X = df_data.drop(columns=[target_col])
feature_names = list(X.columns)
D = len(feature_names)

# Define problem for FAST
problem = {
    "num_vars": D,
    "names": feature_names,
    "bounds": [[float(X[col].min()), float(X[col].max())] for col in feature_names]
}

# Detect evaluated model sheets
ignore_sheets = [
    "Data", "Encoded_Data", "RFE_Report", "Selected_Data_RFE", "SMOTE_Data", "Balanced_Data",
    "Probs", "Probs(RFE)", "Probs(SMOTE)", "Brier_Decomposition", "Brier_Decomposition(RFE)",
    "predicts", "Statistical_t-test", "Entropy_Summary", "Entropy_Uncertainty",
    "FAST_Sensitivity", "FAST", "Run time"
]

model_sheets = [
    s for s in xl.sheet_names
    if s not in ignore_sheets
    and not s.endswith("_Metrics(RFE)")
    and not s.endswith("_Metrics(SMOTE)")
    and not s.startswith("Data_after_KFold_")
    and not s.startswith("Model_Comparison_")
]

print("Running FAST Sensitivity for models:", model_sheets)

all_fast_results = []

for model_name in model_sheets:
    try:
        df_m = pd.read_excel(excel_path, sheet_name=model_name, header=1)
        if "y_pred" not in df_m.columns:
            continue
        y_pred = df_m["y_pred"].values.astype(float)
        
        # Trim to nearest multiple of D
        N = len(y_pred)
        valid_len = (N // D) * D
        y_pred_trim = y_pred[:valid_len]
        
        start_t = time.time()
        Si = fast.analyze(problem, y_pred_trim, print_to_console=False)
        end_t = time.time()
        
        df_res = pd.DataFrame({
            "Model": model_name,
            "Feature": problem["names"],
            "S1": np.clip(Si["S1"], 0.0, 1.0).round(6),
            "S1_conf": np.abs(Si["S1_conf"]).round(6),
            "ST": np.clip(Si["ST"], 0.0, 1.0).round(6),
            "ST_conf": np.abs(Si["ST_conf"]).round(6)
        })
        
        df_res = df_res.sort_values(by="ST", ascending=False)
        all_fast_results.append(df_res)
        print(f"[+] FAST completed for {model_name} in {end_t - start_t:.2f}s")
    except Exception as e:
        print(f"[-] Error running FAST for {model_name}: {e}")

if all_fast_results:
    df_fast_combined = pd.concat(all_fast_results, ignore_index=True)
    
    close_excel_file(excel_path)
    with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        df_fast_combined.to_excel(writer, sheet_name="FAST_Sensitivity", index=False)
        df_fast_combined.to_excel(writer, sheet_name="FAST", index=False)
        
    print(f"\nSaved FAST Sensitivity report to sheets 'FAST_Sensitivity' and 'FAST' in {excel_path}")