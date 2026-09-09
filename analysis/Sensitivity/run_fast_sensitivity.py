import os
import time
import numpy as np
import pandas as pd
from openpyxl import load_workbook
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

# Load selected features from Selected_Data_RFE
df_data = pd.read_excel(excel_path, sheet_name="Selected_Data_RFE")
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

# Load predictions from sheet 'predicts'
df_preds = pd.read_excel(excel_path, sheet_name="predicts")
model_cols = [c for c in df_preds.columns if not c.endswith("_y_real")]

# We evaluate FAST for each model
all_fast_results = []

# Get model names
xl = pd.ExcelFile(excel_path)
model_sheets = ["LDA", "LDA + HEOA", "LDA + KOA", "ETC", "ETC + HEOA", "ETC + KOA"]

for idx, model_name in enumerate(model_sheets):
    # Read y_pred for this model from sheet
    df_m = pd.read_excel(excel_path, sheet_name=model_name, header=1)
    y_pred = df_m["y_pred"].values.astype(float)
    
    # Trim to multiple of D
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
    
    # Sort by ST descending
    df_res = df_res.sort_values(by="ST", ascending=False)
    all_fast_results.append(df_res)
    print(f"[+] FAST completed for {model_name} in {end_t - start_t:.2f}s")

df_fast_combined = pd.concat(all_fast_results, ignore_index=True)

# Also create pivot/summary of ST across models
df_st_pivot = df_fast_combined.pivot(index="Feature", columns="Model", values="ST").reset_index()

print("\nFAST Sensitivity Summary (ST Pivot):")
print(df_st_pivot.head(10))

# Save to Excel
close_excel_file(excel_path)
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_fast_combined.to_excel(writer, sheet_name="FAST_Sensitivity", index=False)
    df_fast_combined.to_excel(writer, sheet_name="FAST", index=False)

print(f"\nSaved FAST Sensitivity results to sheets 'FAST_Sensitivity' and 'FAST' in {excel_path}")
