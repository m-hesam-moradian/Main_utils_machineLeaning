import pandas as pd
import numpy as np
import os
import win32com.client
from scipy.stats import ttest_rel
from itertools import combinations

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
sheet_name = "predicts(SMOTE)" if "predicts(SMOTE)" in xl.sheet_names else "predicts"
print(f"Loading predictions sheet '{sheet_name}'...")

df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)

# Dynamically extract model names and predictions
columns = df.columns.tolist()
structured_data = []

for i in range(0, len(columns), 2):
    name = columns[i].strip()
    y_real = df.iloc[:, i].tolist()
    y_predict = df.iloc[:, i + 1].tolist()
    structured_data.append({"name": name, "y_real": y_real, "y_predict": y_predict})

# Build prediction dictionary for the T-test
predictions = {entry["name"]: np.array(entry["y_predict"]) for entry in structured_data}

results = {
    "stats": {},
    "p_values": {},
}

alpha = 0.05

# Perform Paired T-Test (ttest_rel) for all unique model pairs
for model_a, model_b in combinations(predictions.keys(), 2):
    try:
        pred_a = np.array(predictions[model_a], dtype=float)
        pred_b = np.array(predictions[model_b], dtype=float)
        min_len = min(len(pred_a), len(pred_b))
        pred_a = pred_a[:min_len]
        pred_b = pred_b[:min_len]
        valid = ~(np.isnan(pred_a) | np.isnan(pred_b))
        t_stat, p_value = ttest_rel(pred_a[valid], pred_b[valid])
        results["stats"][f"{model_a} vs {model_b}"] = t_stat
        results["p_values"][f"{model_a} vs {model_b}"] = p_value
    except Exception as e:
        results["stats"][f"{model_a} vs {model_b}"] = np.nan
        results["p_values"][f"{model_a} vs {model_b}"] = np.nan
        print(f"Error comparing {model_a} vs {model_b}: {e}")

df_stats = pd.DataFrame(results["stats"].items(), columns=["Comparison", "t-statistic"])
df_p_values = pd.DataFrame(results["p_values"].items(), columns=["Comparison", "P-Value"])

df_results = pd.merge(df_stats, df_p_values, on="Comparison")

def check_significance(p):
    if pd.isna(p) or str(p).lower() == "nan":
        return "NaN"
    try:
        val = float(p)
        return "Significant" if val < alpha else "Not Significant"
    except Exception:
        return "NaN"

df_results["Result (alpha=0.05)"] = df_results["P-Value"].apply(check_significance)
df_results["P-Value"] = df_results["P-Value"].apply(lambda x: f"{float(x):.15f}" if pd.notna(x) and not np.isnan(float(x)) else "NaN")

print("\nPaired T-Test Comparison Results:")
print(df_results)

# Save to Excel sheet 'Statistical_t-test(SMOTE)' or 'Statistical_t-test'
close_excel_file(excel_path)
out_sheet = "Statistical_t-test(SMOTE)" if "SMOTE_Data" in xl.sheet_names else "Statistical_t-test"
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    df_results.to_excel(writer, sheet_name=out_sheet, index=False)

print(f"\nSaved Paired T-Test results to sheet '{out_sheet}' in {excel_path}")