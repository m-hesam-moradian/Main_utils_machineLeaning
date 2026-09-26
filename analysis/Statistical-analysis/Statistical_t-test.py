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

predicts_csv = r"D:\ML\task\predicts(ENN).csv"
out_csv      = r"D:\ML\task\Statistical_t-test(ENN).csv"

print(f"Loading predictions from '{predicts_csv}' ...")
df = pd.read_csv(predicts_csv)

# Dynamically extract model names and predictions
columns = df.columns.tolist()
structured_data = []

# CSV columns are: {model}_y_real, {model}_y_pred, {model}_y_real, ...
# Extract model name from _y_pred columns and use y_pred for t-test
pred_cols = [c for c in columns if str(c).endswith("_y_pred")]
for col in pred_cols:
    model_name = col.replace("_y_pred", "")
    y_pred = df[col].dropna().tolist()
    # find matching y_real column
    real_col = model_name + "_y_real"
    y_real = df[real_col].dropna().tolist() if real_col in columns else y_pred
    structured_data.append({"name": model_name, "y_real": y_real, "y_predict": y_pred})

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

def format_p_value(p):
    if pd.isna(p) or str(p).lower() == "nan":
        return "NaN"
    try:
        val = float(p)
        if val < 0.001:
            return f"{val:.6e}"
        else:
            return f"{val:.6f}"
    except Exception:
        return "NaN"

df_results["Result (alpha=0.05)"] = df_results["P-Value"].apply(check_significance)
df_results["P-Value"] = df_results["P-Value"].apply(format_p_value)


print("\nPaired T-Test Comparison Results:")
print(df_results)


# close_excel_file(excel_path)
df_results.to_csv(out_csv, index=False)
print(f"\nSaved t-test results to: {out_csv}")