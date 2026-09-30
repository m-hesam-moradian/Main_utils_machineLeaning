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

excel_path = r"D:\ML\task\Data.xlsx"
xl = pd.ExcelFile(excel_path)
tags = ["Original"]

with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    for tag in tags:
        sheet_name = f"predicts({tag})"
        out_sheet = f"Statistical_t-test({tag})"
        
        if sheet_name not in xl.sheet_names:
            print(f"Skipping {tag}, sheet {sheet_name} not found.")
            continue
            
        print(f"Loading predictions from '{sheet_name}' ...")
        df = pd.read_excel(xl, sheet_name=sheet_name)
        
        columns = df.columns.tolist()
        structured_data = []

        # Read predictions columns in pairs (y_real, y_pred) if they are structured like that
        # Or if they are identical name like DataCatcher did: df_sub.columns = [f"{sheet}", f"{sheet}"]
        for i in range(0, len(columns), 2):
            if i + 1 < len(columns):
                model_name = str(columns[i]).strip()
                y_real = df.iloc[:, i].dropna().tolist()
                y_pred = df.iloc[:, i + 1].dropna().tolist()
                structured_data.append({"name": model_name, "y_real": y_real, "y_predict": y_pred})
        
        predictions = {entry["name"]: np.array(entry["y_predict"]) for entry in structured_data}
        results = {"stats": {}, "p_values": {}}
        alpha = 0.05

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
                return "Significant" if float(p) < alpha else "Not Significant"
            except Exception:
                return "NaN"

        def format_p_value(p):
            if pd.isna(p) or str(p).lower() == "nan":
                return "NaN"
            try:
                val = float(p)
                return f"{val:.6e}" if val < 0.001 else f"{val:.6f}"
            except Exception:
                return "NaN"

        df_results["Result (alpha=0.05)"] = df_results["P-Value"].apply(check_significance)
        df_results["P-Value"] = df_results["P-Value"].apply(format_p_value)

        df_results.to_excel(writer, sheet_name=out_sheet, index=False)
        print(f"[+] Saved t-test results to: {out_sheet}")