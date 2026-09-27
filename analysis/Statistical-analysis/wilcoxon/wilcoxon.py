import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from itertools import combinations

# Load structured data from Excel
excel_path = r"D:\ML\task\Data.xlsx"
xl = pd.ExcelFile(excel_path)
tags = ["No_SMOTE", "SMOTE"]

with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
    for tag in tags:
        sheet_name = f"predicts({tag})"
        out_sheet = f"Wilcoxon_Stats({tag})"
        
        if sheet_name not in xl.sheet_names:
            print(f"Skipping {tag}, sheet {sheet_name} not found.")
            continue
            
        print(f"\nLoading predictions from '{sheet_name}' ...")
        df = pd.read_excel(xl, sheet_name=sheet_name)
        
        columns = df.columns.tolist()
        structured_data = []

        for i in range(0, len(columns), 2):
            if i + 1 < len(columns):
                name = columns[i].strip()
                y_real = df.iloc[:, i].dropna().tolist()
                y_predict = df.iloc[:, i + 1].dropna().tolist()
                structured_data.append({"name": name, "y_real": y_real, "y_predict": y_predict})

        predictions = {entry["name"]: np.array(entry["y_predict"]) for entry in structured_data}
        results = {"stats": {}, "p_values": {}}

        for model_a, model_b in combinations(predictions.keys(), 2):
            try:
                pred_a = np.array(predictions[model_a], dtype=float)
                pred_b = np.array(predictions[model_b], dtype=float)
                min_len = min(len(pred_a), len(pred_b))
                pred_a = pred_a[:min_len]
                pred_b = pred_b[:min_len]
                stat, p_value = wilcoxon(pred_a, pred_b)
                results["stats"][f"{model_a} vs {model_b}"] = stat
                results["p_values"][f"{model_a} vs {model_b}"] = p_value
            except Exception as e:
                results["stats"][f"{model_a} vs {model_b}"] = np.nan
                results["p_values"][f"{model_a} vs {model_b}"] = np.nan
                print(f"Error comparing {model_a} vs {model_b}: {e}")

        df_stats = pd.DataFrame(results["stats"].items(), columns=["Comparison", "Statistic"])
        df_p_values = pd.DataFrame(results["p_values"].items(), columns=["Comparison", "P-Value"])
        df_results = pd.merge(df_stats, df_p_values, on="Comparison")

        alpha = 0.05
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
        print(f"[+] Saved Wilcoxon results to: {out_sheet}")