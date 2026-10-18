"""
Statistical_t-test.py
=====================
Paired t-test (scipy.stats.ttest_rel) for all unique pairs of model
prediction vectors, taken from the predicts({Tag}) sheet.

Produces: Statistical_t-test({Tag}) sheet
Columns: Comparison | t-statistic | P-Value | Result (alpha=0.05)
"""
import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_rel

DATA = r"/home/z/my-project/task_223/Data.xlsx"
TAGS = ["Chi2", "VIF"]
ALPHA = 0.05

for tag in TAGS:
    print(f"Running paired t-test for predicts({tag}) ...")
    df = pd.read_excel(DATA, sheet_name=f"predicts({tag})")
    columns = df.columns.tolist()

    # Parse pairs (y_real, y_pred) per model
    predictions = {}
    for i in range(0, len(columns), 2):
        if i + 1 < len(columns):
            model_name = str(columns[i]).replace("_y_real", "")
            predictions[model_name] = df.iloc[:, i + 1].dropna().values

    results = {"stats": {}, "p_values": {}}
    for a, b in combinations(predictions.keys(), 2):
        try:
            pa = np.array(predictions[a], dtype=float)
            pb = np.array(predictions[b], dtype=float)
            n = min(len(pa), len(pb))
            valid = ~(np.isnan(pa[:n]) | np.isnan(pb[:n]))
            t_stat, p_val = ttest_rel(pa[:n][valid], pb[:n][valid])
        except Exception:
            t_stat, p_val = np.nan, np.nan
        comp = f"{a} vs {b}"
        results["stats"][comp]     = t_stat
        results["p_values"][comp]  = p_val

    df_t = pd.merge(
        pd.DataFrame(list(results["stats"].items()),
                      columns=["Comparison", "t-statistic"]),
        pd.DataFrame(list(results["p_values"].items()),
                      columns=["Comparison", "P-Value"]),
        on="Comparison")

    df_t["Result (alpha=0.05)"] = df_t["P-Value"].apply(
        lambda p: "NaN" if pd.isna(p)
                  else ("Significant" if float(p) < ALPHA
                        else "Not Significant"))
    df_t["P-Value"] = df_t["P-Value"].apply(
        lambda p: "NaN" if pd.isna(p)
                  else (f"{float(p):.6e}" if float(p) < 0.001
                        else f"{float(p):.6f}"))

    print(df_t.to_string(index=False))

    with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                         if_sheet_exists="replace") as writer:
        df_t.to_excel(writer, sheet_name=f"Statistical_t-test({tag})",
                       index=False)
    print(f"[+] Saved: Statistical_t-test({tag})")
