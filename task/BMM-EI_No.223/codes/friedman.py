"""
friedman.py
===========
Friedman chi-square test (scipy.stats.friedmanchisquare) for all 3-way
combinations of model prediction vectors, taken from predicts({Tag}).

Uses high-precision mpmath to represent very small p-values that scipy
would otherwise underflow to 0.

Produces: Friedman_Stats({Tag}) sheet
Columns: Comparison | Statistic | P-Value
"""
import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import friedmanchisquare
import mpmath as mp

mp.mp.dps = 100  # 100 decimal digits of precision

DATA = r"/home/z/my-project/task_223/Data.xlsx"
TAGS = ["Chi2", "VIF"]


def format_p_value(p_value):
    if p_value is None or pd.isna(p_value):
        return "NaN"
    p_mp = mp.mpf(str(p_value))
    if p_mp == 0:
        return "0"
    exponent = int(mp.floor(mp.log10(p_mp)))
    mantissa = p_mp / mp.power(10, exponent)
    return f"{float(mantissa):.3f}E{exponent:+d}"


def high_precision_friedman_pvalue(statistic, num_models=3):
    df = num_models - 1
    s = mp.mpf(str(statistic))
    if df == 2:
        return mp.exp(-s / 2)
    return mp.gammainc(mp.mpf(df) / 2, s / 2, mp.inf, regularized=True)


for tag in TAGS:
    print(f"Running Friedman test for predicts({tag}) ...")
    df = pd.read_excel(DATA, sheet_name=f"predicts({tag})")
    columns = df.columns.tolist()

    predictions = {}
    for i in range(0, len(columns), 2):
        if i + 1 < len(columns):
            name = str(columns[i]).replace("_y_real", "")
            predictions[name] = df.iloc[:, i + 1].dropna().values

    min_len = min((len(v) for v in predictions.values()), default=0)
    for k in predictions:
        predictions[k] = predictions[k][:min_len]

    rows = []
    for a, b, c in combinations(predictions.keys(), 3):
        comp = f"{a} vs {b} vs {c}"
        try:
            statistic, _ = friedmanchisquare(predictions[a],
                                              predictions[b],
                                              predictions[c])
            hp_p = high_precision_friedman_pvalue(statistic, num_models=3)
            formatted_p = format_p_value(hp_p)
        except Exception:
            statistic, formatted_p = np.nan, "NaN"
        rows.append({
            "Comparison": comp,
            "Statistic":  f"{statistic:.5f}" if pd.notna(statistic) else "NaN",
            "P-Value":    formatted_p,
        })
        print(f"  {comp}: chi2={statistic:.4f}  p={formatted_p}")

    df_f = pd.DataFrame(rows)
    with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                         if_sheet_exists="replace") as writer:
        df_f.to_excel(writer, sheet_name=f"Friedman_Stats({tag})", index=False)
    print(f"[+] Saved: Friedman_Stats({tag})")
