"""
McNemar Test for all model pairs
=================================
Reads predicts(ENN).csv, computes McNemar chi-squared test
for all unique model pairs, saves McNemar(ENN).csv to task/.

McNemar tests whether two classifiers disagree significantly
on the same test set (uses contingency table of paired errors).
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import chi2
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
PREDICTS_CSV = r"D:\ML\task\predicts(ENN).csv"
OUT_CSV      = r"D:\ML\task\McNemar(ENN).csv"
ALPHA        = 0.05
# ──────────────────────────────────────────────────────────────────────────────

print(f"Loading predictions from '{PREDICTS_CSV}' ...")
df = pd.read_csv(PREDICTS_CSV)

# Extract model names and y_real/y_pred arrays
pred_cols  = [c for c in df.columns if c.endswith("_y_pred")]
model_names = [c.replace("_y_pred", "") for c in pred_cols]

# Get y_real (use first model's y_real — all should be identical)
y_real = df[f"{model_names[0]}_y_real"].values.astype(int)

predictions = {}
for name in model_names:
    predictions[name] = df[f"{name}_y_pred"].dropna().values.astype(int)

print(f"Models: {model_names}\n")

rows = []
for m_a, m_b in combinations(model_names, 2):
    p_a = predictions[m_a]
    p_b = predictions[m_b]
    n   = min(len(p_a), len(p_b), len(y_real))
    ya, yb, yr = p_a[:n], p_b[:n], y_real[:n]

    # Contingency table
    # b = A wrong, B right  |  c = A right, B wrong
    correct_a = (ya == yr)
    correct_b = (yb == yr)

    b = np.sum(~correct_a &  correct_b)   # A wrong, B right
    c = np.sum( correct_a & ~correct_b)   # A right, B wrong

    # McNemar statistic with continuity correction
    if (b + c) == 0:
        chi2_stat = 0.0
        p_value   = 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value   = 1 - chi2.cdf(chi2_stat, df=1)

    rows.append({
        "Comparison":           f"{m_a} vs {m_b}",
        "b (A wrong, B right)": int(b),
        "c (A right, B wrong)": int(c),
        "Chi2_statistic":       round(chi2_stat, 6),
        "P-Value":              f"{p_value:.6e}" if p_value < 0.001 else f"{p_value:.6f}",
        "Result (alpha=0.05)":  "Significant" if p_value < ALPHA else "Not Significant"
    })
    print(f"  {m_a} vs {m_b}: chi2={chi2_stat:.4f}  p={p_value:.4e}")

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print(df_out.to_string(index=False))
