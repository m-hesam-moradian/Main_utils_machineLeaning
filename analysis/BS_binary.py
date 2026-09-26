"""
Binary Brier Score & Decomposition
===================================
Works with y_real + y_pred ONLY (no probabilities needed).
For hard binary predictions, p_hat = y_pred (0 or 1).

Metrics per model:
  - Brier Score      = mean((y_pred - y_true)^2)  == error rate for hard labels
  - Reliability      = calibration component
  - Resolution       = sharpness component
  - Uncertainty      = base rate variance  o*(1-o)

Reads 'Probs(ENN)' sheet (columns: {model}_y_real, {model}_y_pred)
Saves 'Brier_Decomposition(ENN)' sheet into task/Data.xlsx
"""

import numpy as np
import pandas as pd
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────
excel_path = r"D:\ML\task\Data.xlsx"
probs_sheet   = "Probs(ENN)"
out_sheet     = "Brier_Decomposition(ENN)"
N_BINS        = 10
# ──────────────────────────────────────────────────────────────────────────────


def brier_score_binary(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    """Brier Score = mean squared error between probability and true label."""
    return float(np.mean((p_hat.astype(float) - y_true.astype(float)) ** 2))


def brier_decompose_binary(y_true: np.ndarray, p_hat: np.ndarray,
                           n_bins: int = 10) -> dict:
    """
    Murphy (1973) decomposition:
        BS = Reliability - Resolution + Uncertainty

    For hard predictions (p_hat in {0,1}), bins are just 0-group and 1-group.
    """
    y = y_true.astype(float)
    p = p_hat.astype(float)
    N = len(y)

    o_bar = y.mean()                          # overall base rate
    uncertainty = o_bar * (1.0 - o_bar)      # irreducible component

    # bin by unique p values (0 and 1 for hard predictions)
    unique_pvals = np.unique(p)
    reliability = 0.0
    resolution  = 0.0

    for pval in unique_pvals:
        idx = p == pval
        n_k = idx.sum()
        o_k = y[idx].mean()                  # observed frequency in bin
        reliability += (n_k / N) * (pval  - o_k) ** 2
        resolution  += (n_k / N) * (o_k   - o_bar) ** 2

    bs = brier_score_binary(y, p)
    return {
        "Brier_Score":   round(bs,          8),
        "Reliability":   round(reliability, 8),
        "Resolution":    round(resolution,  8),
        "Uncertainty":   round(uncertainty, 8),
        "BS_check":      round(reliability - resolution + uncertainty, 8),
    }


# ─── LOAD ─────────────────────────────────────────────────────────────────────
print(f"Reading '{probs_sheet}' from {excel_path} ...")
df_probs = pd.read_excel(excel_path, sheet_name=probs_sheet)
print(f"Shape: {df_probs.shape}")
print("Columns:", df_probs.columns.tolist()[:8], "...")

# ─── DETECT MODELS ────────────────────────────────────────────────────────────
cols = df_probs.columns.tolist()
real_cols = [c for c in cols if str(c).endswith("_y_real")]
model_names = [c.replace("_y_real", "") for c in real_cols]
print(f"\nDetected models: {model_names}")

# ─── COMPUTE ──────────────────────────────────────────────────────────────────
rows = []
for name in model_names:
    y_real_col = f"{name}_y_real"
    y_pred_col = f"{name}_y_pred"

    y_true = df_probs[y_real_col].dropna().astype(int).values
    y_pred = df_probs[y_pred_col].dropna().astype(int).values

    # align lengths (should be equal)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    decomp = brier_decompose_binary(y_true, y_pred, n_bins=N_BINS)

    row = {"Model": name, "N": n}
    row.update(decomp)
    rows.append(row)
    print(f"  {name}: BS={decomp['Brier_Score']:.6f}  "
          f"Rel={decomp['Reliability']:.6f}  "
          f"Res={decomp['Resolution']:.6f}  "
          f"Unc={decomp['Uncertainty']:.6f}")

df_out = pd.DataFrame(rows)

# ─── SAVE ─────────────────────────────────────────────────────────────────────
with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl",
                    if_sheet_exists="replace") as writer:
    df_out.to_excel(writer, sheet_name=out_sheet, index=False)

print(f"\nSaved '{out_sheet}' to {excel_path}")
print(df_out.to_string(index=False))
