"""
Entropy-Based Uncertainty Analysis (V2) — CSV version
======================================================
Reads Probs(ENN).csv (y_real, y_pred, prob_0, prob_1 per model)
Computes normalized Shannon entropy: H(p) = -sum(p * log(p)) / log(K)
Saves:
  Entropy_Uncertainty(ENN).csv  — per-sample uncertainty per model
  Entropy_Summary(ENN).csv      — mean & std per model
"""

import numpy as np
import pandas as pd
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROBS_CSV  = r"D:\ML\task\Probs(ENN).csv"
OUT_UNCERT = r"D:\ML\task\Entropy_Uncertainty(ENN).csv"
OUT_SUMM   = r"D:\ML\task\Entropy_Summary(ENN).csv"
# ──────────────────────────────────────────────────────────────────────────────

def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """H(p) = -sum(p_i * log(p_i)) / log(K)  — normalized to [0, 1]"""
    K = probs.shape[1]
    if K <= 1:
        return np.zeros(len(probs))
    probs = np.clip(probs, 1e-12, 1.0)
    H = -np.sum(probs * np.log(probs), axis=1)
    return H / np.log(K)

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading '{PROBS_CSV}' ...")
df = pd.read_csv(PROBS_CSV)
print(f"Shape: {df.shape}")

# Detect models from column names ending in _y_real
real_cols   = [c for c in df.columns if c.endswith("_y_real")]
model_names = [c.replace("_y_real", "") for c in real_cols]
print(f"Models: {model_names}\n")

# ── Compute uncertainty ───────────────────────────────────────────────────────
comparison_df = pd.DataFrame()
summary_rows  = []

for name in model_names:
    print(f"  Processing {name} ...", end=" ")
    y_true = df[f"{name}_y_real"].values
    y_pred = df[f"{name}_y_pred"].values

    # Build prob matrix from prob_0, prob_1 columns
    prob_cols = [c for c in df.columns if c.startswith(f"{name}_prob_")]
    probs = df[prob_cols].values.astype(float)

    uncertainty = normalized_entropy(probs)

    if comparison_df.empty:
        comparison_df["y_true"] = y_true.astype(int)

    comparison_df[f"y_pred_{name}"]       = y_pred.astype(int)
    comparison_df[f"Uncertainty_{name}"]  = uncertainty.round(6)

    mean_h = float(np.mean(uncertainty))
    std_h  = float(np.std(uncertainty))
    summary_rows.append({
        "Model":                    name,
        "Mean_Entropy_Uncertainty": round(mean_h, 6),
        "Std_Entropy_Uncertainty":  round(std_h,  6),
    })
    print(f"Mean H={mean_h:.4f}  Std={std_h:.4f}")

df_summary = pd.DataFrame(summary_rows)

# ── Save ──────────────────────────────────────────────────────────────────────
comparison_df.to_csv(OUT_UNCERT, index=False)
df_summary.to_csv(OUT_SUMM, index=False)

print(f"\nSaved: {OUT_UNCERT}")
print(f"Saved: {OUT_SUMM}")
print("\nEntropy Summary:")
print(df_summary.to_string(index=False))
