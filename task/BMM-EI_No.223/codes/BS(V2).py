"""
Step 6 - Brier Score Decomposition (BS V2)
==========================================
Reads Probs({Tag}) sheet, computes multiclass Brier score + reliability /
resolution / uncertainty decomposition for every model variant, and saves
the report to Brier_Decomposition({Tag}).

Mirrors the No.219 reference BS(V2).py logic:
  - brier_score_multiclass = mean_i sum_k (p_ik - y_ik)^2
  - per-class one-vs-rest binary decomposition (reliability, resolution, unc.)
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = r"/home/z/my-project/task_223/Data.xlsx"

SLOTS = ["HGBC", "HGBC + PEOA", "ETC", "ETC + PEOA"]
TAGS  = ["Chi2", "VIF"]
N_CLASSES = 4

# ------------------------------------------------------------------
# Load Probs({Tag}) and split into per-model blocks
# ------------------------------------------------------------------
def load_models_from_excel(path, sheet_name="Probs"):
    df = pd.read_excel(path, sheet_name=sheet_name)
    columns = df.columns.tolist()
    # Each model block starts with a "{label}_y_real" column
    start_cols = [i for i, c in enumerate(columns)
                  if str(c).endswith("_y_real")]
    models = {}
    for idx, start in enumerate(start_cols):
        model_name = str(columns[start]).replace("_y_real", "")
        end = start_cols[idx + 1] if idx + 1 < len(start_cols) else len(columns)
        block = df.iloc[:, start:end].copy()
        # Standardize column names
        rename = {}
        for c in block.columns:
            c_str = str(c)
            if c_str.endswith("_y_real"):
                rename[c] = "y_true"
            elif c_str.endswith("_y_pred"):
                rename[c] = "y_pred"
            elif "_prob_" in c_str:
                # keep as prob_0, prob_1, ...
                rename[c] = c_str.split("_prob_")[-1].join(["prob_", ""])
        block = block.rename(columns=rename)
        block = block.dropna(subset=["y_true"]).reset_index(drop=True)
        models[model_name] = block
    return models

# ------------------------------------------------------------------
# Brier score (multiclass)
# ------------------------------------------------------------------
def brier_score_multiclass(y_true, prob_matrix, labels):
    p = np.asarray(prob_matrix, dtype=float)
    n, K = p.shape
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx.get(int(v), -1) for v in y_true])
    if (y_idx == -1).any():
        raise ValueError("y_true contains labels not in labels list.")
    y_onehot = np.zeros_like(p)
    y_onehot[np.arange(n), y_idx] = 1.0
    return float(((p - y_onehot) ** 2).sum(axis=1).mean())

# ------------------------------------------------------------------
# Binary Brier decomposition (one-vs-rest for each class)
# ------------------------------------------------------------------
def brier_decomposition_binary(y_true, prob_pos, n_bins=10, strategy="quantile"):
    y = np.asarray(y_true).astype(float)
    p = np.asarray(prob_pos).astype(float)
    N = len(y)
    if strategy == "quantile":
        try:
            bins = pd.qcut(p, q=n_bins, duplicates="drop")
        except Exception:
            bins = pd.cut(p, bins=n_bins)
    else:
        bins = pd.cut(p, bins=n_bins)
    df = pd.DataFrame({"y": y, "p": p, "bin": bins})
    o = df["y"].mean()
    reliability = 0.0
    resolution = 0.0
    for _, g in df.groupby("bin"):
        ng = len(g)
        if ng == 0:
            continue
        p_g = g["p"].mean()
        o_g = g["y"].mean()
        reliability += (ng / N) * ((o_g - p_g) ** 2)
        resolution  += (ng / N) * ((o_g - o) ** 2)
    uncertainty = o * (1.0 - o)
    brier = reliability - resolution + uncertainty
    return {"brier": float(brier),
            "reliability": float(reliability),
            "resolution": float(resolution),
            "uncertainty": float(uncertainty)}

def brier_decomposition_multiclass(y_true, prob_matrix, labels,
                                     n_bins=10, strategy="quantile"):
    res = {}
    for i, lab in enumerate(labels):
        p_k = prob_matrix[:, i]
        y_k = (np.array(y_true) == lab).astype(float)
        res[lab] = brier_decomposition_binary(y_k, p_k, n_bins=n_bins,
                                                strategy=strategy)
    return res

# ------------------------------------------------------------------
# Run for both scenarios
# ------------------------------------------------------------------
for tag in TAGS:
    print(f"\n{'=' * 70}\n  Brier Decomposition for tag: {tag}\n{'=' * 70}")
    sheet = f"Probs({tag})"
    out_sheet = f"Brier_Decomposition({tag})"

    models = load_models_from_excel(DATA, sheet_name=sheet)
    print(f"  Detected models: {list(models.keys())}")

    all_reports = []
    for model_name, df_m in models.items():
        print(f"  Processing {model_name} ...")
        y_true = df_m["y_true"].astype(int).values
        prob_cols = [c for c in df_m.columns if str(c).startswith("prob_")]
        prob_matrix = df_m[prob_cols].values.astype(float)
        labels = list(range(N_CLASSES))

        # Overall multiclass Brier
        bs = brier_score_multiclass(y_true, prob_matrix, labels=labels)

        # Decomposition per class
        decomp = brier_decomposition_multiclass(y_true, prob_matrix,
                                                  labels=labels, n_bins=10)
        # Build a single-row report for this model
        row = {"Model": model_name, "Brier_multiclass": round(bs, 8)}
        for lab, comp in decomp.items():
            row[f"Brier_decomp_{lab}_brier"]        = round(comp["brier"], 8)
            row[f"Brier_decomp_{lab}_reliability"]  = round(comp["reliability"], 8)
            row[f"Brier_decomp_{lab}_resolution"]   = round(comp["resolution"], 8)
            row[f"Brier_decomp_{lab}_uncertainty"]  = round(comp["uncertainty"], 8)
        all_reports.append(row)

    final_report = pd.DataFrame(all_reports)
    print(f"\n  Brier Decomposition Report ({tag}):")
    print(final_report.to_string(index=False))

    # Save
    with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                         if_sheet_exists="replace") as writer:
        final_report.to_excel(writer, sheet_name=out_sheet, index=False)
    print(f"  [+] Saved sheet: {out_sheet}")

print("\n[+] Brier decomposition complete for both tags.")
