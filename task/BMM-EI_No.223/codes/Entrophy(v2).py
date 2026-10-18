"""
Step 9 - Entropy-based Uncertainty Analysis
===========================================
Consumes Probs({Tag}) and computes normalized Shannon entropy per
prediction per model:
  H(p) = -sum(p_i * log(p_i)) / log(K)
where K = number of classes (4).

Produces two sheets per tag:
  - Entropy_Uncertainty({Tag}) : per-sample entropy for each model + y_pred
  - Entropy_Summary({Tag})     : mean & std entropy per model

Mirrors the No.219 Entrophy(v2).py reference.
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = r"/home/z/my-project/task_223/Data.xlsx"
TAGS = ["Chi2", "VIF"]

# ------------------------------------------------------------------
# Load Probs({Tag}) and split per-model blocks
# ------------------------------------------------------------------
def load_models_from_excel(path, sheet_name="Probs"):
    df = pd.read_excel(path, sheet_name=sheet_name)
    columns = df.columns.tolist()
    start_cols = [i for i, c in enumerate(columns)
                  if str(c).endswith("_y_real")]
    models = {}
    for idx, start in enumerate(start_cols):
        model_name = str(columns[start]).replace("_y_real", "")
        end = start_cols[idx + 1] if idx + 1 < len(start_cols) else len(columns)
        block = df.iloc[:, start:end].copy()
        rename = {}
        for c in block.columns:
            c_str = str(c)
            if c_str.endswith("_y_real"):
                rename[c] = "y_true"
            elif c_str.endswith("_y_pred"):
                rename[c] = "y_pred"
            elif "_prob_" in c_str:
                # convert {label}_prob_0 -> prob_0
                rename[c] = "prob_" + c_str.split("_prob_")[-1]
        block = block.rename(columns=rename)
        block = block.dropna(subset=["y_true"]).reset_index(drop=True)
        models[model_name] = block
    return models

# ------------------------------------------------------------------
# Normalized Shannon entropy
# ------------------------------------------------------------------
def normalized_entropy(probs):
    n_classes = probs.shape[1]
    if n_classes <= 1:
        return np.zeros(len(probs))
    probs = np.clip(probs, 1e-12, 1.0)
    ent = -np.sum(probs * np.log(probs), axis=1)
    return ent / np.log(n_classes)

# ------------------------------------------------------------------
# Run for each scenario tag
# ------------------------------------------------------------------
for tag in TAGS:
    print(f"\n{'=' * 70}\n  Entropy Uncertainty for tag: {tag}\n{'=' * 70}")
    sheet = f"Probs({tag})"

    models = load_models_from_excel(DATA, sheet_name=sheet)
    print(f"  Models detected: {list(models.keys())}")

    comparison_df = pd.DataFrame()
    avg_summary = []

    for model_name, df_m in models.items():
        prob_cols = [c for c in df_m.columns if str(c).startswith("prob_")]
        # Order prob cols by index
        prob_cols = sorted(prob_cols, key=lambda c: int(c.split("_")[-1]))
        probs = df_m[prob_cols].values.astype(float)
        uncertainty = normalized_entropy(probs)

        if comparison_df.empty:
            comparison_df["y_true"] = df_m["y_true"].astype(int)
        comparison_df[f"y_pred_{model_name}"] = df_m["y_pred"].astype(int)
        comparison_df[f"Uncertainty_{model_name}"] = uncertainty

        avg_summary.append({
            "Model": model_name,
            "Mean_Entropy_Uncertainty": float(np.mean(uncertainty)),
            "Std_Entropy_Uncertainty":  float(np.std(uncertainty)),
        })

    df_avg = pd.DataFrame(avg_summary)

    print(f"\n  Per-sample uncertainty matrix shape: {comparison_df.shape}")
    print(f"\n  Entropy Summary ({tag}):")
    print(df_avg.to_string(index=False))

    # Save both sheets
    out_unc  = f"Entropy_Uncertainty({tag})"
    out_sum  = f"Entropy_Summary({tag})"
    print(f"\n  Writing sheets: {out_unc}, {out_sum}")
    with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                         if_sheet_exists="replace") as writer:
        df_avg.to_excel(writer, sheet_name=out_sum, index=False)
        comparison_df.to_excel(writer, sheet_name=out_unc, index=False)

print("\n[+] Entropy uncertainty complete for both tags.")
