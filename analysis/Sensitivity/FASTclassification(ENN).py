"""
FAST Sensitivity Analysis for Classification
=============================================
Reads feature data from task/Data.xlsx (data_after_chi2 sheet)
Reads model predictions from individual model CSVs in task/
Saves FAST_Sensitivity(ENN).csv to task/

Uses SALib FAST method to compute first-order (S1) and
total-order (ST) sensitivity indices per feature per model.
"""

import os
import time
import numpy as np
import pandas as pd
from SALib.analyze import fast

# ── CONFIG ────────────────────────────────────────────────────────────────────
EXCEL_PATH  = r"D:\ML\task\Data.xlsx"
TASK_DIR    = r"D:\ML\task"
DATA_SHEET  = "data_after_chi2"           # chi2-selected feature data
OUT_CSV     = os.path.join(TASK_DIR, "FAST_Sensitivity(ENN).csv")

MODEL_CSVS = {
    "KNNC":       os.path.join(TASK_DIR, "KNNC.csv"),
    "KNNC + ROA": os.path.join(TASK_DIR, "KNNC_ROA.csv"),
    "KNNC + CFOA":os.path.join(TASK_DIR, "KNNC_CFOA.csv"),
    "BC":         os.path.join(TASK_DIR, "BC.csv"),
    "BC + ROA":   os.path.join(TASK_DIR, "BC_ROA.csv"),
    "BC + CFOA":  os.path.join(TASK_DIR, "BC_CFOA.csv"),
}
# ──────────────────────────────────────────────────────────────────────────────

# ── Load feature data ─────────────────────────────────────────────────────────
print(f"Loading feature data from sheet '{DATA_SHEET}' ...")
df_data = pd.read_excel(EXCEL_PATH, sheet_name=DATA_SHEET)
target_col   = df_data.columns[-1]
X            = df_data.drop(columns=[target_col])
feature_names = list(X.columns)
D            = len(feature_names)
print(f"Features ({D}): {feature_names[:5]} ...")

# ── Define FAST problem ───────────────────────────────────────────────────────
problem = {
    "num_vars": D,
    "names":    feature_names,
    "bounds":   [[float(X[col].min()), float(X[col].max())] for col in feature_names]
}

# ── Run FAST per model ────────────────────────────────────────────────────────
all_results = []

for model_name, csv_path in MODEL_CSVS.items():
    print(f"\n[>] FAST: {model_name} ...", end=" ", flush=True)
    try:
        # Read y_pred from model CSV — it is in column 1 (0-indexed), starting from row 2
        # CSV layout: row0=title, row1=headers, row2+=data
        df_m = pd.read_csv(csv_path, header=None, skiprows=2)
        # column 0 = y_real, column 1 = y_pred
        y_pred = pd.to_numeric(df_m.iloc[:, 1], errors='coerce').dropna().values.astype(float)

        # Trim to nearest multiple of D for FAST requirement
        N         = len(y_pred)
        valid_len = (N // D) * D
        y_trim    = y_pred[:valid_len]

        start = time.time()
        Si    = fast.analyze(problem, y_trim, print_to_console=False)
        elapsed = time.time() - start

        df_res = pd.DataFrame({
            "Model":   model_name,
            "Feature": feature_names,
            "S1":      np.clip(Si["S1"],  0.0, 1.0).round(6),
            "S1_conf": np.abs(Si["S1_conf"]).round(6),
            "ST":      np.clip(Si["ST"],  0.0, 1.0).round(6),
            "ST_conf": np.abs(Si["ST_conf"]).round(6),
        })
        df_res = df_res.sort_values(by="ST", ascending=False).reset_index(drop=True)
        all_results.append(df_res)
        print(f"Done in {elapsed:.2f}s  (N_trim={valid_len})")

    except Exception as e:
        print(f"ERROR: {e}")

# ── Save ──────────────────────────────────────────────────────────────────────
if all_results:
    df_combined = pd.concat(all_results, ignore_index=True)
    df_combined.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Shape: {df_combined.shape}")
    print(df_combined.groupby("Model")[["S1","ST"]].mean().round(4))
else:
    print("No results generated.")
