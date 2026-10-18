"""
DataCatcher_with_probability.py
===============================
Catches y_real, y_pred AND class probabilities for all model variants
side-by-side into a single 'Probs({Tag})' sheet.

Consumes: .npt files in data/ (each = [y_real, y_pred, prob_0..prob_3])
Produces: Probs({Tag}) sheet with columns per model:
    {Model}_y_real | {Model}_y_pred | {Model}_prob_0 | {Model}_prob_1 |
    {Model}_prob_2 | {Model}_prob_3

This sheet feeds the Brier Decomposition and Entropy Uncertainty analyses.
"""
import os
import numpy as np
import pandas as pd

DATA = r"/home/z/my-project/task_223/Data.xlsx"
NPT_DIR = r"/home/z/my-project/task_223/data"

SLOTS = [
    (1, "HGBC",          "model1"),
    (2, "HGBC + PEOA",   "model2"),
    (4, "ETC",           "model4"),
    (5, "ETC + PEOA",    "model5"),
]
TAGS = ["Chi2", "VIF"]

for tag in TAGS:
    print(f"Building Probs({tag}) ...")
    blocks = []
    for slot, label, key in SLOTS:
        npt_path = os.path.join(NPT_DIR, f"{tag}_{key}.npt")
        arr = np.load(npt_path, allow_pickle=False)
        y_real = arr[:, 0].astype(int)
        y_pred = arr[:, 1].astype(int)
        probs  = arr[:, 2:6].astype(float)
        blocks.append(pd.DataFrame({
            f"{label}_y_real":  y_real,
            f"{label}_y_pred":  y_pred,
            f"{label}_prob_0":  probs[:, 0],
            f"{label}_prob_1":  probs[:, 1],
            f"{label}_prob_2":  probs[:, 2],
            f"{label}_prob_3":  probs[:, 3],
        }))
        print(f"  Loaded {label}: n={len(y_real)}")

    df = pd.concat(blocks, axis=1)
    print(f"  Probs({tag}) shape: {df.shape}")

    with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                         if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=f"Probs({tag})", index=False)
    print(f"  [+] Saved sheet: Probs({tag})")
