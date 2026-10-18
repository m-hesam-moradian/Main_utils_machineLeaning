"""
DataCatcher.py
==============
Catches y_real and y_pred (NO probabilities) for all model variants
side-by-side into a single 'predicts({Tag})' sheet.

Consumes: model report sheets HGBC({Tag}), HGBC + PEOA({Tag}), ETC({Tag}),
          ETC + PEOA({Tag})  (or the .npt files)
Produces: predicts({Tag}) sheet with columns:
    {Model}_y_real | {Model}_y_pred  (per model variant)

This sheet feeds the Statistical_t-test and Friedman test analyses.
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
    print(f"Building predicts({tag}) ...")
    blocks = []
    for slot, label, key in SLOTS:
        npt_path = os.path.join(NPT_DIR, f"{tag}_{key}.npt")
        arr = np.load(npt_path, allow_pickle=False)
        y_real = arr[:, 0].astype(int)
        y_pred = arr[:, 1].astype(int)
        blocks.append(pd.DataFrame({
            f"{label}_y_real": y_real,
            f"{label}_y_pred": y_pred,
        }))
        print(f"  Loaded {label}: n={len(y_real)}")

    df = pd.concat(blocks, axis=1)
    print(f"  predicts({tag}) shape: {df.shape}")

    with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                         if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=f"predicts({tag})", index=False)
    print(f"  [+] Saved sheet: predicts({tag})")
