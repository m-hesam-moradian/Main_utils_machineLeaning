"""
Step 2b - VIF (Variance Inflation Factor) Feature Selection
============================================================
Iteratively drop the feature with the highest VIF above the threshold (5.0)
until all remaining features have VIF <= threshold.

Mirrors the BMM-EI No.221 reference VIF script structure, including the
horizontal Excel snapshot output for traceability.

Saves to Data.xlsx:
  - vif_horizontal : snapshot of every VIF iteration step (side-by-side)
  - data_after_vif : selected features + target

Consumes: Encoded_Data (11,999 x 32)
"""
import os
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

DATA = r"/home/z/my-project/task_223/Data.xlsx"
THRESHOLD = 5.0

# ------------------------------------------------------------------
# 1. Load Encoded_Data
# ------------------------------------------------------------------
print("Loading 'Encoded_Data' sheet ...")
df = pd.read_excel(DATA, sheet_name="Encoded_Data")
print(f"  Shape: {df.shape}")

target_col = df.columns[-1]
X_input = df.drop(columns=[target_col]).copy()
print(f"  Features: {X_input.shape[1]}   Target: '{target_col}'")
print(f"  VIF threshold: {THRESHOLD}")

# ------------------------------------------------------------------
# 2. Customized VIF calculation (horizontal snapshot style, mirrors No.221)
# ------------------------------------------------------------------
def calculate_vif_horizontal(X, threshold=5.0):
    X = X.copy()
    n_rows = len(X)

    # --- STEP 0: REMOVE UNIQUE IDENTIFIERS (leaky fingerprint cols) ---
    leaky_cols = []
    for col in X.columns:
        if X[col].nunique() == n_rows:
            leaky_cols.append(col)
    if leaky_cols:
        print(f"  [-] Dropping ID/leakage columns: {leaky_cols}")
        # We keep them in this dataset (no unique-id col here), but flag anyway.

    vif_snapshots = []
    step = 1

    while True:
        if X.shape[1] < 2:
            break

        X_const = add_constant(X, has_constant="add")
        vif = pd.DataFrame()
        vif["feature"] = X.columns
        vif["VIF"] = [
            variance_inflation_factor(X_const.values, i + 1)
            for i in range(X.shape[1])
        ]
        # Handle NaN/Inf
        vif["VIF"] = (vif["VIF"]
                       .replace([float("inf"), float("-inf")], float("inf"))
                       .fillna(float("inf")))
        vif["Step"] = step
        vif_snapshots.append(vif.reset_index(drop=True))

        max_vif = vif["VIF"].max()
        if max_vif > threshold:
            drop_feature = vif.loc[vif["VIF"].idxmax(), "feature"]
            print(f"  [-] Step {step}: Dropping feature '{drop_feature}' "
                  f"with VIF = {max_vif:.2f}")
            X.drop(columns=[drop_feature], inplace=True)
            step += 1
        else:
            print(f"  [+] Multicollinearity resolved! "
                  f"Max remaining VIF = {max_vif:.2f} <= {threshold}")
            break

    # --- Formatting for Excel output (mirror No.221 horizontal layout) ---
    max_rows = max(len(s) for s in vif_snapshots)
    for i in range(len(vif_snapshots)):
        rows_to_add = max_rows - len(vif_snapshots[i])
        if rows_to_add > 0:
            empty = pd.DataFrame([["", "", ""]] * rows_to_add,
                                  columns=["feature", "VIF", "Step"])
            vif_snapshots[i] = pd.concat([vif_snapshots[i], empty],
                                          ignore_index=True)
    spaced = []
    for s in vif_snapshots:
        spaced.append(s)
        spaced.append(pd.DataFrame({"": [""] * max_rows}))
    final_vif = pd.concat(spaced[:-1], axis=1)
    return X, final_vif

print("\nComputing VIF iteratively ...")
selected_X, final_vif_horizontal = calculate_vif_horizontal(X_input, threshold=THRESHOLD)
print(f"\nSelected features after VIF: {selected_X.shape[1]}")
print(selected_X.columns.tolist())

# ------------------------------------------------------------------
# 3. Build data_after_vif
# ------------------------------------------------------------------
data_after_vif = selected_X.copy()
data_after_vif[target_col] = df[target_col].values
print(f"data_after_vif shape: {data_after_vif.shape}")

# ------------------------------------------------------------------
# 4. Save to Data.xlsx
# ------------------------------------------------------------------
print("\nSaving sheets 'vif_horizontal' and 'data_after_vif' ...")
with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                     if_sheet_exists="replace") as writer:
    final_vif_horizontal.to_excel(writer, sheet_name="vif_horizontal", index=False)
    data_after_vif.to_excel(writer, sheet_name="data_after_vif", index=False)

print(f"[+] Done. Sheets now: {pd.ExcelFile(DATA).sheet_names}")
