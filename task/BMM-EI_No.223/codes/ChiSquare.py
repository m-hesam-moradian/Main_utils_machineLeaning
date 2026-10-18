"""
Step 2a - Chi-Square Feature Selection
======================================
Apply sklearn.feature_selection.chi2 to the Encoded_Data sheet,
report chi2 statistic + p-value for every feature, and select the
top features whose p-value < 0.05 (standard chi2 threshold used in
BMM-EI No.219 reference pipeline).

Saves to Data.xlsx:
  - chi2_report : feature | chi2_stat | p_value | selected
  - data_after_chi2 : selected features + target

Consumes: Encoded_Data (11,999 x 32)
"""
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2

DATA = r"/home/z/my-project/task_223/Data.xlsx"

# ------------------------------------------------------------------
# 1. Load Encoded_Data
# ------------------------------------------------------------------
print("Loading 'Encoded_Data' sheet ...")
df = pd.read_excel(DATA, sheet_name="Encoded_Data")
print(f"  Shape: {df.shape}")

target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"  Features: {X.shape[1]}   Target: '{target_col}'   Classes: {sorted(y.unique())}")

# ------------------------------------------------------------------
# 2. Compute chi2 statistic + p-value
#    chi2 requires non-negative features. All our features are already
#    non-negative (counts, percentages, encoded categorical 0..N).
#    Verify and clip just in case.
# ------------------------------------------------------------------
print("\nComputing Chi-Square statistics ...")
X_nonneg = X.clip(lower=0)
chi2_stats, chi2_pvalues = chi2(X_nonneg, y)

report_df = pd.DataFrame({
    "feature": X.columns,
    "chi2_stat": chi2_stats,
    "p_value": chi2_pvalues,
})
# Significance flag at alpha=0.05
report_df["selected"] = report_df["p_value"] < 0.05
report_df = report_df.sort_values("p_value").reset_index(drop=True)

print("\nChi-Square Report (sorted by p-value ascending):")
print(report_df.to_string(index=False))

n_selected = report_df["selected"].sum()
print(f"\nSelected features (p<0.05): {n_selected} / {len(report_df)}")

# ------------------------------------------------------------------
# 3. Build data_after_chi2 (selected features + target)
# ------------------------------------------------------------------
selected_features = report_df.loc[report_df["selected"], "feature"].tolist()
data_after_chi2 = pd.concat([X[selected_features], y], axis=1)
print(f"data_after_chi2 shape: {data_after_chi2.shape}")
print(f"Selected features: {selected_features}")

# ------------------------------------------------------------------
# 4. Save to Data.xlsx
# ------------------------------------------------------------------
print("\nSaving sheets 'chi2_report' and 'data_after_chi2' ...")
with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                     if_sheet_exists="replace") as writer:
    report_df.to_excel(writer, sheet_name="chi2_report", index=False)
    data_after_chi2.to_excel(writer, sheet_name="data_after_chi2", index=False)

print(f"[+] Done. Sheets now: {pd.ExcelFile(DATA).sheet_names}")
