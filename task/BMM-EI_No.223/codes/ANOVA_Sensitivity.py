"""
Step 8 - ANOVA Sensitivity Analysis
===================================
Per the task spec: 'Sensitivity Analysis: ANOVA'.

For each scenario tag (Chi2, VIF) and each model variant, compute a
one-way ANOVA F-statistic per feature (feature vs. y_pred), producing a
sensitivity ranking. Mirrors the No.204 ANOVA_FDR.py style but adapted
for sensitivity (feature -> model prediction) rather than feature
selection (feature -> y_true).

Saves sheet: ANOVA_Sensitivity({Tag})

Output columns:
  Feature, Model, F_statistic, p_value, sensitivity_rank
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

warnings.filterwarnings("ignore")

DATA = r"/home/z/my-project/task_223/Data.xlsx"
TAGS = ["Chi2", "VIF"]
SLOTS = ["HGBC", "HGBC + PEOA", "ETC", "ETC + PEOA"]

# ------------------------------------------------------------------
# Run for each scenario tag
# ------------------------------------------------------------------
for tag in TAGS:
    print(f"\n{'=' * 70}\n  ANOVA Sensitivity for tag: {tag}\n{'=' * 70}")

    # Source features: use the data_after_{tag.lower()} sheet
    src_sheet = f"data_after_{tag.lower()}"
    print(f"  Loading feature sheet: {src_sheet}")
    df_data = pd.read_excel(DATA, sheet_name=src_sheet).dropna()
    target_col = df_data.columns[-1]
    X_df = df_data.drop(columns=[target_col])
    feature_names = X_df.columns.tolist()
    X = X_df.values
    print(f"  Features: {X.shape[1]}  |  Samples: {X.shape[0]}")

    # Load predicts({Tag}) sheet to get each model's y_pred
    df_pred = pd.read_excel(DATA, sheet_name=f"predicts({tag})")
    columns = df_pred.columns.tolist()

    # Parse pairs (y_real, y_pred) per model
    model_preds = {}
    for i in range(0, len(columns), 2):
        if i + 1 < len(columns):
            model_name = str(columns[i]).replace("_y_real", "")
            y_pred = df_pred.iloc[:, i + 1].dropna().astype(int).values
            model_preds[model_name] = y_pred

    all_rows = []
    for model_name, y_pred in model_preds.items():
        print(f"\n  Computing ANOVA F-stat for model: {model_name}")
        # Need to align X with y_pred length (predicts has 2400 = 20% test set)
        # Use only the test-portion rows (last 20% of X)
        n_test = len(y_pred)
        X_test = X[-n_test:]
        try:
            F_values, p_values = f_classif(X_test, y_pred)
        except Exception as e:
            print(f"    Warning: f_classif failed ({e}); using zeros.")
            F_values = np.zeros(X_test.shape[1])
            p_values = np.ones(X_test.shape[1])

        rows = pd.DataFrame({
            "Feature":    feature_names,
            "Model":      model_name,
            "F_statistic": F_values,
            "p_value":    p_values,
        })
        # Rank within model (1 = most sensitive)
        rows["sensitivity_rank"] = (
            rows["F_statistic"].rank(ascending=False, method='min')
            .astype(int))
        all_rows.append(rows)
        # Print top 5
        top5 = rows.sort_values("F_statistic", ascending=False).head(5)
        print(top5.to_string(index=False))

    final = pd.concat(all_rows, ignore_index=True)
    out_sheet = f"ANOVA_Sensitivity({tag})"
    with pd.ExcelWriter(DATA, mode="a", engine="openpyxl",
                         if_sheet_exists="replace") as writer:
        final.to_excel(writer, sheet_name=out_sheet, index=False)
    print(f"\n  [+] Saved sheet: {out_sheet}  shape={final.shape}")

print("\n[+] ANOVA sensitivity complete for both tags.")
