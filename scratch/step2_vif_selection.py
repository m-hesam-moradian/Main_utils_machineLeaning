import os
import sys
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

excel_path = r"task/Data.xlsx"

def close_excel_file(filepath):
    try:
        import win32com.client
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName).lower() == os.path.abspath(filepath).lower():
                wb.Save()
                wb.Close(SaveChanges=False)
                print("[*] Saved and Closed Excel file:", filepath)
                break
    except Exception as e:
        pass

def calculate_vif_horizontal(X, threshold=5.0):
    X = X.copy()
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

        vif["VIF"] = (
            vif["VIF"]
            .replace([float("inf"), float("-inf")], float("inf"))
            .fillna(float("inf"))
        )

        vif["Step"] = step
        vif_snapshots.append(vif.reset_index(drop=True))

        max_vif = vif["VIF"].max()

        if max_vif > threshold:
            drop_feature = vif.loc[vif["VIF"].idxmax(), "feature"]
            print(f"[-] Step {step}: Dropping feature '{drop_feature}' with VIF = {max_vif:.4f}")
            X = X.drop(columns=[drop_feature])
            step += 1
        else:
            print(f"[+] Multicollinearity resolved! Max remaining VIF = {max_vif:.4f} <= {threshold}")
            break

    # Formatting for Excel Output
    max_rows = max(len(df) for df in vif_snapshots)

    for i in range(len(vif_snapshots)):
        rows_to_add = max_rows - len(vif_snapshots[i])
        if rows_to_add > 0:
            empty_rows = pd.DataFrame(
                [["", "", ""]] * rows_to_add,
                columns=["feature", "VIF", "Step"]
            )
            vif_snapshots[i] = pd.concat([vif_snapshots[i], empty_rows], ignore_index=True)

    spaced_snapshots = []
    for df_snap in vif_snapshots:
        spaced_snapshots.append(df_snap)
        spaced_snapshots.append(pd.DataFrame({"": [""] * max_rows}))

    final_vif = pd.concat(spaced_snapshots[:-1], axis=1)

    return X, final_vif

print("[1] Closing Excel if open...")
close_excel_file(excel_path)

print("[2] Reading Train_Data sheet from", excel_path)
df_train = pd.read_excel(excel_path, sheet_name="Train_Data")
target_column = "Adaptation Status"
X_train = df_train.drop(columns=[target_column]).astype(float)

print(f"Initial Train_Data inputs: {X_train.shape[1]} features, {X_train.shape[0]} samples")

print("[3] Running VIF calculation (threshold = 5.0)...")
selected_X, final_vif_horizontal = calculate_vif_horizontal(X_train, threshold=5.0)

print(f"Remaining features after VIF: {selected_X.shape[1]}")

# Reattach target column
data_after_vif = selected_X.copy()
data_after_vif[target_column] = df_train[target_column].values

# Also create a concise VIF report summary
vif_last_step = final_vif_horizontal.iloc[:, -3:].dropna()
vif_last_step.columns = ["Feature", "Final VIF", "Step"]
vif_summary = vif_last_step[vif_last_step["Feature"] != ""].copy()
vif_summary["Final VIF"] = pd.to_numeric(vif_summary["Final VIF"])
vif_summary = vif_summary.sort_values(by="Final VIF", ascending=False).reset_index(drop=True)

# Write to Excel
print("[4] Writing results to task/Data.xlsx...")
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    final_vif_horizontal.to_excel(writer, sheet_name="vif_horizontal", index=False)
    data_after_vif.to_excel(writer, sheet_name="data_after_vif", index=False)
    vif_summary.to_excel(writer, sheet_name="VIF_Summary", index=False)

print("[+] Successfully saved 'vif_horizontal', 'data_after_vif', and 'VIF_Summary' sheets to task/Data.xlsx!")
