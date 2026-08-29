import os
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def close_excel_file(filepath):
    try:
        import win32com.client
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            try:
                if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                    wb.Save()
                    wb.Close(SaveChanges=False)
                    print("[EXCEL] Saved and Closed Excel file:", filepath)
                    break
            except Exception:
                pass
    except Exception as e:
        print("Note: Excel COM:", e)

def open_excel_file(filepath):
    try:
        import win32com.client
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = True
        excel.Workbooks.Open(os.path.abspath(filepath))
        print("[EXCEL] Opened Excel file:", filepath)
    except Exception as e:
        print("Note: Excel COM Open:", e)

def calculate_vif_detailed(X_input, threshold=5.0):
    X = X_input.copy()
    vif_snapshots = []
    dropped_log = []
    step = 1

    while True:
        if X.shape[1] < 2:
            break

        X_const = add_constant(X, has_constant="add")
        vifs = [
            variance_inflation_factor(X_const.values, i + 1)
            for i in range(X.shape[1])
        ]

        vif = pd.DataFrame({
            "Feature": X.columns,
            "VIF": vifs,
            "Step": step
        })

        vif["VIF"] = (
            vif["VIF"]
            .replace([float("inf"), float("-inf")], float("inf"))
            .fillna(float("inf"))
        )

        vif_snapshots.append(vif.reset_index(drop=True))
        max_vif = vif["VIF"].max()

        if max_vif > threshold:
            drop_feature = vif.loc[vif["VIF"].idxmax(), "Feature"]
            dropped_log.append({"Step": step, "Dropped_Feature": drop_feature, "VIF": max_vif})
            print(f"  [Threshold {threshold}] Step {step}: Dropped '{drop_feature}' with VIF = {max_vif:.4f}")
            X.drop(columns=[drop_feature], inplace=True)
            step += 1
        else:
            print(f"  [Threshold {threshold}] Converged at Step {step}: Max VIF = {max_vif:.4f} <= {threshold}")
            break

    # Format horizontal snapshot
    if vif_snapshots:
        max_rows = max(len(s) for s in vif_snapshots)
        spaced_snapshots = []
        for s in vif_snapshots:
            s_copy = s.copy()
            if len(s_copy) < max_rows:
                empty = pd.DataFrame([["", "", ""]] * (max_rows - len(s_copy)), columns=["Feature", "VIF", "Step"])
                s_copy = pd.concat([s_copy, empty], ignore_index=True)
            spaced_snapshots.append(s_copy)
            spaced_snapshots.append(pd.DataFrame({"": [""] * max_rows}))
        horizontal_df = pd.concat(spaced_snapshots[:-1], axis=1)
    else:
        horizontal_df = pd.DataFrame()

    return X, horizontal_df, dropped_log, vif_snapshots[0] if vif_snapshots else pd.DataFrame()

def main():
    excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
    close_excel_file(excel_path)

    print("Loading Encoded_Data from task/Data.xlsx...")
    df = pd.read_excel(excel_path, sheet_name="Encoded_Data")
    target_column = df.columns[-1]
    X_input = df.drop(columns=[target_column])
    y_input = df[target_column]

    print(f"Initial features: {X_input.shape[1]}, Target: {target_column}")

    # Compute VIF across D1 (15), D2 (10), D3 (5), D4 (2)
    thresholds = [
        ("D1", 15.0),
        ("D2", 10.0),
        ("D3", 5.0),
        ("D4", 2.0),
    ]

    results_summary = []
    datasets = {}
    horizontal_dfs = {}

    for name, thresh in thresholds:
        print(f"\n--- Computing VIF for {name} (Threshold = {thresh}) ---")
        selected_X, horiz_df, dropped_log, initial_vif = calculate_vif_detailed(X_input, threshold=thresh)
        data_after = selected_X.copy()
        data_after[target_column] = y_input

        datasets[name] = data_after
        horizontal_dfs[name] = horiz_df

        results_summary.append({
            "Dataset": name,
            "VIF_Threshold": thresh,
            "Initial_Features": X_input.shape[1],
            "Dropped_Count": len(dropped_log),
            "Retained_Features": selected_X.shape[1],
            "Max_VIF_Remaining": initial_vif["VIF"].max() if not initial_vif.empty else 0.0,
            "Dropped_Features": ", ".join([d["Dropped_Feature"] for d in dropped_log]) if dropped_log else "None"
        })

    summary_df = pd.DataFrame(results_summary)
    print("\n=== VIF Summary ===")
    print(summary_df.to_string(index=False))

    # Save to Excel
    print("\nSaving results and datasets to task/Data.xlsx...")
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        summary_df.to_excel(writer, sheet_name="VIF_Summary", index=False)
        
        # Save baseline initial VIF table
        if not initial_vif.empty:
            initial_vif.sort_values(by="VIF", ascending=False).to_excel(writer, sheet_name="VIF_All_Features", index=False)
        
        # Save horizontal VIF report
        horizontal_dfs["D4"].to_excel(writer, sheet_name="vif_horizontal", index=False)

        # Save selected datasets for each threshold
        for name, d_df in datasets.items():
            d_df.to_excel(writer, sheet_name=f"{name}_Data", index=False)
        
        # Also save standard data_after_vif
        datasets["D4"].to_excel(writer, sheet_name="data_after_vif", index=False)

    print("[SUCCESS] All VIF analysis tables and datasets successfully saved into task/Data.xlsx")
    open_excel_file(excel_path)

if __name__ == "__main__":
    main()