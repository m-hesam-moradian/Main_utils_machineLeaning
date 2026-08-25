import os
import numpy as np
import pandas as pd
import win32com.client
from pathlib import Path

def close_excel_file(filepath):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
        for wb in excel.Workbooks:
            if os.path.abspath(wb.FullName) == os.path.abspath(filepath):
                wb.Save()
                wb.Close(SaveChanges=False)
                print("Saved and Closed Excel file:", filepath)
                break
    except Exception:
        pass

# -------------------------------------------------------
# Load probability sheet and automatically split models
# -------------------------------------------------------
def load_models_from_excel(path, sheet_name="Probs"):
    df = pd.read_excel(path, sheet_name=sheet_name)
    models = {}
    columns = df.columns.tolist()

    # Find model start columns
    start_cols = [
        i for i, c in enumerate(columns)
        if str(c).endswith("_y_real")
    ]

    for idx, start in enumerate(start_cols):
        # Model name
        model_name = str(columns[start]).replace("_y_real", "")

        # End of this model block
        if idx + 1 < len(start_cols):
            end = start_cols[idx + 1]
        else:
            end = len(columns)

        block = df.iloc[:, start:end].copy()

        # Rename columns
        rename = {}
        for c in block.columns:
            c_str = str(c)
            if c_str.endswith("_y_real"):
                rename[c] = "y_true"
            elif c_str.endswith("_y_pred"):
                rename[c] = "y_pred"
            elif "prob" in c_str.lower():
                rename[c] = c_str

        block = block.rename(columns=rename)

        # Keep only valid rows
        block = block.dropna(subset=["y_true"]).reset_index(drop=True)
        models[model_name] = block

    return models

# -------------------------------------------------------
# Normalized entropy
# -------------------------------------------------------
def normalized_entropy(probs):
    n_classes = probs.shape[1]
    if n_classes <= 1:
        return np.zeros(len(probs))

    probs = np.clip(probs, 1e-12, 1)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return entropy / np.log(n_classes)

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    excel_path = r"C:\Users\Sam\Desktop\ML\task\Data.xlsx"
    close_excel_file(excel_path)

    xl = pd.ExcelFile(excel_path)
    if "Probs(RFE)" in xl.sheet_names:
        sheet_name = "Probs(RFE)"
        out_sheet = "Entropy_Uncertainty"
        sum_sheet = "Entropy_Summary"
    elif "Probs(ENN)" in xl.sheet_names:
        sheet_name = "Probs(ENN)"
        out_sheet = "Entropy_Uncertainty"
        sum_sheet = "Entropy_Summary"
    else:
        sheet_name = "Probs"
        out_sheet = "Entropy_Uncertainty"
        sum_sheet = "Entropy_Summary"


    print(f"Loading probability sheet '{sheet_name}'...")
    models = load_models_from_excel(excel_path, sheet_name=sheet_name)

    print("\nDetected Models:")
    for m in models:
        print("-", m)

    comparison_df = pd.DataFrame()
    avg_summary = []

    for model_name, df_m in models.items():
        print(f"\nProcessing {model_name}")
        prob_cols = [c for c in df_m.columns if "prob" in c.lower()]
        probs = df_m[prob_cols].values
        uncertainty = normalized_entropy(probs)

        if comparison_df.empty:
            comparison_df["y_true"] = df_m["y_true"]

        comparison_df[f"y_pred_{model_name}"] = df_m["y_pred"]
        comparison_df[f"Uncertainty_{model_name}"] = uncertainty
        
        avg_summary.append({
            "Model": model_name,
            "Mean_Entropy_Uncertainty": float(np.mean(uncertainty)),
            "Std_Entropy_Uncertainty": float(np.std(uncertainty))
        })

    df_avg = pd.DataFrame(avg_summary)

    print("\nSample Uncertainty Matrix:")
    print(comparison_df.head())
    print("\nAverage Entropy Uncertainty Summary:")
    print(df_avg.to_string(index=False))

    # Save results to Excel sheet 'Entropy_Uncertainty(ENN)' and 'Entropy_Summary(ENN)'
    close_excel_file(excel_path)
    with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        df_avg.to_excel(writer, sheet_name=sum_sheet, index=False)
        comparison_df.to_excel(writer, sheet_name=out_sheet, index=False)
        df_avg.to_excel(writer, sheet_name="Entropy_Summary", index=False)
        comparison_df.to_excel(writer, sheet_name="Entropy_Uncertainty", index=False)

    print(f"\nSaved Entropy Uncertainty report to sheet '{out_sheet}' and '{sum_sheet}' in {excel_path}")

if __name__ == "__main__":
    main()